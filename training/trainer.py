"""Training engine (Phase 5/6): resumable checkpoints, resource-aware metrics and CPU fallback."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from dataset.packed import PackedTokenDataset, epoch_indices
from model.config import MiniCoderConfig, load_config
from model.transformer import MiniCoder7B
from providers.detector import detect_environment
from training.checkpoint import CheckpointManager
from training.config import EarlyStopper, TrainConfig, cosine_lr_lambda, load_train_config
from training.cost_guard import EstimatedCostGuard
from training.dist_utils import all_reduce, barrier, cleanup, init_distributed
from training.metrics import MetricsLogger
from training.resource_manager import ResourceManager


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(req: str, local_rank: int) -> torch.device:
    if req == "cpu" or (req == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available")
    return torch.device("cuda", local_rank)


def resolve_precision(req: str, device: torch.device):
    """-> (applied_name, dtype). CPU always fp32; bf16 falls back to fp16 on GPUs without bf16."""
    if device.type != "cuda" or req == "fp32":
        return "fp32", torch.float32
    if req == "bf16" and torch.cuda.is_bf16_supported():
        return "bf16", torch.bfloat16
    return "fp16", torch.float16


class ControlFiles:
    """STOP / PAUSE files in run_dir + SIGINT/SIGTERM -> safe stop. (The dashboard will write these files.)"""

    def __init__(self, run_dir: Path):
        self.stop_file, self.pause_file, self.checkpoint_file, self.flag = run_dir / "STOP", run_dir / "PAUSE", run_dir / "CHECKPOINT", False
        for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
            if sig is not None:
                try:
                    signal.signal(sig, lambda *_: setattr(self, "flag", True))
                except (ValueError, OSError):
                    pass  # not main thread / unsupported

    def should_stop(self) -> bool:
        return self.flag or self.stop_file.exists()

    def wait_if_paused(self) -> None:
        while self.pause_file.exists() and not self.should_stop():
            time.sleep(1.0)

    def should_checkpoint(self) -> bool:
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink(missing_ok=True)
            return True
        return False


class Trainer:
    def __init__(self, model_cfg: MiniCoderConfig, tcfg: TrainConfig, train_ds, val_ds=None,
                 extra_meta: Optional[Dict[str, Any]] = None, cost_confirmed: bool = False):
        self.mcfg, self.tcfg, self.train_ds, self.val_ds = model_cfg, tcfg, train_ds, val_ds
        # Refuse configured paid runs before initializing CUDA or allocating a model.
        self.cost_guard = EstimatedCostGuard(
            tcfg.gpu_cost_rate_usd_per_hour, tcfg.max_budget_usd,
            confirmed=cost_confirmed, reserve_seconds=tcfg.cost_guard_reserve_seconds)
        self.dist = init_distributed()
        self.device = resolve_device(tcfg.device, self.dist.local_rank)
        self.run_dir = Path(tcfg.run_dir); self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log = MetricsLogger(tcfg.run_dir, enabled=self.dist.is_main)
        seed_everything(tcfg.seed)
        if tcfg.gradient_checkpointing is not None:
            model_cfg.gradient_checkpointing = tcfg.gradient_checkpointing
        self.prec_name, self.amp_dtype = resolve_precision(tcfg.precision or model_cfg.precision, self.device)
        self.model = MiniCoder7B(model_cfg).to(self.device)
        self.auto_tune_report = None
        if tcfg.auto_batch_size:
            from training.auto_tune import probe_auto_batch_size
            self.auto_tune_report = probe_auto_batch_size(
                self.model, self.train_ds, tcfg.micro_batch_size, tcfg.grad_accum,
                tcfg.seq_len, tcfg.max_auto_micro_batch_size, tcfg.auto_tune_steps, self.device)
            chosen = self.auto_tune_report.get("recommendation")
            if chosen:
                tcfg.micro_batch_size = chosen["micro_batch_size"]
                tcfg.grad_accum = chosen["grad_accum"]
            if self.dist.is_main:
                self.log.log("auto_batch_size_probe", **self.auto_tune_report)
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            (no_decay if p.ndim < 2 or "embed" in n else decay).append(p)
        groups = [{"params": decay, "weight_decay": tcfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
        try:
            self.opt = AdamW(groups, lr=tcfg.learning_rate, betas=(tcfg.beta1, tcfg.beta2), eps=1e-8,
                             fused=self.device.type == "cuda")
        except (TypeError, RuntimeError):
            self.opt = AdamW(groups, lr=tcfg.learning_rate, betas=(tcfg.beta1, tcfg.beta2), eps=1e-8)
        self.sched = LambdaLR(self.opt, lambda s: cosine_lr_lambda(s, tcfg.warmup_steps, tcfg.max_steps,
                                                                    tcfg.min_lr_ratio))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_dtype == torch.float16)
        self.raw_model = self.model
        self.environment = detect_environment(torch_module=torch)
        self.last_checkpoint_path = None
        self.stopper = EarlyStopper(tcfg.early_stopping_patience, tcfg.early_stopping_min_delta)
        self.control = ControlFiles(self.run_dir)
        self.tokens_per_step = tcfg.micro_batch_size * tcfg.grad_accum * tcfg.seq_len * self.dist.world_size
        self.applied = {"model": asdict(model_cfg), "train": asdict(tcfg), "device": str(self.device),
                        "precision_applied": self.prec_name, "world_size": self.dist.world_size,
                        "dist_note": self.dist.note, "params": self.raw_model.num_parameters(),
                        "provider_name": self.environment.provider, "gpu_name": self.environment.gpu_name or "N/A",
                        "gpu_vram_total_bytes": self.environment.vram_total_bytes,
                        "gpu_vram_free_bytes": self.environment.vram_free_bytes,
                        "tokens_per_step": self.tokens_per_step, **(extra_meta or {})}
        self.checkpoint_metadata = {
            "model_config": asdict(model_cfg),
            # Runtime controls may change on resume (for example max_steps=8 -> 10).
            # Data/model/optimization semantics remain strict below.
            "train_config": {k: v for k, v in asdict(tcfg).items()
                             if k not in {"resume", "force_resume", "max_steps", "run_dir",
                                          "checkpoint_dir", "eval_every", "eval_batches",
                                          "save_every", "log_every"}},
            "tokenizer_fingerprint": (extra_meta or {}).get("tokenizer_fingerprint"),
            "tokenizer_config": (extra_meta or {}).get("tokenizer_config"),
            "dataset_version": (extra_meta or {}).get("dataset_version"),
        }
        self.ckpt = CheckpointManager(tcfg.checkpoint_dir or str(self.run_dir / "checkpoints"),
                                      tcfg.keep_checkpoints, metadata=self.checkpoint_metadata)
        self.resources = ResourceManager(self.device)
        if self.dist.is_main:
            if self.auto_tune_report is not None:
                self.applied["auto_batch_size_probe"] = self.auto_tune_report
            (self.run_dir / "applied_config.json").write_text(json.dumps(self.applied, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ data
    def _epoch_batches(self, epoch: int) -> List[List[int]]:
        idx = epoch_indices(len(self.train_ds), self.tcfg.seed, epoch, self.dist.rank, self.dist.world_size)
        mb = self.tcfg.micro_batch_size
        batches = [idx[i * mb:(i + 1) * mb].tolist() for i in range(len(idx) // mb)]
        if len(batches) < self.tcfg.grad_accum:
            raise ValueError(f"train set too small: {len(self.train_ds)} windows of {self.tcfg.seq_len} tokens give "
                             f"{len(batches)} micro-batches/rank, need >= grad_accum={self.tcfg.grad_accum}. "
                             "Add data or reduce seq_len/micro_batch_size/grad_accum.")
        return batches

    # ------------------------------------------------------------------ eval
    @torch.no_grad()
    def evaluate(self) -> Optional[Dict[str, float]]:
        if self.val_ds is None or len(self.val_ds) == 0:
            return None
        self.raw_model.eval()
        mb, world = self.tcfg.micro_batch_size, self.dist.world_size
        n = min(len(self.val_ds), self.tcfg.eval_batches * mb * world)
        mine = list(range(n))[self.dist.rank::world]
        loss_sum = torch.zeros((), device=self.device, dtype=torch.float64)
        count = torch.zeros((), device=self.device, dtype=torch.float64)
        for i in range(0, len(mine), mb):
            xs, ys = zip(*[self.val_ds[j] for j in mine[i:i + mb]])
            x, y = torch.stack(xs).to(self.device), torch.stack(ys).to(self.device)
            with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype != torch.float32):
                logits = self.raw_model(x)
            loss_sum += F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1),
                                        reduction="sum").double()
            count += y.numel()
        all_reduce(loss_sum, self.dist); all_reduce(count, self.dist)
        self.raw_model.train()
        if count.item() == 0:
            return None
        loss = loss_sum.item() / count.item()
        return {"val_loss": loss, "val_ppl": math.exp(min(loss, 20.0)), "val_tokens": int(count.item())}

    # ------------------------------------------------------------------ one optimizer update
    def _update(self, group) -> Optional[Dict[str, float]]:
        n = len(group)
        loss_total = torch.zeros((), device=self.device)
        for i, (x, y) in enumerate(group):
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            sync = self.model.no_sync() if (self.dist.enabled and i < n - 1) else nullcontext()
            with sync:
                with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype != torch.float32):
                    logits = self.model(x)
                loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
                self.scaler.scale(loss / n).backward()   # every micro-batch contributes loss/n -> true mean
            loss_total += loss.detach()
        self.scaler.unscale_(self.opt)
        gn = torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), self.tcfg.grad_clip)
        if not torch.isfinite(gn):
            if self.scaler.is_enabled():
                self.scaler.step(self.opt); self.scaler.update()   # scaler skips the step and lowers its scale
            self.opt.zero_grad(set_to_none=True)
            return None
        self.scaler.step(self.opt); self.scaler.update()
        self.opt.zero_grad(set_to_none=True)
        self.sched.step()
        return {"loss": (loss_total / n).item(), "grad_norm": float(gn)}

    def _save(self, step: int, loss: float, epoch: int, batch_in_epoch: int,
              val_loss: Optional[float] = None) -> None:
        if self.dist.is_main:
            try:
                path = self.ckpt.save(self.raw_model, self.opt, self.sched, step, loss, epoch, self.scaler,
                                      batch_in_epoch, val_loss=val_loss, tokens_seen=step * self.tokens_per_step)
            except Exception as exc:
                self.log.log("checkpoint_failed", step=step, exception_type=type(exc).__name__)
                raise
            self.last_checkpoint_path = path
            self.log.log("checkpoint", step=step, path=path)
        barrier(self.dist)

    def _stop_requested(self) -> bool:
        requested = self.control.should_stop() or getattr(self, "_termination_signal", None) is not None
        flag = torch.tensor(float(requested), device=self.device)
        return bool(all_reduce(flag, self.dist, op=torch.distributed.ReduceOp.MAX if self.dist.enabled else None).item())

    # ------------------------------------------------------------------ main loop
    def train(self) -> Dict[str, Any]:
        old_handlers = {}
        self._termination_signal = None
        def request_stop(signum, _frame):
            self._termination_signal = int(signum)
        for signum in (signal.SIGTERM, signal.SIGINT):
            try:
                old_handlers[signum] = signal.signal(signum, request_stop)
            except (ValueError, OSError):
                pass  # Python only permits signal handler changes from the main thread.
        try:
            return self._train_impl()
        finally:
            for signum, handler in old_handlers.items():
                try:
                    signal.signal(signum, handler)
                except (ValueError, OSError):
                    pass

    def _train_impl(self) -> Dict[str, Any]:
        tc = self.tcfg
        state = {"step": 0, "epoch": 0, "batch_in_epoch": 0, "loss": None}
        if tc.resume:
            state = self.ckpt.load_latest(self.raw_model, self.opt, self.sched, self.scaler, device="cpu",
                                          expected_metadata=self.checkpoint_metadata,
                                          force_resume=tc.force_resume)
        self.last_checkpoint_path = state.get("checkpoint_path") or self.last_checkpoint_path
        step, epoch, start_batch = state["step"], state["epoch"], state.get("batch_in_epoch", 0)
        tokens_seen = int(state.get("tokens_seen") or step * self.tokens_per_step)
        start_step = step
        if self.dist.enabled:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.model = DDP(self.raw_model, device_ids=[self.dist.local_rank] if self.device.type == "cuda" else None)
        self.model.train()
        self.log.log("start", step=step, params=self.applied["params"], precision=self.prec_name,
                     global_step=step, tokens_seen=tokens_seen,
                     total_tokens_target=tc.max_steps * self.tokens_per_step,
                     provider_name=self.environment.provider, gpu_name=self.environment.gpu_name or "N/A",
                     gpu_vram_total_bytes=self.environment.vram_total_bytes,
                     gpu_vram_free_bytes=self.environment.vram_free_bytes,
                     device=str(self.device), world_size=self.dist.world_size,
                     timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        t0 = tw = time.time(); steps_w = 0; last_loss = state.get("loss") or float("nan")
        last_step_seconds = None
        best_val, reason, skipped_run = float("inf"), "max_steps", 0
        latest_val = None
        cur_batch = start_batch
        try:
            while step < tc.max_steps and reason == "max_steps":
                batches = self._epoch_batches(epoch)
                start_batch -= start_batch % tc.grad_accum
                loader = DataLoader(self.train_ds, batch_sampler=batches[start_batch:], num_workers=tc.num_workers,
                                    pin_memory=self.device.type == "cuda")
                it, consumed = iter(loader), start_batch
                cur_batch = start_batch
                for _ in range((len(batches) - start_batch) // tc.grad_accum):
                    self.control.wait_if_paused()
                    if self._stop_requested():
                        reason = "termination_signal" if self._termination_signal else "stop_requested"
                        if self._termination_signal:
                            self.log.log("termination_signal", step=step, signal=self._termination_signal)
                        break
                    if self.cost_guard.should_stop_before_next_step(last_step_seconds):
                        reason = "estimated_gpu_cost_budget_reached"
                        self.log.log("cost_guard_stop", step=step, **self.cost_guard.report())
                        break
                    group = [next(it) for _ in range(tc.grad_accum)]
                    consumed += tc.grad_accum
                    cur_batch = consumed
                    batch_ready_time = self.resources.before_batch()
                    out = self._update(group)
                    resource_snapshot = self.resources.after_batch(batch_ready_time)
                    last_step_seconds = resource_snapshot.get("step_seconds")
                    if out is None:
                        skipped_run += 1
                        self.log.log("skipped_step", step=step, reason="non_finite_grad")
                        if skipped_run > 50:
                            raise RuntimeError("50 consecutive non-finite gradient steps; training diverged")
                        continue
                    skipped_run = 0; step += 1; steps_w += 1; last_loss = out["loss"]
                    if step % tc.log_every == 0 or step == 1:
                        now = time.time(); dt = max(1e-9, now - tw)
                        rec = {"step": step, "epoch": epoch, "loss": round(out["loss"], 5),
                               "grad_norm": round(out["grad_norm"], 4), "lr": self.opt.param_groups[0]["lr"],
                               "global_step": step, "tokens_seen": step * self.tokens_per_step,
                               "total_tokens_target": tc.max_steps * self.tokens_per_step,
                               "training_loss": round(out["loss"], 5), "learning_rate": self.opt.param_groups[0]["lr"],
                               "validation_loss": latest_val,
                               "provider_name": self.environment.provider,
                               "gpu_name": self.environment.gpu_name or "N/A",
                               "gpu_vram_total_bytes": self.environment.vram_total_bytes,
                               "gpu_vram_free_bytes": self.environment.vram_free_bytes,
                               "gpu_memory_used_bytes": resource_snapshot.get("gpu_memory_used_bytes"),
                               "gpu_memory_total_bytes": resource_snapshot.get("gpu_memory_total_bytes"),
                               "gpu_utilization_percent": resource_snapshot.get("gpu_utilization_percent"),
                               "cpu_ram_used_bytes": resource_snapshot.get("ram_used_bytes"),
                               "cpu_ram_total_bytes": resource_snapshot.get("ram_total_bytes"),
                               "data_wait_seconds": resource_snapshot.get("data_wait_seconds"),
                               "step_seconds": resource_snapshot.get("step_seconds"),
                               "micro_batch_size": tc.micro_batch_size, "grad_accum": tc.grad_accum,
                               "sequence_length": tc.seq_len, "trainable_parameters": self.applied["params"],
                               "optimizer": "AdamW",
                               "checkpoint_path": self.last_checkpoint_path,
                               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                               "tokens_per_s": round(steps_w * self.tokens_per_step / dt, 1),
                               "tokens_per_second": round(steps_w * self.tokens_per_step / dt, 1),
                               "steps_per_s": round(steps_w / dt, 4)}
                        completed = max(1, step - start_step)
                        rec["estimated_remaining_seconds"] = round(
                            max(0, tc.max_steps - step) * max(0.0, now - t0) / completed, 1)
                        rec["estimated_gpu_compute_cost_usd"] = round(self.cost_guard.estimated_cost_usd(), 6)
                        if self.device.type == "cuda":
                            rec["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated(self.device) / 1024 ** 3, 2)
                            rec["gpu_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(self.device))
                            try: rec["gpu_util"] = torch.cuda.utilization(self.device)
                            except Exception: pass
                        self.log.log("train", **rec); tw, steps_w = now, 0
                        if self.dist.is_main: print(json.dumps(rec))
                    if tc.eval_every and step % tc.eval_every == 0:
                        ev = self.evaluate()
                        if ev:
                            latest_val = ev["val_loss"]
                            best_val = min(best_val, ev["val_loss"]); self.log.log("eval", step=step, **ev)
                            if self.dist.is_main: print(json.dumps({"step": step, **ev}))
                            if self.stopper.update(ev["val_loss"]):
                                reason = "early_stopping"; break
                    if step % tc.save_every == 0:
                        self._save(step, last_loss, epoch, consumed, latest_val)
                    elif self.control.should_checkpoint():
                        self._save(step, last_loss, epoch, consumed, latest_val)
                    if step >= tc.max_steps:
                        break
                else:
                    epoch += 1; start_batch = 0; cur_batch = 0   # epoch finished normally
                    continue
                break   # stop requested / early stopping / max_steps reached mid-epoch
        except torch.cuda.OutOfMemoryError:
            self.log.log("error", kind="oom", step=step)
            print(f"CUDA OOM at step {step}. Try: micro_batch_size={max(1, tc.micro_batch_size // 2)}, "
                  f"grad_accum={tc.grad_accum * 2 if tc.micro_batch_size > 1 else tc.grad_accum}, shorter seq_len, "
                  "or gradient_checkpointing=true.")
            torch.cuda.empty_cache(); cleanup(self.dist); raise SystemExit(3)
        ev = self.evaluate()
        if ev:
            self.log.log("eval", step=step, final=True, **ev)
        self._save(step, last_loss, epoch, cur_batch, latest_val)
        summary = {"final_step": step, "steps_this_run": step - start_step, "last_train_loss": last_loss,
                   "global_step": step, "tokens_seen": step * self.tokens_per_step,
                   "total_tokens_target": tc.max_steps * self.tokens_per_step,
                   "trainable_parameters": self.applied["params"], "optimizer": "AdamW",
                   "precision": self.prec_name, "micro_batch_size": tc.micro_batch_size,
                   "grad_accum": tc.grad_accum, "sequence_length": tc.seq_len,
                   "provider_name": self.environment.provider,
                   "gpu_name": self.environment.gpu_name or "N/A",
                   "gpu_vram_total_bytes": self.environment.vram_total_bytes,
                   "checkpoint_path": self.last_checkpoint_path,
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "final_eval": ev, "best_val_loss": None if best_val == float("inf") else best_val,
                   "stop_reason": reason, "seconds": round(time.time() - t0, 1)}
        summary["cost_guard"] = self.cost_guard.report()
        self.log.log("end", **summary)
        if self.dist.is_main:
            (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        cleanup(self.dist)
        return summary


def build_from_dirs(model_cfg: MiniCoderConfig, tcfg: TrainConfig, cost_confirmed: bool = False) -> Trainer:
    d = Path(tcfg.data_dir)
    train_ds = PackedTokenDataset(str(d / "train.bin"), tcfg.seq_len)
    val_path = d / "val.bin"
    val_ds = PackedTokenDataset(str(val_path), tcfg.seq_len) if val_path.exists() else None
    vocab = train_ds.meta["vocab_size"]
    if model_cfg.vocab_size != vocab:
        print(f"NOTE: model vocab_size {model_cfg.vocab_size} -> tokenizer vocab {vocab}")
        model_cfg.vocab_size = vocab
    if tcfg.seq_len > model_cfg.max_position_embeddings:
        raise ValueError("seq_len exceeds model max_position_embeddings")
    meta = {"dataset_version": train_ds.meta.get("dataset_version"), "tokenizer_fingerprint": train_ds.meta.get("tokenizer_fingerprint"),
            "tokenizer_config": train_ds.meta.get("tokenizer_config"),
            "tokenizer_is_fallback": train_ds.meta.get("tokenizer_fallback"), "train_windows": len(train_ds),
            "val_windows": len(val_ds) if val_ds else 0}
    return Trainer(model_cfg, tcfg, train_ds, val_ds, meta, cost_confirmed=cost_confirmed)


def main() -> None:
    ap = argparse.ArgumentParser(description="MiniCoder trainer")
    ap.add_argument("--config", default="config/model_tiny.yaml")
    ap.add_argument("--train-config", default=None)
    ap.add_argument("--data-dir"); ap.add_argument("--run-dir"); ap.add_argument("--max-steps", type=int)
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--force-resume", action="store_true")
    ap.add_argument("--confirm-potential-cost", action="store_true",
                    help="explicitly approve the configured estimated GPU budget (not a hard provider billing cap)")
    ap.add_argument("--gpu-cost-rate-usd-per-hour", type=float)
    ap.add_argument("--max-budget-usd", type=float)
    ap.add_argument("--auto-batch-size", action="store_true",
                    help="run measured local forward/backward candidates before training")
    ap.add_argument("--max-auto-micro-batch-size", type=int)
    ap.add_argument("--smoke-test", action="store_true")
    a = ap.parse_args()
    if a.smoke_test:
        from training.smoke import run_smoke_test
        raise SystemExit(0 if run_smoke_test() else 1)
    tcfg = load_train_config(a.train_config, data_dir=a.data_dir, run_dir=a.run_dir, max_steps=a.max_steps,
                             resume=True if a.resume else None, force_resume=True if a.force_resume else None,
                             gpu_cost_rate_usd_per_hour=a.gpu_cost_rate_usd_per_hour,
                             max_budget_usd=a.max_budget_usd,
                             auto_batch_size=True if a.auto_batch_size else None,
                             max_auto_micro_batch_size=a.max_auto_micro_batch_size)
    build_from_dirs(load_config(a.config), tcfg, cost_confirmed=a.confirm_potential_cost).train()


if __name__ == "__main__":
    main()
