"""Resource measurements used by training; unavailable values remain None."""
from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None
try:
    import pynvml
except ImportError:  # pragma: no cover
    pynvml = None


@dataclass
class ResourceSnapshot:
    gpu_memory_used_bytes: Optional[int] = None
    gpu_memory_total_bytes: Optional[int] = None
    gpu_utilization_percent: Optional[float] = None
    ram_used_bytes: Optional[int] = None
    ram_total_bytes: Optional[int] = None
    data_wait_seconds: Optional[float] = None
    step_seconds: Optional[float] = None
    oom: bool = False


class ResourceManager:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._last_batch_end = None
        self._step_start = None
        self._nvml_handle = None
        if pynvml is not None and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index or 0)
            except Exception:
                self._nvml_handle = None

    def snapshot(self) -> Dict[str, Any]:
        result = ResourceSnapshot()
        if psutil is not None:
            try:
                vm = psutil.virtual_memory()
                result.ram_used_bytes, result.ram_total_bytes = int(vm.used), int(vm.total)
            except Exception:
                pass
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info(self.device)
                result.gpu_memory_total_bytes = int(total)
                result.gpu_memory_used_bytes = int(total - free)
            except Exception:
                pass
            if self._nvml_handle is not None:
                try:
                    result.gpu_utilization_percent = float(pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle).gpu)
                except Exception:
                    pass
        return asdict(result)

    def before_batch(self):
        self._step_start = time.perf_counter()
        return time.perf_counter()

    def after_batch(self, batch_ready_time: Optional[float] = None) -> Dict[str, Any]:
        now = time.perf_counter()
        snap = self.snapshot()
        if self._step_start is not None:
            snap["step_seconds"] = now - self._step_start
        if self._last_batch_end is not None and batch_ready_time is not None:
            snap["data_wait_seconds"] = max(0.0, batch_ready_time - self._last_batch_end)
        self._last_batch_end = now
        return snap

    @staticmethod
    def oom_recommendation(config: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(config)
        if "micro_batch_size" in out:
            out["micro_batch_size"] = max(1, int(out["micro_batch_size"]) // 2)
        if "grad_accum" in out:
            out["grad_accum"] = max(1, int(out["grad_accum"]) * 2)
        if "seq_len" in out:
            out["seq_len"] = max(128, int(out["seq_len"]) // 2)
        out["gradient_checkpointing"] = True
        return out

    def close(self):
        if self._nvml_handle is not None and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def probe_training_config(model, batch_factory, candidates=None, device=None, steps=2) -> Dict[str, Any]:
    """Run real forward/backward probes and return measured recommendations.

    ``batch_factory(micro_batch_size, seq_len)`` must return ``(x, y)`` tensors.
    Failed candidates are recorded, never replaced by made-up measurements.
    """
    device = device or next(model.parameters()).device
    candidates = candidates or [{"micro_batch_size": 1, "seq_len": 128}, {"micro_batch_size": 2, "seq_len": 128}]
    results = []
    was_training = model.training
    model.train()
    for candidate in candidates:
        record = dict(candidate)
        t0 = time.perf_counter()
        try:
            model.zero_grad(set_to_none=True)
            x, y = batch_factory(candidate["micro_batch_size"], candidate["seq_len"])
            x, y = x.to(device), y.to(device)
            for _ in range(steps):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
                loss.backward()
                model.zero_grad(set_to_none=True)
            elapsed = max(time.perf_counter() - t0, 1e-9)
            record.update({"status": "ok", "loss": float(loss.detach()),
                           "tokens_per_second": float(x.numel() * steps / elapsed),
                           "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None})
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            record.update({"status": "oom" if "out of memory" in str(exc).lower() else "error", "error": str(exc)})
            if device.type == "cuda": torch.cuda.empty_cache()
        results.append(record)
        gc.collect()
    valid = [r for r in results if r.get("status") == "ok"]
    recommendation = max(valid, key=lambda r: r.get("tokens_per_second", 0)) if valid else None
    if not was_training: model.eval()
    return {"candidates": results, "recommendation": recommendation}
