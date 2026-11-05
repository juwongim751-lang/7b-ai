"""Training run config (torch-free so it can be unit-tested anywhere)."""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class TrainConfig:
    data_dir: str = "data/processed"
    run_dir: str = "runs/default"
    checkpoint_dir: Optional[str] = None      # default: <run_dir>/checkpoints
    seq_len: int = 512
    micro_batch_size: int = 2
    grad_accum: int = 8
    max_steps: int = 1000                     # optimizer updates (NOT micro-batches)
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    precision: Optional[str] = None           # None -> model config value
    gradient_checkpointing: Optional[bool] = None
    eval_every: int = 100
    eval_batches: int = 20
    early_stopping_patience: int = 0          # 0 = off (counted in evaluations)
    early_stopping_min_delta: float = 0.0
    save_every: int = 200
    keep_checkpoints: int = 3
    log_every: int = 10
    seed: int = 1337
    num_workers: int = 0
    device: str = "auto"                      # auto | cpu | cuda
    resume: bool = False
    force_resume: bool = False
    gpu_cost_rate_usd_per_hour: Optional[float] = None
    max_budget_usd: Optional[float] = None
    cost_guard_reserve_seconds: float = 60.0
    auto_batch_size: bool = False
    max_auto_micro_batch_size: int = 8
    auto_tune_steps: int = 2

    def __post_init__(self) -> None:
        for n in ("seq_len", "micro_batch_size", "grad_accum", "max_steps", "log_every", "save_every"):
            if getattr(self, n) < 1:
                raise ValueError(f"{n} must be >= 1")
        if self.precision not in (None, "fp32", "fp16", "bf16"):
            raise ValueError("precision must be fp32/fp16/bf16 or null")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto/cpu/cuda")
        if (self.gpu_cost_rate_usd_per_hour is None) != (self.max_budget_usd is None):
            raise ValueError("gpu_cost_rate_usd_per_hour and max_budget_usd must be set together")
        if self.gpu_cost_rate_usd_per_hour is not None and self.gpu_cost_rate_usd_per_hour <= 0:
            raise ValueError("gpu_cost_rate_usd_per_hour must be positive")
        if self.max_budget_usd is not None and self.max_budget_usd <= 0:
            raise ValueError("max_budget_usd must be positive")
        if self.cost_guard_reserve_seconds < 0:
            raise ValueError("cost_guard_reserve_seconds cannot be negative")
        if self.max_auto_micro_batch_size < 1 or self.auto_tune_steps < 1:
            raise ValueError("auto batch probe limits must be positive")


def load_train_config(path: Optional[str] = None, **overrides: Any) -> TrainConfig:
    data: Dict[str, Any] = {}
    if path:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    known = {f.name for f in fields(TrainConfig)}
    data.update({k: v for k, v in overrides.items() if v is not None})
    bad = set(data) - known
    if bad:
        raise ValueError(f"unknown train config keys: {sorted(bad)}")
    return TrainConfig(**data)


def cosine_lr_lambda(step: int, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    """LR multiplier for the optimizer-update index `step` (0-based). Linear warmup then cosine to min_ratio."""
    warmup = max(1, min(warmup, max(1, total // 2)))   # warmup can never swallow a short run
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


class EarlyStopper:
    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience, self.min_delta, self.best, self.bad = patience, min_delta, float("inf"), 0

    def update(self, val_loss: float) -> bool:
        """Returns True when training should stop."""
        if val_loss < self.best - self.min_delta:
            self.best, self.bad = val_loss, 0
        else:
            self.bad += 1
        return self.patience > 0 and self.bad >= self.patience
