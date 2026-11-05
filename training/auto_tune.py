"""Preflight tuning based on measured forward/backward probes."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import torch

from training.resource_manager import probe_training_config


def run_auto_tune(model, batch_factory: Callable, output_path: str = "applied_config.json",
                  candidates=None, device=None, steps: int = 2, base_config: Dict[str, Any] | None = None):
    measured = probe_training_config(model, batch_factory, candidates, device, steps)
    applied = dict(base_config or {})
    if measured.get("recommendation"):
        applied.update({k: v for k, v in measured["recommendation"].items()
                        if k in {"micro_batch_size", "seq_len"}})
    payload = {"applied_config": applied, "probe": measured,
               "performance_claim": "none; before/after measurements are required"}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return payload


def probe_auto_batch_size(model, dataset, micro_batch_size, grad_accum, seq_len,
                          max_micro_batch_size=8, steps=2, device=None):
    """Measure candidate batch sizes and choose the highest observed throughput.

    Effective batch remains approximately constant. No selection is made when all
    candidates fail; the caller keeps user settings and the report explains why.
    """
    device = device or next(model.parameters()).device
    effective = int(micro_batch_size) * int(grad_accum)
    upper = min(int(max_micro_batch_size), effective, len(dataset))
    candidates = []
    candidate = 1
    while candidate <= upper:
        candidates.append({"micro_batch_size": candidate, "seq_len": min(int(seq_len), dataset[0][0].shape[-1])})
        candidate *= 2
    if int(micro_batch_size) not in [item["micro_batch_size"] for item in candidates] and int(micro_batch_size) <= upper:
        candidates.append({"micro_batch_size": int(micro_batch_size),
                           "seq_len": min(int(seq_len), dataset[0][0].shape[-1])})
    candidates = sorted({(item["micro_batch_size"], item["seq_len"]) for item in candidates})
    candidates = [{"micro_batch_size": b, "seq_len": s} for b, s in candidates]

    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    was_training = model.training
    def batch_factory(batch_size, length):
        xs, ys = [], []
        for index in range(batch_size):
            x, y = dataset[index % len(dataset)]
            xs.append(x[:length]); ys.append(y[:length])
        return torch.stack(xs), torch.stack(ys)
    try:
        report = probe_training_config(model, batch_factory, candidates, device, steps)
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        np.random.set_state(numpy_rng); random.setstate(python_rng)
        model.train(was_training)
    chosen = report.get("recommendation")
    recommendation = None
    if chosen:
        micro = int(chosen["micro_batch_size"])
        accum = max(1, round(effective / micro))
        recommendation = {"micro_batch_size": micro, "grad_accum": accum,
                          "seq_len": int(chosen["seq_len"]),
                          "measured_tokens_per_second": chosen["tokens_per_second"],
                          "effective_batch_before": effective,
                          "effective_batch_after": micro * accum}
    return {"candidates": report["candidates"], "recommendation": recommendation,
            "selection_basis": "highest measured tokens/sec among successful probes",
            "model_weights_updated_during_probe": False,
            "performance_claim": "probe measurement only; not a production throughput guarantee"}
