"""Recommend a model scale from observed local resources; never starts training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from model.config import load_config
from providers.detector import detect_environment
from training.memory_estimator import estimate_training_memory_bytes

PRESET_FILES = [
    ("tiny_10m", "config/model_tiny_10m.yaml"),
    ("100m", "config/model_100m.yaml"),
    ("300m", "config/model_300m.yaml"),
    ("1b", "config/model_1b.yaml"),
    ("3b", "config/model_3b.yaml"),
    ("7b", "config/model_7b.yaml"),
]


def dataset_token_count(data_dir):
    path = Path(data_dir) / "manifest.json"
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        value = doc.get("tokenization", {}).get("total_tokens")
        return int(value) if value is not None else None
    except (OSError, ValueError, TypeError):
        return None


def recommend_plan(target="auto", data_dir="data/processed", seq_len=512,
                   micro_batch_size=1, precision="bf16", safety_fraction=0.80,
                   environment=None):
    if not 0.1 <= safety_fraction <= 0.95:
        raise ValueError("safety_fraction must be within [0.1, 0.95]")
    names = dict(PRESET_FILES)
    if target != "auto" and target not in names:
        raise ValueError(f"unknown target {target!r}; choose auto or {', '.join(names)}")
    env = environment or detect_environment()
    dataset_tokens = dataset_token_count(data_dir)
    choices = PRESET_FILES if target == "auto" else PRESET_FILES[:[n for n, _ in PRESET_FILES].index(target)+1]
    reports = []
    for name, config_path in choices:
        cfg = load_config(config_path)
        length = min(int(seq_len), cfg.max_position_embeddings)
        estimate = estimate_training_memory_bytes(cfg, micro_batch_size, length, precision,
                                                   gradient_checkpointing=True)
        memory_capacity = env.vram_free_bytes if env.gpu_available else env.system_ram_available_bytes
        safe_capacity = int(memory_capacity * safety_fraction) if memory_capacity else None
        fits = safe_capacity is not None and estimate["estimated_training_bytes"] <= safe_capacity
        reports.append({"preset": name, "config": config_path, "seq_len": length,
                        "memory": estimate, "observed_capacity_bytes": memory_capacity,
                        "capacity_kind": "free_vram" if env.gpu_available else "available_system_ram",
                        "safe_capacity_bytes": safe_capacity, "fits_heuristic": bool(fits)})
    fitting = [r for r in reports if r["fits_heuristic"]]
    selected = fitting[-1] if fitting else None
    requested_index = [n for n, _ in PRESET_FILES].index(target) if target != "auto" else None
    selected_index = [n for n, _ in PRESET_FILES].index(selected["preset"]) if selected else None
    fallback = bool(target != "auto" and selected and selected_index < requested_index)
    if target != "auto" and selected and selected["preset"] != target:
        fallback = True
    if target != "auto" and selected is None:
        recommendation = "no_fit"
    elif target != "auto" and fallback:
        recommendation = "smaller_preset_requires_user_confirmation"
    elif selected:
        recommendation = "selected_by_observed_resources"
    else:
        recommendation = "insufficient_observed_memory"
    plan_cfg = None
    if selected:
        tokens_per_update = int(micro_batch_size) * 8 * int(selected["seq_len"])
        planned_steps = (max(1, dataset_tokens // tokens_per_update)
                         if dataset_tokens is not None else 100)
        plan_cfg = {
            "config": selected["config"],
            "seq_len": selected["seq_len"],
            "micro_batch_size": int(micro_batch_size),
            "grad_accum": 8,
            "precision": precision,
            "gradient_checkpointing": True,
            "max_steps_one_dataset_pass_estimate": min(planned_steps, 10000),
            "steps_basis": "manifest token count and effective batch" if dataset_tokens is not None else "conservative 100-step probe because dataset token count is unknown",
        }
    return {
        "provider": env.provider,
        "gpu_available": env.gpu_available,
        "gpu_name": env.gpu_name,
        "free_vram_bytes": env.vram_free_bytes,
        "available_system_ram_bytes": env.system_ram_available_bytes,
        "dataset_tokens": dataset_tokens,
        "target": target,
        "selected": selected,
        "recommended_training_config": plan_cfg,
        "recommendation": recommendation,
        "requires_user_confirmation": fallback,
        "automatic_downsize_performed": False,
        "candidates": reports,
        "note": "Memory estimates are heuristic; this command only recommends a preset. It does not start training, submit cloud jobs, or infer quota. Training 7B requires a successful model/device-specific memory probe.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="auto", help="auto, tiny_10m, 100m, 300m, 1b, 3b, 7b")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    args = parser.parse_args()
    print(json.dumps(recommend_plan(args.target, args.data_dir, args.seq_len,
                                    args.micro_batch_size, args.precision), indent=2))


if __name__ == "__main__":
    main()
