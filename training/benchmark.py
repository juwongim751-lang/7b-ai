"""Small local GPU/CPU benchmark; remote provider claims are never inferred."""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model.config import load_config
from model.transformer import MiniCoder7B
from providers.detector import detect_environment


def run_benchmark(output_dir="reports", steps=5, batch_size=2, seq_len=32,
                  config_path="config/model_tiny.yaml"):
    if min(steps, batch_size, seq_len) < 1:
        raise ValueError("steps, batch_size and seq_len must be positive")
    env = detect_environment()
    device = torch.device("cuda" if env.gpu_available else "cpu")
    cfg = load_config(config_path)
    if seq_len > cfg.max_position_embeddings:
        raise ValueError("seq_len exceeds Tiny model context")
    model = MiniCoder7B(cfg).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    tokens = steps * batch_size * seq_len
    result = {
        "provider_detected": env.provider,
        "environment_signals": list(env.signals),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "gpu_available": env.gpu_available, "gpu_name": env.gpu_name or "N/A",
        "vram_total_bytes": env.vram_total_bytes, "vram_free_bytes": env.vram_free_bytes,
        "memory_bandwidth": "N/A (not measured)", "device_used": str(device),
        "model": config_path, "parameters": model.num_parameters(),
        "batch_size": batch_size, "sequence_length": seq_len, "steps": steps,
        "elapsed_seconds": round(elapsed, 6), "tokens_measured": tokens,
        "tokens_per_second": round(tokens / max(elapsed, 1e-12), 2),
        "final_loss": losses[-1], "gpu_peak_memory_bytes":
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "execution_time_limit": "N/A (local process does not expose provider session quota)",
        "remote_job_submission": False,
        "note": "A small correctness/performance probe, not a provider-wide or 7B throughput estimate.",
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gpu_benchmark.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Local GPU/CPU benchmark", "",
             "This report measures the environment running this command only. It does not test cloud account access, quotas, or other providers.", "",
             f"- Detected provider: `{env.provider}`",
             f"- Actual device: `{device}` ({env.gpu_name or 'N/A'})",
             f"- Torch / CUDA: `{torch.__version__}` / `{torch.version.cuda or 'N/A'}`",
             f"- Model config / parameters: `{config_path}` / `{result['parameters']:,}`",
             f"- Batch × sequence: `{batch_size} × {seq_len}`; measured steps: `{steps}`",
             f"- Measured throughput: `{result['tokens_per_second']}` tokens/s",
             f"- Peak GPU allocation: `{result['gpu_peak_memory_bytes'] if result['gpu_peak_memory_bytes'] is not None else 'N/A'}` bytes",
             f"- Matrix memory bandwidth: `{result['memory_bandwidth']}`",
             f"- Final synthetic loss: `{result['final_loss']:.5f}`", "",
             result["note"], ""]
    (out / "gpu_benchmark.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--config", default="config/model_tiny.yaml")
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.output_dir, args.steps, args.batch_size, args.seq_len,
                                   args.config), indent=2))


if __name__ == "__main__":
    main()
