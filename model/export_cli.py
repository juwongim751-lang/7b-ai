"""Export a MiniCoder model from a locally generated, integrity-checked checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model.config import MiniCoderConfig
from model.export import ModelExporter
from model.transformer import MiniCoder7B
from training.checkpoint import CheckpointManager


def export_checkpoint(checkpoint_path, output_dir, pack_bitnet=False):
    checkpoint = Path(checkpoint_path).resolve(strict=True)
    if not checkpoint.is_file() or not checkpoint.name.startswith("checkpoint_step_") or checkpoint.suffix != ".pt":
        raise ValueError("expected a checkpoint_step_*.pt file")
    manager = CheckpointManager(checkpoint.parent)
    manager._verify(checkpoint)
    # Checkpoints are local application artifacts with optimizer/RNG state; do not
    # load arbitrary user-downloaded .pt files from an untrusted source.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw = state.get("model_config") or state.get("metadata", {}).get("model_config")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("checkpoint does not include a model configuration")
    valid = {field.name for field in __import__("dataclasses").fields(MiniCoderConfig)}
    cfg = MiniCoderConfig(**{key: value for key, value in raw.items() if key in valid})
    model = MiniCoder7B(cfg)
    model.load_state_dict(state["model"])
    output = ModelExporter(model, cfg).export(output_dir, pack_bitnet=pack_bitnet)
    return {"output_dir": output, "checkpoint_step": int(state.get("step", -1)),
            "parameters": model.num_parameters(), "packed_ternary": bool(pack_bitnet)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pack-bitnet", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(args.checkpoint, args.output_dir, args.pack_bitnet), indent=2))


if __name__ == "__main__":
    main()
