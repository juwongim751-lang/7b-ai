"""Load MiniCoder model-only exports, including packed ternary reference weights."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from model.config import MiniCoderConfig
from model.transformer import MiniCoder7B


def load_exported_model(export_dir, device="cpu"):
    root = Path(export_dir).resolve(strict=True)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    cfg = MiniCoderConfig(
        vocab_size=config["vocab_size"], hidden_size=config["hidden_size"],
        num_layers=config["num_hidden_layers"], num_heads=config["num_attention_heads"],
        num_kv_heads=config["num_key_value_heads"], intermediate_size=config["intermediate_size"],
        max_position_embeddings=config["max_position_embeddings"], rope_theta=config.get("rope_theta", 10000.0),
        tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
        use_bitnet=bool(config.get("use_bitnet", False)),
        activation_bits=int(config.get("activation_bits", 8)), precision="fp32",
    )
    model = MiniCoder7B(cfg)
    packed = bool(config.get("packed_ternary", False))
    if packed:
        from model.bitpack import convert_bitlinear_modules
        if not cfg.use_bitnet:
            raise ValueError("packed export must set use_bitnet=true")
        if convert_bitlinear_modules(model) == 0:
            raise ValueError("packed export has no BitLinear modules")
    manifest = json.loads((root / "weights_manifest.json").read_text(encoding="utf-8"))
    weights = root / manifest["weights"]
    if not weights.is_file() or weights.parent != root:
        raise ValueError("weights file must be a regular file inside the export directory")
    if weights.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError("install safetensors to load this export") from exc
        state = load_file(str(weights), device=str(device))
    elif weights.suffix == ".bin":
        state = torch.load(weights, map_location=device, weights_only=True)
    else:
        raise ValueError("unsupported model weight format")
    result = model.load_state_dict(state, strict=False)
    tolerated = {"lm_head.weight"} if cfg.tie_word_embeddings else set()
    missing = set(result.missing_keys) - tolerated
    if missing or result.unexpected_keys:
        raise ValueError(f"export state mismatch: missing={sorted(missing)}, unexpected={result.unexpected_keys}")
    model.to(device).eval()
    return model, cfg, manifest
