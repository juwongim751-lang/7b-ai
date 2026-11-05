"""MiniCoder model/training-hyperparameter config (Phase 2: presets, tying, validation)."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict

import yaml

PRESETS: Dict[str, Dict[str, Any]] = {
    "7b": dict(vocab_size=32000, hidden_size=4096, num_layers=32, num_heads=32, num_kv_heads=8,
               intermediate_size=14336, max_position_embeddings=4096),
    "small": dict(vocab_size=32000, hidden_size=768, num_layers=12, num_heads=12, num_kv_heads=4,
                  intermediate_size=2048, max_position_embeddings=2048, tie_word_embeddings=True,
                  gradient_checkpointing=False),
    "tiny": dict(vocab_size=512, hidden_size=128, num_layers=2, num_heads=4, num_kv_heads=2,
                 intermediate_size=256, max_position_embeddings=256, tie_word_embeddings=True,
                 precision="fp32", gradient_checkpointing=False, warmup_steps=5, max_steps=50),
    "bitnet_tiny": dict(vocab_size=512, hidden_size=128, num_layers=2, num_heads=4, num_kv_heads=2,
                         intermediate_size=256, max_position_embeddings=256, tie_word_embeddings=False,
                         precision="fp32", gradient_checkpointing=False, use_bitnet=True),
}


@dataclass
class MiniCoderConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 8
    intermediate_size: int = 14336
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    precision: str = "bf16"
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    max_steps: int = 100000
    grad_clip: float = 1.0
    gradient_checkpointing: bool = True
    # --- added in Phase 2 ---
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02
    use_bitnet: bool = False
    activation_bits: int = 8

    def __post_init__(self) -> None:
        for name in ("vocab_size", "hidden_size", "num_layers", "num_heads", "num_kv_heads",
                     "intermediate_size", "max_position_embeddings"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if (self.hidden_size // self.num_heads) % 2:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")
        if self.activation_bits < 2 or self.activation_bits > 16:
            raise ValueError("activation_bits must be between 2 and 16")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def calculate_parameters(self) -> int:
        d, hd = self.hidden_size, self.head_dim
        emb = self.vocab_size * d
        attn = d * d + 2 * d * self.num_kv_heads * hd + d * d
        mlp = (2 if self.use_bitnet else 3) * d * self.intermediate_size
        norms = 2 * d
        head = 0 if self.tie_word_embeddings else self.vocab_size * d
        return emb + self.num_layers * (attn + mlp + norms) + d + head


def config_from_dict(data: Dict[str, Any]) -> MiniCoderConfig:
    if data.get("preset") and data["preset"] not in PRESETS:
        raise ValueError(f"unknown preset {data['preset']!r}; choose from {sorted(PRESETS)}")
    base = dict(PRESETS[data["preset"]]) if data.get("preset") else {}
    params = dict(base)
    params.update(data.get("parameters", {}) or {})
    params.update(data.get("training", {}) or {})
    known = {f.name for f in fields(MiniCoderConfig)}
    unknown = set(params) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return MiniCoderConfig(**params)


def load_config(path: str = "config/model.yaml") -> MiniCoderConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return config_from_dict(data)
