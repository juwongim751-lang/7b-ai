"""Portable fixed-rate ternary packing for inference/export.

Five trits are encoded as one base-3 integer in a uint8 (0..242), yielding
1.6 bits/weight excluding per-layer scales and metadata. This is a practical
fixed-rate approximation to log2(3)=1.585 bits, not a bit-exact 1.58-bit codec
and not an accelerated integer GEMM kernel.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from model.quantization import quantize_activation_symmetric, quantize_ternary_absmean

_RADIX_POWERS = (1, 3, 9, 27, 81)


def pack_ternary_5(values: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Pack {-1,0,+1} tensor in row-major order to uint8; pad with zero trits."""
    if values.numel() == 0:
        raise ValueError("values must not be empty")
    flat = values.detach().to(torch.int8).reshape(-1)
    if not bool(torch.all((flat >= -1) & (flat <= 1))):
        raise ValueError("all values must be in {-1,0,+1}")
    n = flat.numel()
    pad = (-n) % 5
    # digit 0,1,2 represents ternary -1,0,+1 respectively.
    digits = (flat + 1).to(torch.int64)
    if pad:
        digits = torch.cat([digits, torch.ones(pad, dtype=torch.int64, device=digits.device)])
    groups = digits.reshape(-1, 5)
    powers = torch.tensor(_RADIX_POWERS, dtype=torch.int64, device=digits.device)
    packed = (groups * powers).sum(dim=1).to(torch.uint8)
    return packed, tuple(values.shape)


def unpack_ternary_5(packed: torch.Tensor, shape: tuple[int, ...], device=None,
                     dtype=torch.float32) -> torch.Tensor:
    expected = math.ceil(math.prod(shape) / 5)
    if packed.dtype != torch.uint8 or packed.numel() != expected:
        raise ValueError("packed tensor has invalid dtype or size")
    codes = packed.to(device=device or packed.device, dtype=torch.int64).reshape(-1)
    powers = torch.tensor(_RADIX_POWERS, dtype=torch.int64, device=codes.device)
    digits = torch.stack([(codes // p) % 3 for p in powers.tolist()], dim=-1)
    values = (digits.reshape(-1)[:math.prod(shape)] - 1).reshape(shape)
    return values.to(dtype=dtype)


class PackedTernaryLinear(nn.Module):
    """Inference-only F.linear module storing ternary weights in 1.6-bit blocks."""

    def __init__(self, in_features: int, out_features: int, activation_bits: int = 8,
                 eps: float = 1e-5, bias: bool = False):
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError("linear dimensions must be positive")
        self.in_features, self.out_features = int(in_features), int(out_features)
        self.activation_bits, self.eps = int(activation_bits), float(eps)
        self.register_buffer("packed_weight", torch.empty(0, dtype=torch.uint8))
        self.register_buffer("weight_scale", torch.ones((), dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias = None

    @classmethod
    def from_bitlinear(cls, layer):
        packed = cls(layer.in_features, layer.out_features, layer.activation_bits,
                     layer.eps, bias=layer.bias is not None)
        with torch.no_grad():
            scale = layer.weight.detach().abs().mean().clamp_min(layer.eps)
            ternary = torch.round(layer.weight.detach() / scale).clamp(-1, 1).to(torch.int8)
            codes, _ = pack_ternary_5(ternary)
            packed.packed_weight = codes
            packed.weight_scale = scale.float().reshape(()).cpu()
            if layer.bias is not None:
                packed.bias = layer.bias.detach().float().cpu().clone()
        return packed

    def quantized_weight(self, device, dtype):
        ternary = unpack_ternary_5(self.packed_weight, (self.out_features, self.in_features),
                                   device=device, dtype=dtype)
        return ternary * self.weight_scale.to(device=device, dtype=dtype)

    def forward(self, x):
        xq = quantize_activation_symmetric(x, bits=self.activation_bits, eps=self.eps)
        weight = self.quantized_weight(x.device, x.dtype)
        bias = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
        return F.linear(xq, weight, bias)


def convert_bitlinear_modules(module: nn.Module) -> int:
    """Replace BitLinear children in-place for packed inference export; return count."""
    from model.bitlinear import BitLinear
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, BitLinear):
            setattr(module, name, PackedTernaryLinear.from_bitlinear(child))
            replaced += 1
        else:
            replaced += convert_bitlinear_modules(child)
    return replaced
