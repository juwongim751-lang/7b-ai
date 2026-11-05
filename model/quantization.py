"""Reference quantization primitives for BitNet-style quantization-aware training.

These are correctness-oriented PyTorch implementations, not fused low-bit kernels.
"""
from __future__ import annotations

import torch


def quantize_ternary_absmean(weight: torch.Tensor, eps: float = 1e-5,
                             straight_through: bool = True) -> torch.Tensor:
    """Absmean ternary weight quantization to {-1, 0, +1}.

    The absolute mean scale is used to normalize weights before round-and-clip.
    The returned values are ternary-valued in the forward pass. With STE enabled,
    gradients use the identity surrogate through the quantizer.
    """
    if weight.numel() == 0:
        raise ValueError("weight must be non-empty")
    scale = weight.detach().abs().mean().clamp_min(eps)
    q = torch.round(weight / scale).clamp(-1, 1)
    if straight_through:
        return weight + (q - weight).detach()
    return q


def quantize_activation_symmetric(x: torch.Tensor, bits: int = 8, eps: float = 1e-8,
                                  straight_through: bool = True) -> torch.Tensor:
    """Symmetric per-token activation quantization followed by dequantization.

    The last dimension is quantized independently for every token. This mirrors
    the W1.58A8 reference path; it does not imply that activations are stored in
    packed int8 form or that this function uses an integer GEMM kernel.
    """
    if bits < 2 or bits > 16:
        raise ValueError("activation bits must be between 2 and 16")
    qmax = (1 << (bits - 1)) - 1
    xf = x.float()
    scale = xf.abs().amax(dim=-1, keepdim=True).clamp_min(eps) / qmax
    q = torch.round(xf / scale).clamp(-qmax, qmax) * scale
    q = q.to(dtype=x.dtype)
    if straight_through:
        return x + (q - x).detach()
    return q
