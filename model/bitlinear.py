"""A transparent BitLinear reference layer for ternary QAT experiments."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from model.quantization import quantize_activation_symmetric, quantize_ternary_absmean


class BitLinear(nn.Module):
    """Linear layer with absmean ternary weights and symmetric quantized activations.

    Master weights and optimizer state remain floating point during training.
    This reference implementation dequantizes ternary weights for ``F.linear``;
    it is for correctness/testing, not a compressed checkpoint or speedup claim.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 activation_bits: int = 8, eps: float = 1e-5):
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        self.in_features, self.out_features = int(in_features), int(out_features)
        self.activation_bits, self.eps = int(activation_bits), float(eps)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            bound = self.in_features ** -0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def quantized_weight(self) -> torch.Tensor:
        # Preserve the layer's absolute-mean magnitude after ternarization.
        scale = self.weight.detach().abs().mean().clamp_min(self.eps)
        ternary = quantize_ternary_absmean(self.weight, self.eps, straight_through=True)
        return ternary * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xq = quantize_activation_symmetric(x, bits=self.activation_bits, eps=self.eps)
        return F.linear(xq, self.quantized_weight(), self.bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, activation_bits={self.activation_bits}")
