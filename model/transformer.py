"""MiniCoder decoder-only Transformer (Phase 2).

RMSNorm, RoPE, GQA, SwiGLU, SDPA attention, KV cache, generate().
Class/attribute names are kept compatible with the original file
(MiniCoder7B, embed_tokens, layers[i].attn.{q,k,v,o}, ffn.{gate,up,down}, norm, lm_head).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model.bitlinear import BitLinear
from model.config import MiniCoderConfig

KV = Tuple[torch.Tensor, torch.Tensor]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * y).to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    """Computes cos/sin once per forward (not per layer). positions: [B,S] -> [B,1,S,D]."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs = positions.float().unsqueeze(-1) * self.inv_freq.float()
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)


def _linear(in_features: int, out_features: int, cfg: MiniCoderConfig) -> nn.Module:
    if cfg.use_bitnet:
        return BitLinear(in_features, out_features, bias=False, activation_bits=cfg.activation_bits)
    return nn.Linear(in_features, out_features, bias=False)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, intermediate: int, cfg: MiniCoderConfig):
        super().__init__()
        self.bitnet = cfg.use_bitnet
        self.gate = None if self.bitnet else _linear(dim, intermediate, cfg)
        self.up = _linear(dim, intermediate, cfg)
        self.down = _linear(intermediate, dim, cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bitnet:
            return self.down(F.relu(self.up(x)).square())
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MiniCoderAttention(nn.Module):
    def __init__(self, cfg: MiniCoderConfig):
        super().__init__()
        self.nh, self.nkv, self.hd = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        self.rep = self.nh // self.nkv
        d = cfg.hidden_size
        self.q = _linear(d, self.nh * self.hd, cfg)
        self.k = _linear(d, self.nkv * self.hd, cfg)
        self.v = _linear(d, self.nkv * self.hd, cfg)
        self.o = _linear(self.nh * self.hd, d, cfg)

    def forward(self, x, cos, sin, attn_mask=None, is_causal=False,
                past_kv: Optional[KV] = None, use_cache: bool = False):
        b, s, _ = x.shape
        q = self.q(x).view(b, s, self.nh, self.hd).transpose(1, 2)    # [B,H,S,D]
        k = self.k(x).view(b, s, self.nkv, self.hd).transpose(1, 2)   # [B,Hkv,S,D]
        v = self.v(x).view(b, s, self.nkv, self.hd).transpose(1, 2)
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        present = (k, v) if use_cache else None  # cache stores un-expanded KV heads
        if self.rep > 1:  # query head i uses kv head i // rep (same order as repeat_interleave)
            t = k.shape[2]
            k = k[:, :, None].expand(b, self.nkv, self.rep, t, self.hd).reshape(b, self.nh, t, self.hd)
            v = v[:, :, None].expand(b, self.nkv, self.rep, t, self.hd).reshape(b, self.nh, t, self.hd)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(b, s, self.nh * self.hd)
        return self.o(out), present


class MiniCoderBlock(nn.Module):
    def __init__(self, cfg: MiniCoderConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = MiniCoderAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.ffn = SwiGLU(cfg.hidden_size, cfg.intermediate_size, cfg)

    def forward(self, x, cos, sin, attn_mask=None, is_causal=False, past_kv=None, use_cache=False):
        h, present = self.attn(self.attn_norm(x), cos, sin, attn_mask, is_causal, past_kv, use_cache)
        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        return x, present


def _sample(logits: torch.Tensor, temperature: float, top_k: Optional[int], top_p: Optional[float]):
    if temperature <= 0:
        return logits.argmax(-1)
    logits = logits / temperature
    if top_k:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sl, si = torch.sort(logits, descending=True)
        probs = sl.softmax(-1)
        remove = (probs.cumsum(-1) - probs) > top_p
        sl = sl.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, si, sl)
    return torch.multinomial(logits.softmax(-1), 1).squeeze(-1)


class MiniCoder7B(nn.Module):
    """Despite the legacy name, size is fully determined by MiniCoderConfig."""

    def __init__(self, cfg: MiniCoderConfig):
        super().__init__()
        self.config = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.rope_theta)
        self.layers = nn.ModuleList([MiniCoderBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = _linear(cfg.hidden_size, cfg.vocab_size, cfg)
        self.gradient_checkpointing = cfg.gradient_checkpointing
        self.apply(self._init_weights)
        # residual-scaled init for projections that write into the residual stream
        scale = (2 * cfg.num_layers) ** -0.5
        for name, p in self.named_parameters():
            if name.endswith("attn.o.weight") or name.endswith("ffn.down.weight"):
                nn.init.normal_(p, mean=0.0, std=cfg.initializer_range * scale)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding, BitLinear)):
            nn.init.normal_(m.weight, mean=0.0, std=self.config.initializer_range)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())  # parameters() de-duplicates tied weights

    @staticmethod
    def _build_mask(attention_mask, s: int, past_len: int, device):
        """Returns (bool mask [B or 1,1,S,T] or None, is_causal). True = may attend."""
        kv_len = past_len + s
        if attention_mask is None:
            if past_len == 0:
                return None, s > 1
            if s == 1:
                return None, False
        causal = torch.ones(s, kv_len, dtype=torch.bool, device=device).tril(diagonal=past_len)
        mask = causal[None, None]
        if attention_mask is not None:
            if attention_mask.shape[-1] != kv_len:
                raise ValueError(f"attention_mask length {attention_mask.shape[-1]} != kv length {kv_len}")
            mask = mask & attention_mask[:, None, None, :].to(torch.bool)
            mask = mask | ~mask.any(dim=-1, keepdim=True)  # fully-masked (pad) rows: avoid NaN
        return mask, False

    def forward(self, input_ids, attention_mask=None, positions=None,
                past_key_values: Optional[List[KV]] = None, use_cache: bool = False):
        b, s = input_ids.shape
        past_len = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        if past_len + s > self.config.max_position_embeddings:
            raise ValueError(f"sequence length {past_len + s} exceeds max_position_embeddings "
                             f"{self.config.max_position_embeddings}")
        if positions is None:
            positions = torch.arange(past_len, past_len + s, device=input_ids.device)
        if positions.dim() == 1:
            positions = positions[None, :].expand(b, -1)
        cos, sin = self.rope(positions)
        mask, is_causal = self._build_mask(attention_mask, s, past_len, input_ids.device)
        x = self.embed_tokens(input_ids)
        presents: List[KV] = []
        for i, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[i]
            if self.training and self.gradient_checkpointing and not use_cache:
                def run(a, c, sn, mk, layer=layer, ic=is_causal):
                    return layer(a, c, sn, mk, ic)[0]
                x = checkpoint(run, x, cos, sin, mask, use_reentrant=False)
            else:
                x, present = layer(x, cos, sin, mask, is_causal, past, use_cache)
                if use_cache:
                    presents.append(present)
        logits = self.lm_head(self.norm(x))
        return (logits, presents) if use_cache else logits

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 64, temperature: float = 1.0,
                 top_k: Optional[int] = None, top_p: Optional[float] = None,
                 eos_token_id: Optional[int] = None) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            out = input_ids
            logits, past = self(input_ids, use_cache=True)
            for _ in range(max_new_tokens):
                if out.shape[1] >= self.config.max_position_embeddings:
                    break
                nxt = _sample(logits[:, -1, :].float(), temperature, top_k, top_p)
                out = torch.cat([out, nxt[:, None]], dim=1)
                if eos_token_id is not None and bool((nxt == eos_token_id).all()):
                    break
                logits, past = self(nxt[:, None], past_key_values=past, use_cache=True)
            return out
        finally:
            self.train(was_training)


MiniCoder = MiniCoder7B
