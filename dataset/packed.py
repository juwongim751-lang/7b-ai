"""Packed token dataset + deterministic, rank-disjoint epoch ordering (Phase 3/4).
`epoch_indices` is numpy-only; PackedTokenDataset imports torch lazily."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np


def epoch_indices(n: int, seed: int, epoch: int, rank: int = 0, world: int = 1) -> np.ndarray:
    """Same permutation on every rank (seed+epoch); rank r takes every world-th item -> disjoint shards.
    The permutation is padded (wrap-around) to a multiple of `world` so all ranks get equal length."""
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    perm = np.random.default_rng(seed + 1_000_003 * epoch).permutation(n)
    pad = (-n) % world
    if pad:
        perm = np.concatenate([perm, perm[:pad]])
    return perm[rank::world]


class PackedTokenDataset:
    """Non-overlapping windows of seq_len+1 tokens from a flat .bin (documents already BOS...EOS delimited).
    No padding is needed. NOTE: attention is not reset at document boundaries."""

    def __init__(self, bin_path: str, seq_len: int):
        p = Path(bin_path)
        meta = json.loads(p.with_suffix(".meta.json").read_text(encoding="utf-8"))
        self.meta, self.seq_len = meta, seq_len
        self.data = np.memmap(p, dtype=np.dtype(meta["dtype"]), mode="r")
        self.n = (len(self.data) - 1) // seq_len if len(self.data) > seq_len else 0

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        import torch
        if not 0 <= i < self.n:
            raise IndexError(i)
        s = i * self.seq_len
        chunk = np.asarray(self.data[s:s + self.seq_len + 1], dtype=np.int64)
        t = torch.from_numpy(chunk)
        return t[:-1], t[1:]
