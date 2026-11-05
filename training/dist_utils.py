"""Minimal DDP helpers with automatic single-process fallback. UNVERIFIED on multi-GPU (no GPU available)."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    note: str = ""

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistInfo:
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws <= 1:
        hint = ""
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            hint = (f"{torch.cuda.device_count()} GPUs visible but running single-process; for DDP launch: "
                    f"torchrun --nproc_per_node={torch.cuda.device_count()} -m training.trainer ...")
        return DistInfo(note=hint)
    if torch.cuda.is_available() and torch.cuda.device_count() < 2:
        return DistInfo(note="WORLD_SIZE>1 but only one GPU: falling back to single-process training")
    rank, local = int(os.environ["RANK"]), int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if (torch.cuda.is_available() and sys.platform != "win32") else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    dist.init_process_group(backend=backend)
    return DistInfo(True, rank, local, ws, f"DDP backend={backend}")


def all_reduce(t: torch.Tensor, info: DistInfo, op=None) -> torch.Tensor:
    if info.enabled:
        dist.all_reduce(t, op=op or dist.ReduceOp.SUM)
    return t


def barrier(info: DistInfo) -> None:
    if info.enabled:
        dist.barrier()


def cleanup(info: DistInfo) -> None:
    if info.enabled and dist.is_initialized():
        dist.destroy_process_group()
