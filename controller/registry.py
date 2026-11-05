"""In-memory registry with bounded, allowlisted worker metadata."""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


SAFE_CAPABILITY_FIELDS = {"cuda", "cuda_version", "python_version", "torch_version", "hostname", "platform"}
ALLOWED_BACKENDS = {"gloo", "nccl", "mpi", "unknown"}


@dataclass
class WorkerInfo:
    worker_id: str
    provider: str
    gpu_name: str
    vram_total: int
    vram_available: int
    status: str = "available"
    last_heartbeat: float = 0.0
    network_id: Optional[str] = None
    runtime: str = "unknown"
    distributed_backend: str = "unknown"
    reachable: bool = False
    address: Optional[str] = None
    port: int = 0
    capabilities: dict = field(default_factory=dict)


def _short_text(value, label: str, limit: int = 128) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{label} must be a non-empty string of at most {limit} characters")
    return value.strip()


def _nonnegative_int(value, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a non-negative integer") from exc
    if result < 0 or result > 2**50:
        raise ValueError(f"{label} is outside the supported range")
    return result


class WorkerRegistry:
    def __init__(self, timeout_threshold=30):
        self.workers = {}
        self.timeout_threshold = max(1, int(timeout_threshold))

    def register(self, info: Dict) -> str:
        if not isinstance(info, dict):
            raise ValueError("worker payload must be a JSON object")
        provider = _short_text(info.get("provider"), "provider")
        gpu_name = _short_text(info.get("gpu_name"), "gpu_name")
        vram_total = _nonnegative_int(info.get("vram_total", 0), "vram_total")
        vram_available = min(_nonnegative_int(info.get("vram_available", 0), "vram_available"), vram_total)
        backend = info.get("distributed_backend", "unknown")
        if backend not in ALLOWED_BACKENDS:
            backend = "unknown"
        port = _nonnegative_int(info.get("port", 0), "port")
        if port > 65535:
            raise ValueError("port must be between 0 and 65535")
        network_id = info.get("network_id")
        if not isinstance(network_id, str) or not network_id.strip() or len(network_id) > 128:
            network_id = None
        address = info.get("address")
        if not isinstance(address, str) or len(address) > 255:
            address = None
        runtime = info.get("runtime", "unknown")
        if not isinstance(runtime, str) or len(runtime) > 64:
            runtime = "unknown"
        worker_id = str(uuid.uuid4())  # never accept an externally claimed identity
        capabilities = {key: info[key] for key in SAFE_CAPABILITY_FIELDS if key in info and
                        isinstance(info[key], (str, int, float, bool)) and
                        (not isinstance(info[key], str) or len(info[key]) <= 256)}
        worker = WorkerInfo(worker_id, provider, gpu_name, vram_total, vram_available,
                            last_heartbeat=time.time(), network_id=network_id,
                            runtime=runtime, distributed_backend=backend,
                            reachable=info.get("reachable") is True, address=address, port=port,
                            capabilities=capabilities)
        self.workers[worker_id] = worker
        return worker_id

    def update_heartbeat(self, worker_id, info: Optional[Dict] = None):
        if worker_id not in self.workers:
            return False
        worker = self.workers[worker_id]
        worker.last_heartbeat = time.time()
        worker.status = "available"
        if isinstance(info, dict):
            if "vram_available" in info:
                worker.vram_available = min(_nonnegative_int(info["vram_available"], "vram_available"), worker.vram_total)
            if "reachable" in info:
                worker.reachable = info["reachable"] is True
            if "address" in info and isinstance(info["address"], str) and len(info["address"]) <= 255:
                worker.address = info["address"]
            if "port" in info:
                port = _nonnegative_int(info["port"], "port")
                if port > 65535:
                    raise ValueError("port must be between 0 and 65535")
                worker.port = port
        return True

    def get_active_workers(self) -> List[WorkerInfo]:
        now = time.time()
        active = []
        for worker in self.workers.values():
            if now - worker.last_heartbeat < self.timeout_threshold:
                active.append(worker)
            else:
                worker.status = "offline"
        return active

    def get_worker_status(self):
        return {worker_id: asdict(info) for worker_id, info in self.workers.items()}
