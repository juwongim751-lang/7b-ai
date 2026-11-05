"""Truthful inspection adapter for a Google Colab runtime.

This adapter only inspects the Python process and its local filesystem. It does
not mount Drive, install packages, submit cloud jobs, or connect to another
Colab session. ``run_worker_command`` preserves the legacy BaseProvider shape
by returning a *local command string*; it never executes that command.
"""
from collections.abc import Mapping
import importlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Any, Callable

from providers.base import BaseProvider


_UNSET = object()


class ColabProvider(BaseProvider):
    """Inspect the current process for Colab indicators and local resources."""

    def __init__(
        self,
        env: Mapping[str, str] | None = None,
        content_path: str | os.PathLike[str] = "/content",
        drive_mount_path: str | os.PathLike[str] | None = None,
        torch_module: Any = _UNSET,
        nvidia_smi_path: str | None | object = _UNSET,
        drive_mount_checker: Callable[[str], bool] | None = None,
    ):
        super().__init__("colab")
        self._env = os.environ if env is None else env
        self.content_path = Path(content_path)
        self.drive_mount_path = Path(drive_mount_path) if drive_mount_path is not None else self.content_path / "drive"
        self._torch_module = torch_module
        self._nvidia_smi_path = nvidia_smi_path
        self._drive_mount_checker = drive_mount_checker or os.path.ismount

    @staticmethod
    def _is_set(env: Mapping[str, str], key: str) -> bool:
        value = env.get(key)
        return value is not None and bool(str(value).strip())

    def detect_runtime(self) -> dict[str, Any]:
        """Return deterministic indicators for the current local process."""
        indicators = {
            "colab_release_tag": self._is_set(self._env, "COLAB_RELEASE_TAG"),
            "colab_gpu": self._is_set(self._env, "COLAB_GPU"),
            "content_directory": self.content_path.is_dir(),
        }
        return {"detected": any(indicators.values()), "indicators": indicators}

    def _get_torch(self) -> Any | None:
        if self._torch_module is not _UNSET:
            return self._torch_module
        try:
            return importlib.import_module("torch")
        except (ImportError, OSError):
            return None

    def _inspect_torch(self) -> dict[str, Any]:
        torch = self._get_torch()
        if torch is None:
            return {
                "available": False,
                "cuda_available": None,
                "device_count": None,
                "devices": [],
            }
        result: dict[str, Any] = {
            "available": True,
            "cuda_available": False,
            "device_count": 0,
            "devices": [],
        }
        try:
            cuda = torch.cuda
            result["cuda_available"] = bool(cuda.is_available())
            if result["cuda_available"]:
                count = int(cuda.device_count())
                result["device_count"] = count
                for index in range(count):
                    try:
                        name = str(cuda.get_device_name(index))
                    except Exception:
                        name = None
                    result["devices"].append({"index": index, "name": name})
        except Exception:
            # A broken/partial torch installation must not break local inspection.
            result["cuda_available"] = None
            result["device_count"] = None
            result["devices"] = []
            result["inspection_error"] = "torch_cuda_inspection_failed"
        return result

    def _find_nvidia_smi(self) -> str | None:
        if self._nvidia_smi_path is not _UNSET:
            return self._nvidia_smi_path
        return shutil.which("nvidia-smi")

    def _inspect_nvidia_smi(self) -> dict[str, Any]:
        executable = self._find_nvidia_smi()
        if not executable:
            return {"available": False, "devices": []}
        query = [
            str(executable),
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(
                query,
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return {"available": True, "query_succeeded": False, "devices": []}
        if completed.returncode != 0:
            return {"available": True, "query_succeeded": False, "devices": []}
        devices = []
        for line in completed.stdout.splitlines():
            fields = [part.strip() for part in line.split(",", 2)]
            if len(fields) != 3:
                continue
            name, total, free = fields
            devices.append({"name": name, "memory_total_mib": total, "memory_free_mib": free})
        return {"available": True, "query_succeeded": True, "devices": devices}

    def _mounted_drive_path(self) -> Path | None:
        """Return MyDrive only when the Drive mountpoint is an actual mount."""
        try:
            mounted = self._drive_mount_checker(str(self.drive_mount_path))
        except (OSError, ValueError):
            mounted = False
        my_drive = self.drive_mount_path / "MyDrive"
        if mounted and my_drive.is_dir():
            return my_drive
        return None

    def inspect_runtime(self) -> dict[str, Any]:
        """Inspect local runtime, accelerator tools, and verified Drive access."""
        detection = self.detect_runtime()
        drive_path = self._mounted_drive_path()
        torch_info = self._inspect_torch()
        nvidia_info = self._inspect_nvidia_smi()
        return {
            "provider": self.provider_name,
            "runtime_detected": detection["detected"],
            "indicators": detection["indicators"],
            "gpu": {"torch": torch_info, "nvidia_smi": nvidia_info},
            "drive_mounted": drive_path is not None,
            "checkpoint_storage": str(drive_path / "minicoder_output") if drive_path else str(self._local_storage_path()),
            "capabilities": {
                "inspect_local_runtime": True,
                "inspect_local_gpu": True,
                "persistent_drive_checkpoints": drive_path is not None,
                "generate_local_worker_command": bool(detection["detected"] and shutil.which("torchrun")),
                "execute_worker_command": False,
                "submit_cloud_job": False,
                "access_external_colab_session": False,
            },
        }

    def _local_storage_path(self) -> Path:
        if self.content_path.is_dir():
            return self.content_path / "minicoder_output"
        return Path("minicoder_output")

    def setup_environment(self):
        """Inspect only; deliberately perform no install, mount, or mutation."""
        return self.inspect_runtime()

    def get_storage_path(self) -> str:
        """Prefer Drive only when its mountpoint is verifiably mounted."""
        drive_path = self._mounted_drive_path()
        if drive_path is not None:
            return str(drive_path / "minicoder_output")
        return str(self._local_storage_path())

    def run_worker_command(self, rank: int, world_size: int, master_addr: str, master_port: int):
        """Build (but never run) a one-node command for this local Colab process.

        Multi-node execution and invocation of a separate/external Colab runtime
        are unsupported. The caller remains responsible for executing the
        returned command in this same local environment.
        """
        if not self.detect_runtime()["detected"]:
            raise NotImplementedError(
                "No local Colab runtime was detected; this adapter cannot invoke an external session."
            )
        if int(rank) != 0 or int(world_size) != 1:
            raise NotImplementedError("Only a single local worker is supported; multi-node work is unsupported.")
        executable = shutil.which("torchrun")
        if not executable:
            raise NotImplementedError("torchrun is not installed in the current process environment.")
        port = int(master_port)
        if not 1 <= port <= 65535:
            raise ValueError("master_port must be between 1 and 65535")
        args = [
            executable,
            "--nproc_per_node=1",
            "--nnodes=1",
            "--node_rank=0",
            f"--master_addr={master_addr}",
            f"--master_port={port}",
            "training/trainer.py",
        ]
        return shlex.join(args)
