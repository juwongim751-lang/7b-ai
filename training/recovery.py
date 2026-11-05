"""Integrity-checked checkpoint discovery and portable local transfer helpers.

Cross-provider transport is intentionally a local directory operation here: callers
must arrange an authorized mount/upload/download and then use these verifiers.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

from training.checkpoint import CheckpointManager


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def list_valid_checkpoints(checkpoint_dir: str | Path) -> List[Dict[str, Any]]:
    """Return newest-first checkpoints having a valid digest and readable manifest."""
    root = Path(checkpoint_dir).resolve()
    manager = CheckpointManager(root)
    valid = []
    for path in reversed(manager._checkpoint_files()):
        try:
            manager._verify(path)
            manifest_path = path.with_suffix(path.suffix + ".manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("file") != path.name or manifest.get("sha256") != _digest(path):
                raise IOError("manifest does not match checkpoint")
            valid.append({"path": str(path), "step": manager._step(path), "sha256": manifest["sha256"],
                          "size_bytes": path.stat().st_size,
                          "val_loss": manifest.get("val_loss"), "integrity": "verified"})
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return valid


def transfer_checkpoint(source: str | Path, destination_dir: str | Path) -> Dict[str, Any]:
    """Copy one verified .pt checkpoint and sidecars atomically into a destination.

    A transfer is not marked successful until the destination digest and manifest
    both match the source. Existing files are never overwritten with partial data.
    """
    src = Path(source).resolve(strict=True)
    if not src.is_file() or not src.name.startswith("checkpoint_step_") or src.suffix != ".pt":
        raise ValueError("source must be a checkpoint_step_*.pt file")
    manager = CheckpointManager(src.parent)
    manager._verify(src)
    src_manifest = src.with_suffix(src.suffix + ".manifest.json")
    manifest = json.loads(src_manifest.read_text(encoding="utf-8"))
    source_digest = _digest(src)
    if manifest.get("file") != src.name or manifest.get("sha256") != source_digest:
        raise IOError("source manifest/checksum mismatch")

    dst_root = Path(destination_dir).resolve()
    dst_root.mkdir(parents=True, exist_ok=True)
    outputs = []
    for origin in (src, src.with_suffix(src.suffix + ".sha256"), src_manifest):
        if not origin.is_file():
            raise IOError(f"checkpoint sidecar missing: {origin.name}")
        target = dst_root / origin.name
        tmp = target.with_name(target.name + f".partial.{os.getpid()}")
        try:
            with origin.open("rb") as read_stream, tmp.open("wb") as write_stream:
                shutil.copyfileobj(read_stream, write_stream, length=1024 * 1024)
                write_stream.flush()
                os.fsync(write_stream.fileno())
            if _digest(tmp) != _digest(origin):
                raise IOError(f"copy verification failed: {origin.name}")
            os.replace(tmp, target)
            outputs.append(target)
        finally:
            tmp.unlink(missing_ok=True)

    dest = dst_root / src.name
    CheckpointManager(dst_root)._verify(dest)
    copied_manifest = json.loads(dest.with_suffix(dest.suffix + ".manifest.json").read_text(encoding="utf-8"))
    if copied_manifest.get("sha256") != _digest(dest) or copied_manifest.get("file") != dest.name:
        raise IOError("destination manifest validation failed")
    return {"path": str(dest), "step": manager._step(src), "sha256": _digest(dest),
            "size_bytes": dest.stat().st_size, "integrity": "verified", "status": "complete"}
