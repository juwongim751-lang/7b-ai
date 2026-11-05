"""Safe, resumable checkpoint management for MiniCoder training."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

_STEP_RE = re.compile(r"checkpoint_step_(\d+)\.pt$")
_SUPPORTED_FORMAT_VERSIONS = {2, 3}


class CheckpointManager:
    """Atomic checkpoints with step ordering, checksums, metadata and best retention.

    ``torch.load`` is intentionally used with ``weights_only=False`` because optimizer,
    scheduler and Python/NumPy RNG states are part of a resumable training state. The
    file is loaded only from the manager's own checkpoint directory and its SHA-256 is
    verified before deserialization.
    """

    def __init__(self, checkpoint_dir="checkpoints", keep=3, metadata: Optional[Dict[str, Any]] = None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep = max(1, int(keep))
        self.metadata = metadata or {}
        self.warning_log = self.checkpoint_dir / "checkpoint_warnings.log"

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): CheckpointManager._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [CheckpointManager._jsonable(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return value

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    def _warn(self, message: str) -> None:
        with self.warning_log.open("a", encoding="utf-8") as fh:
            fh.write(message + "\n")
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    def _atomic_write_bytes(self, target: Path, data: bytes) -> None:
        tmp = target.with_name(target.name + f".tmp.{os.getpid()}")
        with tmp.open("wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
        self._fsync_directory(target.parent)

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        fd = os.open(str(directory), flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _atomic_torch_save(self, state: Dict[str, Any], target: Path) -> None:
        tmp = target.with_name(target.name + f".tmp.{os.getpid()}")
        with tmp.open("wb") as fh:
            torch.save(state, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
        self._fsync_directory(target.parent)

    @staticmethod
    def _step(path: Path) -> int:
        match = _STEP_RE.match(path.name)
        return int(match.group(1)) if match else -1

    def _checkpoint_files(self):
        return sorted((p for p in self.checkpoint_dir.glob("checkpoint_step_*.pt") if _STEP_RE.match(p.name)),
                      key=self._step)

    def save(self, model, optimizer, scheduler, step, loss, epoch, scaler=None,
             batch_in_epoch=0, metadata: Optional[Dict[str, Any]] = None,
             val_loss: Optional[float] = None, tokens_seen: Optional[int] = None):
        merged_meta = dict(self.metadata)
        merged_meta.update(metadata or {})
        model_cfg = self._jsonable(merged_meta.get("model_config", {}))
        train_cfg = self._jsonable(merged_meta.get("train_config", {}))
        tokenizer_cfg = self._jsonable(merged_meta.get("tokenizer_config"))
        state = {
            "format_version": 3,
            "step": int(step), "epoch": int(epoch), "batch_in_epoch": int(batch_in_epoch),
            "tokens_seen": int(tokens_seen) if tokens_seen is not None else None,
            "random_seed": train_cfg.get("seed"),
            "model_config": model_cfg,
            "tokenizer_config": tokenizer_cfg,
            "training_config": train_cfg,
            "bitnet_config": {"use_bitnet": bool(model_cfg.get("use_bitnet", False)),
                              "activation_bits": model_cfg.get("activation_bits", 8)},
            "loss": float(loss) if loss is not None else None,
            "val_loss": float(val_loss) if val_loss is not None else None,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
            "rng": {"torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "numpy": np.random.get_state(), "python": random.getstate()},
            "metadata": self._jsonable(merged_meta),
        }
        target = self.checkpoint_dir / f"checkpoint_step_{int(step):09d}.pt"
        self._atomic_torch_save(state, target)
        digest = self._sha256(target)
        self._atomic_write_bytes(target.with_suffix(target.suffix + ".sha256"), (digest + "  " + target.name + "\n").encode())
        manifest = {"file": target.name, "step": int(step), "sha256": digest,
                    "metadata": self._jsonable(merged_meta), "val_loss": state["val_loss"]}
        self._atomic_write_bytes(target.with_suffix(target.suffix + ".manifest.json"),
                                 (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
        self._prune()
        return str(target)

    def _verify(self, path: Path) -> None:
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if not sidecar.exists():
            raise IOError(f"missing checksum for {path.name}")
        expected = sidecar.read_text(encoding="utf-8").split()[0]
        actual = self._sha256(path)
        if expected != actual:
            raise IOError(f"checksum mismatch for {path.name}: expected {expected}, got {actual}")

    def _prune(self) -> None:
        files = self._checkpoint_files()
        # Keep the newest N by step, plus the best checkpoint if one exists.
        best = self._best_file(files)
        keep = set(files[-self.keep:])
        if best is not None:
            keep.add(best)
        for old in files:
            if old not in keep:
                old.unlink(missing_ok=True)
                old.with_suffix(old.suffix + ".sha256").unlink(missing_ok=True)
                old.with_suffix(old.suffix + ".manifest.json").unlink(missing_ok=True)

    def _best_file(self, files=None):
        candidates = []
        for path in files if files is not None else self._checkpoint_files():
            manifest = path.with_suffix(path.suffix + ".manifest.json")
            try:
                val = json.loads(manifest.read_text()).get("val_loss") if manifest.exists() else None
                if val is not None:
                    candidates.append((float(val), self._step(path), path))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return min(candidates)[2] if candidates else None

    def save_best(self, *args, **kwargs):
        """Compatibility helper: save a checkpoint carrying ``val_loss``."""
        if kwargs.get("val_loss") is None and len(args) < 11:
            raise ValueError("save_best requires val_loss")
        return self.save(*args, **kwargs)

    def _validate_metadata(self, actual: Dict[str, Any], expected: Optional[Dict[str, Any]], force_resume: bool):
        if not expected:
            return
        mismatches = {}
        for key, want in expected.items():
            have = actual.get(key)
            if want is not None and have != want:
                mismatches[key] = {"checkpoint": have, "current": want}
        if mismatches:
            message = "checkpoint metadata mismatch: " + json.dumps(mismatches, sort_keys=True, default=str)
            if not force_resume:
                raise ValueError(message + "; use force_resume=True/--force-resume to override")
            self._warn("FORCED RESUME: " + message)

    def load_latest(self, model, optimizer, scheduler=None, scaler=None, device="cpu",
                    expected_metadata: Optional[Dict[str, Any]] = None, force_resume: bool = False):
        files = self._checkpoint_files()
        if not files:
            return {"step": 0, "epoch": 0, "batch_in_epoch": 0, "loss": None, "checkpoint_path": None}
        bad = []
        state = None
        chosen = None
        # Highest step wins; corrupt files fall back to the previous valid step.
        for path in reversed(files):
            try:
                self._verify(path)
                candidate = torch.load(path, map_location=device, weights_only=False)
                version = int(candidate.get("format_version", 0))
                if version not in _SUPPORTED_FORMAT_VERSIONS:
                    raise ValueError(f"unsupported checkpoint format_version={version}; supported={sorted(_SUPPORTED_FORMAT_VERSIONS)}")
                self._validate_metadata(candidate.get("metadata", {}), expected_metadata, force_resume)
                state, chosen = candidate, path
                break
            except ValueError:
                # A valid checkpoint with incompatible semantics must not be silently
                # treated as corruption or replaced by an older incompatible checkpoint.
                raise
            except Exception as exc:
                bad.append(f"{path.name}: {exc}")
        if state is None:
            raise RuntimeError("no valid checkpoint remains; " + "; ".join(bad))
        if bad:
            self._warn("Ignoring corrupt checkpoint(s): " + "; ".join(bad) + f"; resumed from {chosen.name}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        if scaler and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        rng = state.get("rng", {})
        if rng.get("torch") is not None: torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("cuda") is not None: torch.cuda.set_rng_state_all(rng["cuda"])
        if rng.get("numpy") is not None: np.random.set_state(rng["numpy"])
        if rng.get("python") is not None: random.setstate(rng["python"])
        print(f"Resumed {chosen} at step {state['step']}")
        return {"step": int(state["step"]), "epoch": int(state["epoch"]),
                "batch_in_epoch": int(state.get("batch_in_epoch", 0)), "loss": state.get("loss"),
                "tokens_seen": state.get("tokens_seen"), "val_loss": state.get("val_loss"),
                "checkpoint_path": str(chosen), "metadata": state.get("metadata", {})}
