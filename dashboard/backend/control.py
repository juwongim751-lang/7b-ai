from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


class TrainingProcessManager:
    """Manage one local trainer process and its file-based controls safely."""

    def __init__(self, project_root="."):
        self.root = Path(project_root).resolve()
        self.process: Optional[subprocess.Popen] = None
        self.run_dir: Optional[Path] = None
        self.command = None
        self._start_spec = None

    def _inside_root(self, value: str | Path, label: str, must_exist: bool = False) -> Path:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve(strict=must_exist)
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"{label} must remain inside the project directory") from exc
        return resolved

    def _flag(self, name: str) -> Path:
        if not self.run_dir:
            raise RuntimeError("training has not been started")
        return self.run_dir / name

    @staticmethod
    def _validated_extra_args(extra_args: Optional[Sequence[str]]) -> list[str]:
        """Accept only numeric/runtime controls; no arbitrary paths, shell or credentials."""
        args = list(extra_args or [])
        allowed_value_flags = {"--max-steps", "--device"}
        allowed_bool_flags = {"--resume", "--force-resume", "--smoke-test"}
        out = []
        i = 0
        while i < len(args):
            flag = args[i]
            if flag in allowed_bool_flags:
                out.append(flag)
                i += 1
                continue
            if flag not in allowed_value_flags or i + 1 >= len(args):
                raise ValueError(f"unsupported extra training argument: {flag!r}")
            value = args[i + 1]
            if flag == "--max-steps":
                try:
                    if int(value) < 1:
                        raise ValueError
                except (TypeError, ValueError) as exc:
                    raise ValueError("--max-steps must be a positive integer") from exc
                value = str(int(value))
            elif value not in {"auto", "cpu", "cuda"}:
                raise ValueError("--device must be auto, cpu, or cuda")
            out.extend((flag, value))
            i += 2
        return out

    def start(self, config="config/model_tiny.yaml", train_config=None, run_dir="runs/dashboard",
              resume=False, extra_args=None, gpu_cost_rate_usd_per_hour=None,
              max_budget_usd=None, confirm_potential_cost=False):
        if self.process and self.process.poll() is None:
            raise RuntimeError("training is already running")
        cost_enabled = gpu_cost_rate_usd_per_hour is not None or max_budget_usd is not None
        if cost_enabled:
            if gpu_cost_rate_usd_per_hour is None or max_budget_usd is None:
                raise ValueError("paid run requires both GPU cost rate and budget cap")
            if float(gpu_cost_rate_usd_per_hour) <= 0 or float(max_budget_usd) <= 0:
                raise ValueError("GPU cost rate and budget cap must be positive")
            if not confirm_potential_cost:
                raise PermissionError("paid run requires explicit confirm_potential_cost=true")
        elif confirm_potential_cost:
            raise ValueError("cost confirmation requires an explicit rate and budget")
        config_path = self._inside_root(config, "model config", must_exist=True)
        if not config_path.is_file() or config_path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError("model config must be an existing YAML file")
        train_path = None
        if train_config:
            train_path = self._inside_root(train_config, "training config", must_exist=True)
            if not train_path.is_file() or not train_path.suffix.lower() in {".yaml", ".yml"}:
                raise ValueError("training config must be an existing YAML file")
        self.run_dir = self._inside_root(run_dir, "run directory")
        if self.run_dir == self.root:
            raise ValueError("run directory cannot be the project root")
        self.run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [sys.executable, "-m", "training.trainer", "--config", str(config_path)]
        cmd += ["--run-dir", str(self.run_dir)]
        if train_path:
            cmd += ["--train-config", str(train_path)]
        if resume:
            cmd += ["--resume"]
        cmd += self._validated_extra_args(extra_args)
        if cost_enabled:
            cmd += ["--gpu-cost-rate-usd-per-hour", str(float(gpu_cost_rate_usd_per_hour)),
                    "--max-budget-usd", str(float(max_budget_usd)), "--confirm-potential-cost"]
        self.command = cmd
        self._start_spec = {"config": str(config_path.relative_to(self.root)),
                            "train_config": str(train_path.relative_to(self.root)) if train_path else None,
                            "run_dir": str(self.run_dir.relative_to(self.root)), "extra_args": list(extra_args or []),
                            "gpu_cost_rate_usd_per_hour": gpu_cost_rate_usd_per_hour,
                            "max_budget_usd": max_budget_usd,
                            "confirm_potential_cost": bool(confirm_potential_cost)}
        log_path = self.run_dir / "dashboard_process.log"
        with log_path.open("a", encoding="utf-8") as log:
            self.process = subprocess.Popen(cmd, cwd=self.root, stdout=log, stderr=subprocess.STDOUT,
                                            start_new_session=(os.name != "nt"))
        return self.status()

    def stop(self):
        self._flag("STOP").touch()
        return self.status()

    def pause(self):
        self._flag("PAUSE").touch()
        return self.status()

    def resume(self):
        self._flag("PAUSE").unlink(missing_ok=True)
        return self.status()

    def checkpoint(self):
        self._flag("CHECKPOINT").touch()
        return self.status()

    def _checkpoint_dir(self):
        if self.run_dir is None:
            raise RuntimeError("training has not been started")
        applied_path = self.run_dir / "applied_config.json"
        checkpoint_dir = self.run_dir / "checkpoints"
        if applied_path.is_file():
            try:
                applied = json.loads(applied_path.read_text(encoding="utf-8"))
                configured = applied.get("train", {}).get("checkpoint_dir")
                if configured:
                    checkpoint_dir = self._inside_root(configured, "checkpoint directory", must_exist=False)
            except (OSError, ValueError, TypeError):
                pass
        return checkpoint_dir

    def _latest_checkpoint(self):
        from training.recovery import list_valid_checkpoints
        if self.run_dir is None:
            raise RuntimeError("training has not been started")
        valid = list_valid_checkpoints(self._checkpoint_dir())
        if not valid:
            raise FileNotFoundError("no integrity-verified checkpoint exists for this run")
        return valid[0]

    def restore(self):
        """Start a new local trainer process resuming its newest verified checkpoint."""
        if self.process and self.process.poll() is None:
            raise RuntimeError("stop/pause the current process before restoring a new process")
        checkpoint = self._latest_checkpoint()
        if not self._start_spec:
            raise RuntimeError("original training configuration is unavailable")
        if self.run_dir:
            for name in ("STOP", "PAUSE", "CHECKPOINT"):
                (self.run_dir / name).unlink(missing_ok=True)
        spec = dict(self._start_spec)
        spec["resume"] = True
        result = self.start(**spec)
        result["restored_checkpoint"] = checkpoint
        return result

    def export_model(self, output_dir=None, pack_bitnet=False):
        """Export the latest verified local checkpoint to a project-contained directory."""
        checkpoint = self._latest_checkpoint()
        output = self._inside_root(output_dir or (self.run_dir / "export"), "export directory")
        cmd = [sys.executable, "-m", "model.export_cli", "--checkpoint", checkpoint["path"],
               "--output-dir", str(output)]
        if pack_bitnet:
            cmd.append("--pack-bitnet")
        completed = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True,
                                   timeout=3600, check=False)
        log_path = self.run_dir / "export_process.log"
        log_path.write_text((completed.stdout or "") + "\n--- STDERR ---\n" + (completed.stderr or ""),
                            encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"model export failed (exit {completed.returncode}); see {log_path.name}")
        return {"status": "exported", "output_dir": str(output), "source_checkpoint": checkpoint,
                "packed_ternary": bool(pack_bitnet), "stdout_tail": completed.stdout[-2000:]}

    def _metrics(self):
        if not self.run_dir:
            return []
        p = self.run_dir / "metrics.jsonl"
        if not p.exists():
            return []
        out = []
        for line in p.read_text(encoding="utf-8").splitlines()[-100:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def status(self):
        metrics = self._metrics()
        latest = next((item for item in reversed(metrics) if item.get("kind") == "train"), {})
        checkpoint = next((item for item in reversed(metrics) if item.get("kind") == "checkpoint"), {})
        running = bool(self.process and self.process.poll() is None)
        return {"status": "running" if running else
                ("finished" if self.process and self.process.poll() == 0 else "idle"),
                "pid": self.process.pid if self.process else None,
                "returncode": self.process.poll() if self.process else None,
                "run_dir": str(self.run_dir) if self.run_dir else None,
                "command": self.command,
                "control_files": {x: (self.run_dir / x).exists()
                                  for x in ("STOP", "PAUSE", "CHECKPOINT")} if self.run_dir else {},
                "latest_metric": latest, "latest_event": metrics[-1] if metrics else {},
                "checkpoint_status": "saved" if checkpoint else "not_saved",
                "checkpoint_path": checkpoint.get("path") or latest.get("checkpoint_path"),
                "recent_metrics": metrics}

    def close(self):
        if self.process and self.process.poll() is None:
            self.stop()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    os.killpg(self.process.pid, signal.SIGTERM)
                else:
                    self.process.terminate()
