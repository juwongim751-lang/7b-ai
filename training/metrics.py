"""Append-only JSONL metrics log (read later by the dashboard)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict


class MetricsLogger:
    def __init__(self, run_dir: str, enabled: bool = True):
        self.path = Path(run_dir) / "metrics.jsonl"
        self.enabled = enabled
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, kind: str, **fields: Any) -> Dict[str, Any]:
        rec = {"t": round(time.time(), 3), "kind": kind, **fields}
        if self.enabled:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        return rec

    @staticmethod
    def read(path: str):
        p = Path(path)
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []
