"""Small local registry for user-supplied datasets and provenance."""
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from dataset.public import ALLOWED_LICENSES


class DatasetRegistry:
    def __init__(self, registry_path="data/dataset_registry.json"):
        self.path = Path(registry_path)

    def _read(self):
        try:
            doc = json.loads(self.path.read_text(encoding="utf-8"))
            return doc if isinstance(doc, dict) else {}
        except FileNotFoundError:
            return {}

    def list(self):
        return sorted(self._read().values(), key=lambda item: item.get("name", ""))

    def register(self, name, path, license_id, source_url=None, revision=None, notes=""):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", str(name)):
            raise ValueError("name must be a short alphanumeric identifier")
        if license_id not in ALLOWED_LICENSES:
            raise ValueError("unknown or non-reviewed license; dataset not registered")
        data_path = Path(path).expanduser().resolve(strict=True)
        if not data_path.is_file() and not data_path.is_dir():
            raise ValueError("dataset path must be a regular file or directory")
        if source_url:
            parsed = urlparse(source_url)
            if parsed.scheme != "https" or not parsed.netloc:
                raise ValueError("source_url must be HTTPS")
        entry = {"name": str(name), "path": str(data_path), "license": license_id,
                 "source_url": source_url, "revision": revision,
                 "license_verified_by": "caller_attestation_not_automatically_verified",
                 "notes": str(notes)[:1000], "registered_at_utc": datetime.now(timezone.utc).isoformat()}
        records = self._read()
        records[entry["name"]] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(records, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
        finally:
            Path(tmp_name).unlink(missing_ok=True)
        return entry
