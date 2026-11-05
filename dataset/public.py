"""Explicit, license-recorded public Hugging Face dataset ingestion.

The caller must verify the dataset's actual license and provide its official
source URL. This helper does not scrape the web, use gated credentials, or run
remote dataset scripts.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ALLOWED_LICENSES = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause",
                    "CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0", "CC-BY-3.0",
                    "CC-BY-SA-3.0", "Unlicense", "ODC-BY-1.0"}


def load_public_hf_dataset(dataset_id: str, output_jsonl, *, license_id: str,
                           source_url: str, split="train", text_field="text",
                           max_samples=100_000, revision=None):
    """Stream up to max_samples into JSONL and persist public-source provenance."""
    if license_id not in ALLOWED_LICENSES:
        raise ValueError("license_id must be a reviewed permissive license identifier")
    parsed = urlparse(source_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("source_url must be an official HTTPS dataset URL")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?", dataset_id):
        raise ValueError("dataset_id must be an org/name or dataset name")
    if not isinstance(max_samples, int) or not 1 <= max_samples <= 1_000_000:
        raise ValueError("max_samples must be in [1, 1000000]")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install the optional datasets package to use public HF datasets") from exc
    # Public-only streaming. No user token is read, and dataset Python code is disabled.
    stream = load_dataset(dataset_id, split=split, streaming=True, revision=revision,
                          trust_remote_code=False)
    output = Path(output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = dropped_empty = scanned = 0
    with output.open("w", encoding="utf-8", newline="\n") as target:
        for row in stream:
            scanned += 1
            value = row.get(text_field) if isinstance(row, dict) else None
            if not isinstance(value, str) or not value.strip():
                dropped_empty += 1
                continue
            record = {"text": value, "source": dataset_id, "license": license_id,
                      "source_url": source_url, "split": split}
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if count >= max_samples:
                break
            if scanned >= max_samples * 10:
                break
    provenance = {
        "dataset_id": dataset_id, "revision": revision, "split": split,
        "source_url": source_url, "license": license_id,
        "license_verified_by": "caller_attestation_not_automatically_verified",
        "text_field": text_field, "samples_written": count, "rows_scanned": scanned,
        "empty_or_missing_field_dropped": dropped_empty,
        "streaming": True, "trust_remote_code": False, "credential_used": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_jsonl": output.name,
    }
    manifest = output.with_suffix(output.suffix + ".source.json")
    manifest.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return provenance
