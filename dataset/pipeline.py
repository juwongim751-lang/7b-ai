"""Training-data pipeline (Phase 3). Torch-free; needs numpy (+ pyarrow only for .parquet).

raw files (jsonl/json/txt/parquet/source files)
  -> normalize -> filter -> quality score -> exact dedup -> near-dup (MinHash-LSH)
  -> optional rebalance -> deterministic train/val split (hash based) -> leak check
  -> train.jsonl / val.jsonl / manifest.json / stats.json  -> (tokenize) -> train.bin / val.bin

Quality scores are HEURISTICS; they are not validated against model performance.
Scale note: dedup indexes are in-memory (fine up to ~1e6 docs; not for web-scale corpora).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
import re
import zlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import yaml

EXT_LANG = {".py": "python", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".c": "c", ".h": "c",
            ".cs": "csharp", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript",
            ".tsx": "typescript", ".java": "java", ".go": "go", ".rs": "rust", ".lua": "lua",
            ".sql": "sql", ".sh": "shell", ".ps1": "shell", ".html": "html", ".css": "css",
            ".json": "json", ".md": "markdown", ".txt": "text"}
TEXT_KEYS = ("text", "content", "code", "body")
PROMPT_KEYS = ("prompt", "instruction", "question", "explanation")
RESP_KEYS = ("completion", "response", "output", "answer", "solution")
SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__"}
CURLY = {"cpp", "c", "csharp", "javascript", "typescript", "java", "go", "rust"}
SENSITIVE_PATTERNS = (
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----", re.I)),
    ("known_api_token", re.compile(r"(?:AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|glpat-[A-Za-z0-9_-]{20,}|xox[baprs]-[A-Za-z0-9-]{16,}|sk-[A-Za-z0-9]{20,})")),
    ("credential_assignment", re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password|passwd)\b\s*[:=]\s*['\"]?(?!\$\{|<|your_|example|changeme|dummy)[A-Za-z0-9_./+=-]{16,}")),
    ("email_address", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)),
)


# ----------------------------------------------------------------------------- data model
@dataclass
class Sample:
    id: str
    text: str
    lang: str
    task: str
    source: str
    license: str
    prompt: str = ""
    quality: float = 0.0
    quality_parts: Dict[str, float] = field(default_factory=dict)


@dataclass
class DataConfig:
    sources: List[Dict[str, Any]] = field(default_factory=list)
    out_dir: str = "data/processed"
    val_ratio: float = 0.02
    min_val_samples: int = 1
    seed: int = 1337
    min_chars: int = 20
    max_chars: int = 200_000
    max_line_len: int = 2000
    max_nonprintable_ratio: float = 0.02
    min_quality: float = 0.45
    strict_syntax: bool = True          # drop python/json that definitively fail to parse
    scan_sensitive: bool = True         # conservative pattern scan; not a full DLP/PII guarantee
    drop_unknown_license: bool = True   # require provenance before a sample enters the training set
    near_dup: bool = True
    near_dup_threshold: float = 0.85
    allowed_licenses: List[str] = field(default_factory=list)   # empty = allow all (still recorded)
    balance_by: Optional[str] = None    # "lang" | "task" | None
    balance_targets: Dict[str, float] = field(default_factory=dict)
    drop_unlisted: bool = False
    share_warn: float = 0.6
    quality_weights: Dict[str, float] = field(default_factory=lambda: {
        "syntax": 0.30, "completeness": 0.15, "information": 0.20, "repetition": 0.15,
        "length": 0.10, "match": 0.10})


def load_data_config(path: str) -> DataConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    known = set(DataConfig.__dataclass_fields__)
    bad = set(raw) - known
    if bad:
        raise ValueError(f"unknown data config keys: {sorted(bad)}")
    return DataConfig(**raw)


# ----------------------------------------------------------------------------- readers
def _norm_text(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = "\n".join(l.rstrip() for l in t.split("\n")).strip("\n")
    return t + "\n" if t else ""


def sample_id(text: str) -> str:
    return hashlib.sha256(_norm_text(text).encode("utf-8", "replace")).hexdigest()


def _first(d: Dict[str, Any], keys) -> Optional[str]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _records_from_obj(o: Any) -> Iterator[Dict[str, Any]]:
    if isinstance(o, list):
        for x in o:
            yield from _records_from_obj(x)
    elif isinstance(o, dict):
        text, resp, prompt = _first(o, TEXT_KEYS), _first(o, RESP_KEYS), _first(o, PROMPT_KEYS)
        if resp is not None:
            yield {"text": resp, "prompt": prompt or "", "lang": o.get("lang") or o.get("language"),
                   "task": o.get("task"), "license": o.get("license")}
        elif text is not None:
            yield {"text": text, "prompt": "", "lang": o.get("lang") or o.get("language"),
                   "task": o.get("task"), "license": o.get("license")}
        else:  # e.g. {"data": [...]}
            for v in o.values():
                if isinstance(v, list):
                    yield from _records_from_obj(v)


def iter_file_records(path: Path, stats: Counter) -> Iterator[Dict[str, Any]]:
    ext = path.suffix.lower()
    try:
        if ext == ".jsonl":
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for n, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        yield from _records_from_obj(json.loads(line))
                    except json.JSONDecodeError:
                        stats["bad_json_line"] += 1
        elif ext == ".json":
            try:
                yield from _records_from_obj(json.loads(path.read_text(encoding="utf-8", errors="replace")))
            except json.JSONDecodeError:
                stats["bad_json_file"] += 1
        elif ext == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as e:
                raise RuntimeError("Reading .parquet needs `pip install pyarrow`") from e
            for rec in pq.read_table(str(path)).to_pylist():
                yield from _records_from_obj(rec)
        elif ext in EXT_LANG:  # source / text file = one sample
            yield {"text": path.read_text(encoding="utf-8", errors="replace"), "prompt": "",
                   "lang": EXT_LANG[ext], "task": None, "license": None}
        else:
            stats["unsupported_ext"] += 1
    except OSError:
        stats["unreadable_file"] += 1


def iter_source_files(path: str) -> Iterator[Path]:
    p = Path(path)
    if p.is_file():
        yield p
    elif p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.is_file() and not (set(f.parts) & SKIP_DIRS):
                yield f
    else:
        raise FileNotFoundError(f"source path not found: {path}")


# ----------------------------------------------------------------------------- analysis
def detect_language(text: str, hint: Optional[str] = None) -> str:
    if hint in EXT_LANG.values() or hint in ("other",):
        return hint  # type: ignore[return-value]
    pats = {"python": r"^\s*(def |class |import |from \w+ import )", "lua": r"\bfunction\b.*\bend\b|\blocal \w+\s*=",
            "cpp": r"#include\s*<|std::|\bnamespace\b", "csharp": r"\busing System|\bnamespace \w+\s*\{|\bpublic (static )?(class|void)",
            "javascript": r"\bconst \w+\s*=|=>|\bfunction \w+\(|console\.log", "java": r"\bpublic class\b|System\.out",
            "go": r"^package \w+|\bfunc \w+\(", "rust": r"\bfn \w+\(|\blet mut\b", "sql": r"(?i)\bselect\b.*\bfrom\b",
            "shell": r"^#!/bin/(ba)?sh|\becho \$"}
    scores = {l: len(re.findall(p, text, re.M)) for l, p in pats.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "text"


def _bracket_balanced(text: str, lang: str) -> bool:
    stack, pairs, i, n = [], {")": "(", "]": "[", "}": "{"}, 0, len(text)
    while i < n:
        c = text[i]
        if c == "/" and text[i:i + 2] == "//":
            i = text.find("\n", i); i = n if i < 0 else i; continue
        if c == "/" and text[i:i + 2] == "/*":
            i = text.find("*/", i + 2); i = n if i < 0 else i + 2; continue
        if c in "\"`" or (c == "'" and lang != "rust"):
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
                if c != "`" and j < n and text[j] == "\n":
                    break
            i = j + 1; continue
        if c in "([{":
            stack.append(c)
        elif c in pairs:
            if not stack or stack.pop() != pairs[c]:
                return False
        i += 1
    return not stack


def check_syntax(text: str, lang: str) -> Tuple[Optional[bool], bool]:
    """-> (ok or None if unknown, definitive). python/json are definitive; curly-language check is heuristic."""
    if lang == "python":
        try:
            ast.parse(text); return True, True
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            return False, True
    if lang == "json":
        try:
            json.loads(text); return True, True
        except ValueError:
            return False, True
    if lang in CURLY:
        return _bracket_balanced(text, lang), False
    return None, False


_PLACEHOLDER = re.compile(r"TODO|FIXME|lorem ipsum|\.\.\.\s*$|your code here|NotImplemented", re.I | re.M)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def explanation_code_overlap(prompt: str, code: str) -> float:
    """Fraction of identifier-like words in the explanation that also appear in the code (0..1).
    Weak signal only: cannot tell whether the code is semantically right."""
    pw = {w.lower() for w in _IDENT.findall(prompt)}
    cw = {w.lower() for w in _IDENT.findall(code)}
    if not pw:
        return 0.5
    return min(1.0, 2.0 * len(pw & cw) / len(pw))


def quality_score(text: str, lang: str, syntax_ok: Optional[bool], prompt: str,
                  cfg: DataConfig) -> Tuple[float, Dict[str, float]]:
    n = len(text)
    lines = [l for l in text.split("\n") if l.strip()]
    syntax = 1.0 if syntax_ok else (0.0 if syntax_ok is False else 0.6)
    completeness = 1.0
    if _PLACEHOLDER.search(text):
        completeness -= 0.4
    if lang in CURLY and syntax_ok is False:
        completeness -= 0.3
    if n and not text.endswith("\n"):
        completeness -= 0.1
    completeness = max(0.0, completeness)
    if n >= 200:
        ratio = len(zlib.compress(text.encode("utf-8", "replace"), 6)) / max(1, len(text.encode("utf-8", "replace")))
        if ratio < 0.25:
            info = max(0.0, (ratio - 0.03) / 0.22)
        elif ratio > 0.7:
            info = max(0.0, 1 - (ratio - 0.7) / 0.3)
        else:
            info = 1.0
    else:
        info = 0.7
    repetition = 1.0 - (1 - len(set(lines)) / len(lines)) if len(lines) > 3 else 1.0
    # length is only a sanity band: NOT monotonic, long files are never rewarded
    length = 1.0 if 80 <= n <= 20_000 else (n / 80 if n < 80 else 20_000 / n)
    match = explanation_code_overlap(prompt, text) if prompt else 1.0
    parts = {"syntax": syntax, "completeness": completeness, "information": round(info, 3),
             "repetition": round(repetition, 3), "length": round(length, 3), "match": round(match, 3)}
    w = cfg.quality_weights
    score = sum(w[k] * parts[k] for k in w) / sum(w.values())
    return round(score, 4), parts


def reject_reason(text: str, cfg: DataConfig) -> Optional[str]:
    if not text.strip():
        return "empty"
    if len(text) < cfg.min_chars:
        return "too_short"
    if len(text) > cfg.max_chars:
        return "too_long"
    if "\x00" in text:
        return "binary"
    if max(len(l) for l in text.split("\n")) > cfg.max_line_len:
        return "long_line_minified_or_generated"
    bad = sum(1 for c in text if (ord(c) < 32 and c not in "\n\t") or c == "\ufffd")
    if bad / len(text) > cfg.max_nonprintable_ratio:
        return "non_printable"
    return None


def sensitive_content_flags(text: str) -> List[str]:
    """Return category names only; never return or log the matched sensitive value."""
    return [name for name, pattern in SENSITIVE_PATTERNS if pattern.search(text)]


# ----------------------------------------------------------------------------- near-duplicate index
_P = (1 << 31) - 1


class NearDupIndex:
    """MinHash (num_perm) + LSH banding. Hashes use crc32 (stable across runs, unlike hash())."""

    def __init__(self, threshold: float = 0.85, num_perm: int = 64, bands: int = 16, seed: int = 1):
        assert num_perm % bands == 0
        rng = np.random.RandomState(seed)
        self.a = rng.randint(1, _P, size=num_perm).astype(np.int64)
        self.b = rng.randint(0, _P, size=num_perm).astype(np.int64)
        self.threshold, self.bands, self.rows = threshold, bands, num_perm // bands
        self.sigs: List[np.ndarray] = []
        self.buckets: Dict[Tuple[int, bytes], List[int]] = defaultdict(list)

    def signature(self, text: str) -> np.ndarray:
        toks = re.findall(r"\w+|[^\w\s]", text)
        k = 5
        grams = [" ".join(toks[i:i + k]) for i in range(max(1, len(toks) - k + 1))] or [text]
        h = np.fromiter((zlib.crc32(g.encode("utf-8", "replace")) % _P for g in set(grams)), dtype=np.int64)
        return ((self.a[:, None] * h[None, :] + self.b[:, None]) % _P).min(axis=1)

    def check_and_add(self, text: str) -> bool:
        """True if a near-duplicate was already indexed (then `text` is NOT added)."""
        sig = self.signature(text)
        keys = [(i, sig[i * self.rows:(i + 1) * self.rows].tobytes()) for i in range(self.bands)]
        cands = {c for key in keys for c in self.buckets.get(key, ())}
        for c in cands:
            if float((self.sigs[c] == sig).mean()) >= self.threshold:
                return True
        idx = len(self.sigs)
        self.sigs.append(sig)
        for key in keys:
            self.buckets[key].append(idx)
        return False


# ----------------------------------------------------------------------------- stages
def load_samples(cfg: DataConfig, removed: Counter) -> List[Sample]:
    out: List[Sample] = []
    for src in cfg.sources:
        name = src.get("name") or Path(src["path"]).name
        for f in iter_source_files(src["path"]):
            for rec in iter_file_records(f, removed):
                lic = rec.get("license") or src.get("license") or "unknown"
                if cfg.drop_unknown_license and str(lic).strip().lower() in {
                    "", "unknown", "noassertion", "no assertion", "unlicensed", "no license", "none"
                }:
                    removed["license_unknown"] += 1
                    continue
                if cfg.allowed_licenses and lic not in cfg.allowed_licenses:
                    removed["license_not_allowed"] += 1
                    continue
                text = _norm_text(rec["text"])
                why = reject_reason(text, cfg)
                if why:
                    removed[why] += 1
                    continue
                lang = detect_language(text, rec.get("lang") or src.get("lang"))
                syntax_ok, definitive = check_syntax(text, lang)
                if cfg.strict_syntax and definitive and syntax_ok is False:
                    removed["bad_syntax"] += 1
                    continue
                prompt = _norm_text(rec["prompt"]) if rec.get("prompt") else ""
                if cfg.scan_sensitive:
                    flags = sensitive_content_flags(text + "\n" + prompt)
                    if flags:
                        for flag in flags:
                            removed["sensitive_" + flag] += 1
                        continue
                q, parts = quality_score(text, lang, syntax_ok, prompt, cfg)
                if q < cfg.min_quality:
                    removed["low_quality"] += 1
                    continue
                out.append(Sample(sample_id(text if not prompt else prompt + "\x01" + text), text, lang,
                                  rec.get("task") or src.get("task") or "code", name, lic, prompt, q, parts))
    return out


def dedup(samples: List[Sample], cfg: DataConfig, removed: Counter) -> List[Sample]:
    samples = sorted(samples, key=lambda s: (-s.quality, s.id))  # keep best copy
    seen, kept = set(), []
    idx = NearDupIndex(cfg.near_dup_threshold) if cfg.near_dup else None
    for s in samples:
        if s.id in seen:
            removed["exact_dup"] += 1
            continue
        seen.add(s.id)
        if idx is not None and idx.check_and_add((s.prompt + "\n" if s.prompt else "") + s.text):
            removed["near_dup"] += 1
            continue
        kept.append(s)
    return kept


def rebalance(samples: List[Sample], cfg: DataConfig, removed: Counter) -> List[Sample]:
    if not cfg.balance_by or not cfg.balance_targets:
        return samples
    key = {"lang": lambda s: s.lang, "task": lambda s: s.task}[cfg.balance_by]
    groups: Dict[str, List[Sample]] = defaultdict(list)
    for s in samples:
        groups[key(s)].append(s)
    tw = {k: w for k, w in cfg.balance_targets.items() if w > 0 and groups.get(k)}
    if not tw:
        return samples
    total_w = sum(tw.values())
    n_max = min(len(groups[k]) / (w / total_w) for k, w in tw.items())  # largest N feasible for all targets
    out: List[Sample] = []
    for k, w in tw.items():
        quota = max(1, int(n_max * w / total_w))
        out.extend(groups[k][:quota])  # groups are already quality-sorted
        removed["rebalance_trim"] += len(groups[k]) - min(quota, len(groups[k]))
    for k, g in groups.items():
        if k not in tw:
            if cfg.drop_unlisted:
                removed["rebalance_unlisted"] += len(g)
            else:
                out.extend(g)
    return out


def split_key(sid: str, seed: int) -> float:
    return int(hashlib.sha256(f"{seed}:{sid}".encode()).hexdigest()[:8], 16) / 2 ** 32


def split(samples: List[Sample], cfg: DataConfig) -> Tuple[List[Sample], List[Sample]]:
    val = [s for s in samples if split_key(s.id, cfg.seed) < cfg.val_ratio]
    need = cfg.min_val_samples if len(samples) >= 2 else 0
    if len(val) < need:
        rest = sorted((s for s in samples if s not in val), key=lambda s: split_key(s.id, cfg.seed))
        val += rest[:need - len(val)]
    vids = {s.id for s in val}
    train = [s for s in samples if s.id not in vids]
    assert_no_leak(train, val)
    return train, val


def assert_no_leak(train: List[Sample], val: List[Sample]) -> None:
    tid = {s.id for s in train}
    common = tid & {s.id for s in val}
    if common:
        raise AssertionError(f"{len(common)} samples appear in both train and val")


def dataset_stats(samples: List[Sample], share_warn: float) -> Dict[str, Any]:
    def dist(f):
        c = Counter(f(s) for s in samples)
        return {k: {"count": v, "share": round(v / max(1, len(samples)), 4)} for k, v in c.most_common()}
    lens = np.array([len(s.text) for s in samples]) if samples else np.array([0])
    q = np.array([s.quality for s in samples]) if samples else np.array([0.0])
    st = {"samples": len(samples), "by_language": dist(lambda s: s.lang), "by_task": dist(lambda s: s.task),
          "by_source": dist(lambda s: s.source), "by_license": dist(lambda s: s.license),
          "chars_p5_p50_p95": [int(np.percentile(lens, p)) for p in (5, 50, 95)],
          "quality_mean": round(float(q.mean()), 4), "quality_p5": round(float(np.percentile(q, 5)), 4)}
    st["warnings"] = [f"{dim[3:]} '{k}' has {v['share']:.0%} of samples (> {share_warn:.0%})"
                      for dim in ("by_language", "by_task") for k, v in st[dim].items() if v["share"] > share_warn]
    if "unknown" in st["by_license"]:
        st["warnings"].append("some samples have unknown license: record licenses in config sources")
    return st


def write_jsonl(path: Path, samples: List[Sample]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for s in samples:
            f.write(json.dumps({"id": s.id, "text": s.text, "prompt": s.prompt, "lang": s.lang, "task": s.task,
                                "source": s.source, "license": s.license, "quality": s.quality},
                               ensure_ascii=False) + "\n")


def build(cfg: DataConfig) -> Dict[str, Any]:
    removed: Counter = Counter()
    samples = load_samples(cfg, removed)
    n_loaded = len(samples)
    samples = dedup(samples, cfg, removed)
    samples = rebalance(samples, cfg, removed)
    train, val = split(samples, cfg)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    version = hashlib.sha256("".join(sorted(s.id for s in train + val)).encode()).hexdigest()[:16]
    manifest = {"dataset_version": version, "config": asdict(cfg), "loaded_after_filters": n_loaded,
                "removed": dict(removed), "train": dataset_stats(train, cfg.share_warn),
                "val": dataset_stats(val, cfg.share_warn),
                "sources": [{k: s.get(k) for k in ("name", "path", "license", "url", "task")} for s in cfg.sources]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


# ----------------------------------------------------------------------------- tokenization to shards
def format_document(tk, rec: Dict[str, Any]) -> List[int]:
    """[BOS] [lang] ([instruction] prompt [response])? text [EOS] (special ids from tokenizer)."""
    ids = [t for t in (tk.bos_token_id, tk.lang_token_id(rec["lang"])) if t is not None]
    if rec.get("prompt"):
        ids += [tk.token_id("<|instruction|>")] + tk.encode(rec["prompt"]) + [tk.token_id("<|response|>")]
    ids += tk.encode(rec["text"])
    if tk.eos_token_id is not None:
        ids.append(tk.eos_token_id)
    return ids


def tokenize_split(jsonl: Path, tk, out_bin: Path, dataset_version: str = "") -> Dict[str, Any]:
    vocab = tk.get_vocab_size()
    dtype = np.uint16 if vocab <= 65535 else np.uint32
    n_tok = n_doc = 0
    with out_bin.open("wb") as f:
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ids = format_document(tk, json.loads(line))
            if max(ids) >= vocab:
                raise ValueError("token id out of range of vocab")
            np.asarray(ids, dtype=dtype).tofile(f)
            n_tok += len(ids); n_doc += 1
    tokenizer_config = {"algorithm": "byte_fallback" if tk.is_fallback else "byte_level_bpe",
                        "requested_vocab_size": getattr(tk, "vocab_size", None),
                        "actual_vocab_size": vocab, "fingerprint": tk.fingerprint(),
                        "special_tokens_schema_version": 1}
    meta = {"file": out_bin.name, "dtype": np.dtype(dtype).name, "n_tokens": n_tok, "n_docs": n_doc,
            "vocab_size": vocab, "tokenizer_fingerprint": tk.fingerprint(),
            "tokenizer_config": tokenizer_config, "tokenizer_fallback": tk.is_fallback,
            "dataset_version": dataset_version}
    out_bin.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def update_manifest_token_counts(data_dir: str | Path, split_meta: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Atomically attach verified shard/tokenizer counts to an existing dataset manifest."""
    root = Path(data_dir)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = manifest.get("dataset_version")
    if not split_meta or not {"train", "val"}.issubset(split_meta):
        raise ValueError("token counts require train and val split metadata")
    normalized = {}
    fingerprints = set()
    for split in ("train", "val"):
        item = dict(split_meta[split])
        if item.get("dataset_version") != version:
            raise ValueError(f"{split} shard belongs to a different dataset version")
        fingerprints.add(item.get("tokenizer_fingerprint"))
        normalized[split] = {"file": item["file"], "n_tokens": int(item["n_tokens"]),
                             "n_docs": int(item["n_docs"]), "dtype": item["dtype"],
                             "tokenizer_fingerprint": item["tokenizer_fingerprint"],
                             "tokenizer_config": item.get("tokenizer_config"),
                             "tokenizer_fallback": bool(item.get("tokenizer_fallback", False))}
    if len(fingerprints) != 1 or None in fingerprints:
        raise ValueError("train/val shards must use the same tokenizer fingerprint")
    manifest["tokenization"] = {"tokenizer_fingerprint": next(iter(fingerprints)),
                                "vocab_size": int(split_meta["train"]["vocab_size"]),
                                "tokenizer_config": split_meta["train"].get("tokenizer_config"),
                                "fallback": bool(split_meta["train"].get("tokenizer_fallback", False)),
                                "splits": normalized,
                                "total_tokens": sum(item["n_tokens"] for item in normalized.values())}
    tmp = manifest_path.with_name(manifest_path.name + f".tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, manifest_path)
    finally:
        tmp.unlink(missing_ok=True)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="MiniCoder data pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--config", default="config/data.yaml")
    t = sub.add_parser("tokenize")
    t.add_argument("--data-dir", default="data/processed"); t.add_argument("--tokenizer", default="dataset/tokenizer.json")
    a = ap.parse_args()
    if a.cmd == "build":
        m = build(load_data_config(a.config))
        print(json.dumps({k: m[k] for k in ("dataset_version", "removed")}, indent=2))
        print("train:", m["train"]["samples"], "val:", m["val"]["samples"])
        for w in m["train"]["warnings"]:
            print("WARNING:", w)
    else:
        from dataset.tokenizer import MiniCoderTokenizer
        tk = MiniCoderTokenizer(); tk.load(a.tokenizer)
        d = Path(a.data_dir); ver = json.loads((d / "manifest.json").read_text(encoding="utf-8"))["dataset_version"]
        split_meta = {}
        for name in ("train", "val"):
            split_meta[name] = tokenize_split(d / f"{name}.jsonl", tk, d / f"{name}.bin", ver)
            print(name, split_meta[name])
        updated = update_manifest_token_counts(d, split_meta)
        print("manifest total_tokens:", updated["tokenization"]["total_tokens"])


if __name__ == "__main__":
    main()
