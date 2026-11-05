"""Byte-level BPE tokenizer for code (Phase 2).

Fixes vs. the original: ByteLevel initial alphabet (no [UNK] for unseen bytes, lossless decode),
code/instruction/FIM/language special tokens, optional BOS/EOS, fingerprint, token statistics.
Fallback (no `tokenizers` package) is a lossless byte tokenizer for smoke tests ONLY.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Dict, Iterable, List, Optional

try:
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer
except ImportError:  # pragma: no cover
    Tokenizer = None

LANGUAGES = ["python", "cpp", "c", "csharp", "javascript", "typescript", "java", "go", "rust", "lua",
             "sql", "shell", "html", "css", "json", "markdown", "text", "other"]
# Order matters: ids 0..3 stay [PAD],[UNK],[BOS],[EOS] as in the original code.
SPECIAL_TOKENS = (["[PAD]", "[UNK]", "[BOS]", "[EOS]",
                   "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|file_sep|>",
                   "<|instruction|>", "<|response|>", "<|code_start|>", "<|code_end|>"]
                  + [f"<|lang:{l}|>" for l in LANGUAGES])


class _ByteTokenizer:
    """Lossless dependency-free fallback. NOT a BPE; vocab = specials + 256 bytes."""

    def __init__(self) -> None:
        self.offset = len(SPECIAL_TOKENS)
        self.ids = {t: i for i, t in enumerate(SPECIAL_TOKENS)}

    def get_vocab_size(self) -> int:
        return self.offset + 256

    def token_to_id(self, token: str) -> Optional[int]:
        return self.ids.get(token)

    def encode(self, text: str):
        return type("Encoding", (), {"ids": [b + self.offset for b in text.encode("utf-8")]})()

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return bytes(int(i) - self.offset for i in ids if self.offset <= int(i) < self.offset + 256
                     ).decode("utf-8", errors="replace")

    def to_str(self) -> str:
        return json.dumps({"type": "byte_fallback", "specials": SPECIAL_TOKENS})

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_str())


class MiniCoderTokenizer:
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self._fallback = Tokenizer is None
        if self._fallback:
            self.tokenizer = _ByteTokenizer()
        else:
            self.tokenizer = self._new_bpe()

    @staticmethod
    def _new_bpe():
        tk = Tokenizer(BPE(unk_token="[UNK]"))
        tk.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tk.decoder = ByteLevelDecoder()
        return tk

    @property
    def is_fallback(self) -> bool:
        return self._fallback

    # ---- training -------------------------------------------------------
    def train_from_iterator(self, texts: Iterable[str]) -> None:
        if self._fallback:
            print("WARNING: `tokenizers` not installed; byte fallback in use (smoke tests only).")
            return
        trainer = BpeTrainer(vocab_size=self.vocab_size, min_frequency=2, special_tokens=SPECIAL_TOKENS,
                             initial_alphabet=ByteLevel.alphabet(), show_progress=False)
        self.tokenizer.train_from_iterator(texts, trainer)

    def train(self, files) -> None:
        files = [str(f) for f in files if os.path.isfile(f)]
        if not files:
            raise ValueError("No readable training text files were provided")

        def gen():
            for f in files:
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    yield fh.read()
        self.train_from_iterator(gen())

    # ---- io -------------------------------------------------------------
    def save(self, path: str = "dataset/tokenizer.json") -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.tokenizer.save(path)

    def load(self, path: str = "dataset/tokenizer.json") -> None:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        try:
            meta = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} is not valid tokenizer JSON: {e}") from e
        if meta.get("type") == "byte_fallback":
            self.tokenizer, self._fallback = _ByteTokenizer(), True
        elif Tokenizer is not None:
            self.tokenizer, self._fallback = Tokenizer.from_file(path), False
        else:
            raise RuntimeError("This tokenizer requires `pip install tokenizers`.")

    def fingerprint(self) -> str:
        return hashlib.sha256(self.tokenizer.to_str().encode("utf-8")).hexdigest()[:16]

    # ---- use ------------------------------------------------------------
    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = list(self.tokenizer.encode(text).ids)
        if add_bos and self.bos_token_id is not None:
            ids = [self.bos_token_id] + ids
        if add_eos and self.eos_token_id is not None:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def token_id(self, token: str) -> Optional[int]:
        return self.tokenizer.token_to_id(token)

    def lang_token_id(self, lang: str) -> Optional[int]:
        return self.token_id(f"<|lang:{lang if lang in LANGUAGES else 'other'}|>")

    @property
    def pad_token_id(self): return self.token_id("[PAD]")
    @property
    def bos_token_id(self): return self.token_id("[BOS]")
    @property
    def eos_token_id(self): return self.token_id("[EOS]")

    def token_stats(self, texts: Iterable[str]) -> Dict[str, float]:
        n_docs = n_tok = n_bytes = unk = bad_rt = 0
        unk_id = self.token_id("[UNK]")
        for t in texts:
            ids = self.encode(t)
            n_docs += 1
            n_tok += len(ids)
            n_bytes += len(t.encode("utf-8"))
            unk += sum(1 for i in ids if i == unk_id)
            bad_rt += int(self.decode(ids) != t)
        return {"docs": n_docs, "tokens": n_tok, "bytes": n_bytes, "unk_tokens": unk,
                "roundtrip_failures": bad_rt, "bytes_per_token": round(n_bytes / max(1, n_tok), 3),
                "tokens_per_doc": round(n_tok / max(1, n_docs), 1), "vocab_size": self.get_vocab_size()}


def _jsonl_texts(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                yield (r.get("prompt") or "") + "\n" + r["text"] if r.get("prompt") else r["text"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a MiniCoder BPE tokenizer from a processed jsonl")
    ap.add_argument("--input", required=True, help="train.jsonl produced by dataset.pipeline")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--output", default="dataset/tokenizer.json")
    a = ap.parse_args()
    tk = MiniCoderTokenizer(a.vocab_size)
    tk.train_from_iterator(_jsonl_texts(a.input))
    tk.save(a.output)
    print(json.dumps(tk.token_stats(_jsonl_texts(a.input)), indent=2))
    print("fingerprint:", tk.fingerprint(), "fallback:", tk.is_fallback)


if __name__ == "__main__":
    main()
