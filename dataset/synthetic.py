"""Synthetic corpus for MECHANICS smoke tests only (not real training data)."""
from __future__ import annotations

import json
import random
from pathlib import Path

OPS = ["+", "-", "*"]


def write_smoke_corpus(root: str, n: int = 160, seed: int = 0) -> dict:
    rnd = random.Random(seed)
    r = Path(root)
    (r / "code").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        a, b, op = rnd.choice("xyzabc"), rnd.choice("pqrstu"), rnd.choice(OPS)
        k = rnd.randint(2, 99)
        body = (f"def op_{i}_{a}{b}({a}, {b}):\n    \"\"\"Compute {a} {op} {b} plus {k}.\"\"\"\n"
                f"    total = {a} {op} {b}\n    for step in range({k % 7 + 2}):\n        total += step * {k}\n"
                f"    return total\n\n\nclass Holder{i}:\n    def __init__(self, v):\n        self.v = v + {k}\n"
                f"    def get(self):\n        return self.v * {i + 1}\n")
        (r / "code" / f"m{i}.py").write_text(body, encoding="utf-8")
    for i in range(n // 4):
        (r / "code" / f"f{i}.js").write_text(
            f"function calc{i}(a, b) {{\n  const k = {i * 3 + 1};\n  if (a > b) {{\n    return a * k;\n  }}\n"
            f"  return [a, b].map(x => x + k).reduce((s, x) => s + x, 0);\n}}\nmodule.exports = {{ calc{i} }};\n",
            encoding="utf-8")
    (r / "code" / "dup.py").write_text((r / "code" / "m0.py").read_text(encoding="utf-8"), encoding="utf-8")   # exact dup
    (r / "code" / "near.py").write_text((r / "code" / "m1.py").read_text(encoding="utf-8") + "# note\n", encoding="utf-8")  # near dup
    (r / "code" / "broken.py").write_text("def broken_function(:\n    total = 0\n    for i in range(10):\n        total += i\n    return total\n", encoding="utf-8")  # syntax error
    (r / "code" / "empty.py").write_text("\n\n", encoding="utf-8")
    pairs = [{"prompt": f"Write a function op_{i} that adds {i} to a number.",
              "completion": f"def op_{i}(n):\n    return n + {i}\n\n\nprint(op_{i}(1))\n", "task": "function", "license": "CC0"}
             for i in range(300, 330)]
    with (r / "pairs.jsonl").open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    return {"planted": {"exact_dup": 1, "near_dup": 1, "bad_syntax": 1, "empty": 1}}
