"""End-to-end CPU smoke test of the Phase 2-4 pipeline on a SYNTHETIC corpus.
Proves mechanics only (data -> tokenizer -> shards -> train -> eval -> checkpoint -> resume). It says nothing
about model quality. Results are computed from real checks, not printed unconditionally."""
from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path

from dataset.pipeline import DataConfig, build, tokenize_split
from dataset.synthetic import write_smoke_corpus
from dataset.tokenizer import MiniCoderTokenizer
from model.config import load_config
from training.config import TrainConfig
from training.trainer import build_from_dirs


def run_smoke_test(workdir: str | None = None, keep: bool = False) -> bool:
    root = Path(workdir or tempfile.mkdtemp(prefix="minicoder_smoke_"))
    results = {}
    try:
        write_smoke_corpus(str(root / "raw"), n=160)
        cfg = DataConfig(sources=[{"path": str(root / "raw" / "code"), "name": "synthetic_code", "license": "CC0"},
                                  {"path": str(root / "raw" / "pairs.jsonl"), "name": "synthetic_pairs", "license": "CC0"}],
                         out_dir=str(root / "data"), val_ratio=0.1, min_val_samples=5)
        man = build(cfg)
        results["data_build"] = man["train"]["samples"] > 50 and man["val"]["samples"] >= 5
        results["filters_hit"] = all(k in man["removed"] for k in ("exact_dup", "bad_syntax")) and \
            (man["removed"].get("near_dup", 0) >= 1)
        tk = MiniCoderTokenizer(vocab_size=1000)
        tk.train_from_iterator(json.loads(l)["text"] for l in (root / "data" / "train.jsonl").read_text("utf-8").splitlines())
        tk.save(str(root / "tokenizer.json"))
        stats = tk.token_stats(["def f(x):\n    return x  # 한글 😀\n"])
        results["tokenizer_roundtrip"] = stats["roundtrip_failures"] == 0 and stats["unk_tokens"] == 0
        for sp in ("train", "val"):
            tokenize_split(root / "data" / f"{sp}.jsonl", tk, root / "data" / f"{sp}.bin", man["dataset_version"])
        mcfg = load_config("config/model_tiny.yaml")
        tcfg = TrainConfig(data_dir=str(root / "data"), run_dir=str(root / "run"), seq_len=64, micro_batch_size=2,
                           grad_accum=2, max_steps=8, warmup_steps=2, eval_every=4, eval_batches=4, save_every=4,
                           log_every=2, learning_rate=3e-3, device="cpu")
        s1 = build_from_dirs(mcfg, tcfg).train()
        results["train_loss_finite"] = math.isfinite(s1["last_train_loss"]) and s1["final_step"] == 8
        results["eval_ran"] = bool(s1["final_eval"]) and math.isfinite(s1["final_eval"]["val_ppl"])
        results["training_state_logged"] = (s1["global_step"] == 8 and s1["tokens_seen"] > 0 and
                                             s1["total_tokens_target"] >= s1["tokens_seen"] and
                                             isinstance(s1["provider_name"], str) and bool(s1["provider_name"]) and
                                             Path(s1["checkpoint_path"]).is_file())
        metrics = [json.loads(l) for l in (root / "run" / "metrics.jsonl").read_text().splitlines()]
        tl = [m["loss"] for m in metrics if m["kind"] == "train"]
        results["loss_decreased"] = len(tl) >= 2 and tl[-1] < tl[0]
        tcfg2 = TrainConfig(**{**tcfg.__dict__, "max_steps": 10, "resume": True})
        s2 = build_from_dirs(load_config("config/model_tiny.yaml"), tcfg2).train()
        results["resume_continues"] = s2["final_step"] == 10 and s2["steps_this_run"] == 2
        results["resume_preserves_state"] = (s2["tokens_seen"] > s1["tokens_seen"] and
                                             Path(s2["checkpoint_path"]).is_file() and
                                             s2["checkpoint_path"].endswith("checkpoint_step_000000010.pt"))
    except Exception as e:  # report, never hide
        results["exception"] = repr(e)
        import traceback; traceback.print_exc()
    ok = bool(results) and all(v is True for v in results.values())
    print(json.dumps({"smoke_results": results, "all_passed": ok, "workdir": str(root)}, indent=2))
    if not keep and workdir is None:
        shutil.rmtree(root, ignore_errors=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run_smoke_test() else 1)
