# Lightning AI Studio

Lightning Studio is supported as a local execution context, not as a remote job-submission or credit-query API. This project detects local Lightning environment signals and the active PyTorch GPU. It does not infer available credits, free-tier limits, or session lifetime from those signals.

## Manual run in an already-open Studio

1. Open a Studio from the user's own Lightning AI account and select an execution resource there. Verify the current plan, GPU allocation, credit balance and any cost before starting a GPU session.
2. Put the MiniCoder project and a processed dataset on the Studio filesystem.
3. Install `requirements-training.txt` if required by that runtime.
4. Run a bounded Tiny job, for example:

```bash
MINICODER_PROVIDER=lightning python -m training.trainer \
  --config config/model_tiny.yaml --data-dir data/processed \
  --run-dir /path/to/persistent-storage/minicoder/run --max-steps 8
```

Use a mounted persistent directory for `--run-dir` if one is available in that Studio. A provider label does not make local disk persistent. `training/benchmark.py` measures only the active Studio runtime; run it there to collect GPU throughput. The task environment has no user Lightning session, so no Lightning GPU/credit behavior was tested.
