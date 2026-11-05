---
title: MiniCoder inference demo
emoji: null
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---

# MiniCoder ZeroGPU inference demo

This Space entrypoint is for short inference/demo requests only. It is not a long-running pretraining worker, and the code does not query account quotas or submit jobs to other providers. Hugging Face may require a ZeroGPU-enabled Space configuration and apply account- and time-dependent usage limits; verify the current terms in the account UI before deployment.

Place an exported MiniCoder model under `model/` with `config.json`, `model.safetensors`, and `tokenizer.json`. Create the weights with `ModelExporter` and copy the tokenizer file explicitly; the model exporter does not package a tokenizer. Use only model artifacts whose license allows the intended public demo. The loader accepts safetensors only; it does not unpickle uploaded `.pt` checkpoints.

The Space invokes `spaces.GPU` only around the inference function and limits generation to 64 new tokens. Actual GPU allocation, cold start, queueing and per-account limits are not validated by local tests. Set the Space's ZeroGPU hardware in its settings; the local code cannot enable it remotely.

## Local validation

From the project root, install `hf_space/requirements.txt`, copy exported assets into `hf_space/model/`, then run `python hf_space/app.py`. This uses local CPU/GPU if available and does not test ZeroGPU service behavior.
