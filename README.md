# MiniCoder

MiniCoder is a research-oriented decoder-only Transformer project with CPU-verifiable Tiny training, integrity-checked checkpoints, optional BitNet-style QAT and packed inference export, measured model planning, local runtime inspection, and Colab/Kaggle notebook workflows.

## Cloud boundary

Kaggle is the exception to inspection-only provider adapters: `providers/kaggle_jobs.py` uses the **official Kaggle CLI** to prepare, submit, inspect status and download kernel output. Submit requires an explicit confirmation flag because it may consume GPU quota. Kaggle credentials stay in the user's preconfigured CLI environment; this project does not request or display their secret values. The dashboard has matching protected Kaggle endpoints.

Colab runtime API/MCP is documented in the official-source research, but this repository does not create Colab runtimes through an external OAuth client. The provided notebook is started by the user and mounts Drive interactively. Lightning remote submission is not integrated. Modal invocation remains behind explicit consent and user-supplied current rate/budget. ZeroGPU remains a short, pre-defined app-function path, not a general training job API. No actual external account, provider GPU, quota, balance or cloud job has been tested.

## Quick start

```bash
python -m pip install -r requirements-training.txt
python -m training.trainer --smoke-test
python -m unittest discover -s tests -v
```

The synthetic smoke path demonstrates Tiny CPU training, evaluation, checkpoint save and resume. It does not establish model quality. For data, measured batch probing, recovery, dashboard and provider workflows, see [`README_TRAINING.md`](README_TRAINING.md) and [`CLOUD_SETUP.md`](CLOUD_SETUP.md).

## Project map

- `model/`: Transformer, BitLinear, quantization, base-3 packed ternary codec, export and loader.
- `dataset/`: license-attested local registry, opt-in public streaming, filtering, manifest and token packing.
- `training/`: training loop, recovery, cost estimate guard, measured auto batch probe, memory planner and benchmark.
- `providers/`: runtime detectors, inspection adapters, official Kaggle CLI client and actual-process scheduler.
- `controller/`, `workers/`, `dashboard/`: worker safety, local controls and Kaggle job controls.
- `notebooks/`: sequential Colab and Kaggle Tiny training, checkpoint/resume and export.
- `docs/`, `reports/`: audits, official policy research, validation and measured outputs.

## Model and compute limits

`model_tiny.yaml` is intentionally smaller than 10M for fast smoke tests. `model_tiny_10m.yaml` and 100M–7B presets represent architecture sizes, not completed or loadable pretrained models. Architecture is not silently reduced. BitNet-style QAT retains floating-point master weights; optional packed ternary files are inference-only reference artifacts without an accelerated integer kernel. See [`BITNET_VALIDATION.md`](BITNET_VALIDATION.md).

A 7B configuration does not mean a free runtime can load/train 7B. Inspect the conservative estimator and run an actual workload-sized probe on the target device. Free GPU access, credentials, quota, runtime lifetime and storage persistence depend on the user's account and may change; never infer them from a provider environment marker. See [`GPU_BENCHMARK_REPORT.md`](GPU_BENCHMARK_REPORT.md).
