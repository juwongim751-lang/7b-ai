# MiniCoder Release Test Report

**Run date:** 2026-11-05 · **Environment:** Linux x86_64 sandbox, Python 3.12.3, PyTorch 2.15.0+cu132, CPU-only (`torch.cuda.is_available() == False`).

## Results

| Check | Result | Scope |
|---|---|---|
| `python -m unittest discover -s tests -v` | **128 tests, OK** | Local CPU and mock/injected cloud/tool APIs. Exact final log: `/tmp/minicoder_v2_final_release_tests.log` (ephemeral test log, not shipped). |
| `python -m tests.integration_test` | **6/6 PASS** | Tokenizer, model fwd/bwd, checkpoint roundtrip, agent/RAG/memory/tools, review/dashboard controls, worker/scheduler. |
| `python -m compileall -q .` | **PASS** | All tracked Python source in project tree. |
| `node --check dashboard/frontend/app.js` | **PASS** | Dashboard client JavaScript syntax. |
| Colab/Kaggle notebooks | **PASS** | Valid nbformat 4 JSON, 7 sequential cells each, embedded code AST syntax after removing IPython magics. Not executed in provider runtime. |
| Kaggle CLI wrapper | **PASS — mocked only** | Requires consent, validates `kernel-metadata.json` and local paths, uses official command shape without shell; submit/status/output subprocess mocked. |
| Auto batch probe | **PASS — CPU** | Real small-model fwd/bwd candidate probe; model weights unchanged and report created in the probe implementation tests. Not a GPU OOM/performance test. |
| Dashboard | **PASS — local API/process mocks** | Auth protected routes, planner response, cost confirmation, start/restore/export controls and Kaggle route wiring. No browser visual test. |
| Source artifact scan | **PASS** | No `.pt`, `.ckpt`, `.env` files found inside project tree. |

## Test coverage additions

New coverage includes ternary packing and export/load equality, training memory estimation and user-confirmed planner fallback, measured automatic microbatch probe, static provider resource honesty, Kaggle official CLI prepare/submit/status/output and consent, cost estimate guard, dataset license/provenance/streaming behavior, checkpoint v3 metadata and unknown format rejection, graceful SIGTERM checkpoint boundary, dashboard local restore/export and request authentication.

## Not run / not established

- No active CUDA device, GPU VRAM, GPU utilization, or CUDA training/performance was available.
- No Kaggle, Colab, Lightning, Hugging Face or Modal user credential was supplied or used; no remote submission, GPU allocation, account quota, balance or billing was inspected.
- Kaggle CLI behavior was mocked; no actual Kaggle kernel was submitted.
- Colab/Kaggle notebook cells were not executed in a live hosted notebook.
- No 100M, 300M, 1B, 3B or 7B model was trained; no 100B-token run or model-quality evaluation was performed.
- Cross-provider artifact transfer was tested on temporary local directories, not across accounts/services.

A passing synthetic or mocked test demonstrates only the asserted path, never provider access, GPU availability, successful full training or useful model quality.
