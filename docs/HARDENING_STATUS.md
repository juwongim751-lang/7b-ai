# MiniCoder hardening status

## Resolved issues

1. **Worker communication**: `workers/worker.py` now performs real `POST /workers/register` and `POST /workers/heartbeat` requests. The server issues the worker ID; the worker no longer creates a local UUID as a fake success. GPU/runtime/network/backend capabilities and heartbeat updates are persisted in `WorkerRegistry`.
2. **Distributed scheduling**: `GPUScheduler` no longer sums unrelated VRAM into an FSDP group. FSDP candidates must share `network_id`, runtime, distributed backend, have recent heartbeats, and explicitly report `reachable=true`. Otherwise the scheduler returns `INDEPENDENT_TASKS`.
3. **Execution security**: `agent/tools.py` and `agent/quality.py` use argv with `shell=False`, command allowlists, repository-root path checks, timeouts, output limits, and approval for Python/repository-changing commands. There is no `shell=True` in agent/worker/controller/dashboard code. This is shell-injection protection, not a full VM/container isolation boundary; production untrusted code should run in a container/VM with a dedicated service account and network policy.
4. **Semantic RAG**: `agent/rag.py` now has an `EmbeddingBackend` interface and a real `SentenceTransformerEmbeddingBackend`. With `--embedding-model`, index/search stores normalized embeddings and combines cosine semantic score with lexical score. Without that optional dependency/configuration, results are explicitly labeled `lexical-only`.
5. **Dashboard frontend**: `dashboard/frontend/index.html`, `app.js`, and `style.css` provide a browser UI for status polling, workers, metrics, and start/pause/resume/checkpoint/stop controls. FastAPI serves it at `/` and `/dashboard/static`.

## Verification

`tests/test_hardening.py` covers real HTTP worker registration/heartbeat, incompatible-vs-compatible scheduler decisions, shell-free approval boundaries, and semantic backend ranking. Combined with the existing suites:

```text
32 tests: OK
compileall: PASS
frontend mount: PASS
shell=True scan: ABSENT
```

The SentenceTransformer model download/live embedding path was not run in this sandbox because the optional package/model is not installed. The fake embedding backend test verifies the replacement contract and ranking logic; install `sentence-transformers` and pass `--embedding-model all-MiniLM-L6-v2` for live semantic indexing.


> **후속 업데이트:** 이후 WorkerRegistry secret allowlist, worker HTTPS/retry 제한, dashboard auth/XSS와 storage gate가 추가됐습니다. 최신 검증·한계는 `FINAL_IMPLEMENTATION_REPORT.md`, `TRAINING_STATUS.md`, `docs/PROJECT_AUDIT.md`를 기준으로 보세요.
