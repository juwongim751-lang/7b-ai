# MiniCoder 7B / Cloud V2 — 최종 구현 보고서

**기준일: 2026-11-05.** 첨부 프로젝트를 확장하고 검사 가능한 CPU·mock 결과를 기록했습니다. 이 문서는 CPU 또는 mocked provider test를 live cloud GPU test로 간주하지 않습니다. 요구 파일별 분석은 [`docs/PROJECT_AUDIT.md`](docs/PROJECT_AUDIT.md), 공식 Cloud V2 API 연구는 [`docs/CLOUD_V2_PROVIDER_API_RESEARCH.md`](docs/CLOUD_V2_PROVIDER_API_RESEARCH.md), 제품별 설정은 [`CLOUD_SETUP.md`](CLOUD_SETUP.md) 참고.

## 판정 요약

| 범위 | 상태 | 실제 근거와 미완성 경계 |
|---|---|---|
| Existing code 유지·통합 | **PASS** | 전체 회귀·신규 unittest **128개 OK**, integration scenario **6/6 PASS**. Python compile, dashboard JS syntax, notebook JSON/code syntax PASS. |
| Kaggle remote job path | **PARTIAL — 구현, live 미검증** | 공식 CLI `kernels push/status/output` 클라이언트, protected dashboard endpoint와 explicit submit/quota consent. subprocess는 mock only. 실제 Kaggle account/job/quota/GPU는 테스트하지 않음. |
| Colab | **PARTIAL** | 사용자 runtime/Drive mount notebook, checkpoint resume·export 흐름. Official Colab API/MCP의 외부 runtime 실행은 조사했지만 OAuth client integration은 미구현. live notebook test 없음. |
| Lightning AI | **PARTIAL** | 로컬 provider inspection/문서. remote Job submitter 미구현, Studio/account test 없음. |
| Hugging Face ZeroGPU | **PARTIAL** | 짧은 app function/demo·inspection, policy research. 임의 장기 train submit 아님; real Space 호출 미검증. |
| Modal | **PARTIAL** | 선택적 GPU app 코드, explicit user consent와 사용자 제공 hourly rate/USD budget guard. 외부 container/run/billing/GPU 미실행. |
| Provider inspection/scheduler | **PARTIAL** | local + 5 cloud inspection adapters. Resource scheduler는 현재 process에서 GPU/RAM/storage 요건을 관찰하고, 다른 머신 VRAM 합산과 remote auto launch를 하지 않음. Quota/cost/throughput/session-aware cross-provider job queue·failover 미구현. Kaggle 공식 CLI는 별도 consent-based submit path. |
| Resource plan / auto batch | **PASS — 제한 범위** | Manifest tokens와 실제 process 자원 기반 heuristic model plan 및 7B fallback 확인. `--auto-batch-size`는 제한된 local microbatch를 actual fwd/bwd probe 후 측정하고 RNG/weights를 보존. CPU probe test PASS. 실제 target CUDA OOM/throughput fit 미확인. |
| BitNet-style QAT | **PARTIAL** | absmean ternary weight, symmetric per-token activation fake quantization, STE reference path 및 Transformer 통합. Floating master weights, integer fused kernel 아님. |
| Ternary packed inference export | **PASS — reference format only** | 5 ternary weights/base-3/uint8 (이론상 순수 payload 1.6 bit/weight), unpack/output equality와 export-load 테스트. Inference-only; optimizer/resume state 또는 accelerated integer GEMM은 아님. |
| Tiny training/resume | **PASS — mechanics only** | CPU synthetic smoke, validation, loss decrease, resume 8→10. 10.05M 구조 짧은 toy CPU run. 언어/코딩 품질 주장은 없음. |
| Checkpoint/recovery | **PASS — CPU/local/mock scope** | Checkpoint v3 model/optimizer/scheduler/scaler/RNG/seed/tokenizer/config/BitNet/token count metadata, checksum/manifest, corrupt fallback, atomic local transfer 테스트. 실제 계정 간 이동 미수행. |
| Dataset provenance | **PARTIAL** | local Dataset registry, license/source attribution, unknown license default exclusion, public Hugging Face streaming opt-in, remote code disabled, bounded samples, pattern secret/email filtering. license 자동 검증 또는 완전 PII/DLP는 아님. |
| Dashboard | **PASS — API/control; cloud test 없음** | 인증된 local trainer Start/Pause/Resume/Stop/Checkpoint/Restore/Export, runtime telemetry/provider plan, Kaggle official CLI prepare/submit/status/output. remote Kaggle submit explicit confirmation required. Browser visual test 미수행. |
| 7B / 100B tokens | **BLOCKED in this runtime** | 환경에 active CUDA GPU 및 user cloud creds 없음. 해당 규모를 실행·완료했다고 주장하지 않음. |

## 구조 규모

| Preset | Actual parameters | 학습 결과 |
|---|---:|---|
| Tiny smoke | 361,088 | 짧은 합성 CPU smoke |
| Tiny 10M | 10,054,272 | 짧은 toy CPU update/checkpoint |
| 100M | 98,841,088 | 미학습 |
| 300M | 304,773,888 | 미학습 |
| 1B | 1,003,554,816 | 미학습 |
| 3B | 2,952,416,256 | 미학습 |
| 7B | 7,241,732,096 | 미학습 |

모델 YAML은 architecture/preset일 뿐 hardware fit 또는 pretrained weights를 보장하지 않습니다. Approximate memory와 제약은 [`GPU_BENCHMARK_REPORT.md`](GPU_BENCHMARK_REPORT.md).

## Release validation

| 명령 또는 확인 | 결과 |
|---|---|
| `python -m unittest discover -s tests -v` | **128 tests OK** |
| `python -m tests.integration_test` | **6/6 PASS** |
| `python -m compileall -q .` | **PASS** |
| `node --check dashboard/frontend/app.js` | **PASS** |
| Colab/Kaggle notebooks | nbformat 4 valid, 7 cells each, Python AST syntax PASS; **실제 Colab/Kaggle에서 실행하지 않음** |
| Kaggle official CLI path | metadata/path/consent/command tests PASS with mocks; **실제 API/계정 호출 없음** |
| Current runtime | PyTorch 2.15.0+cu132; CUDA 13.2 runtime available in build, **`torch.cuda.is_available() == False`** |

Release test scope/detail는 [`TEST_REPORT.md`](TEST_REPORT.md), active training state는 [`TRAINING_STATUS.md`](TRAINING_STATUS.md).

## Actual CPU benchmark

10M config, batch 1 × sequence 16, 5 updates, 80 measured tokens, 0.440046 seconds, **181.8 tokens/s**, final synthetic loss 6.02891. GPU, VRAM, bandwidth, provider account/quota는 N/A/not tested. 100B-token을 이 작은 CPU probe 속도로 단순 나누면 17.43년이지만 7B training forecast가 아니고 operational estimation으로 사용할 수 없습니다. Source JSON/summary: [`reports/gpu_benchmark.json`](reports/gpu_benchmark.json), [`reports/gpu_benchmark.md`](reports/gpu_benchmark.md).

## Cost·security·external limits

Rate/budget guard는 사용자 현재 단가 입력 기반 estimate이며 provider billing hard cap이 아닙니다. Kaggle CLI submit도 사용자 API token 준비와 GPU quota confirmation을 요구하며, 실제 계정 접근이 검증되지 않았습니다. Colab runtime OAuth, Lightning SDK remote submit, generalized multi-provider scheduler, live cross-provider transfer, GPU performance fitting, real 7B training, quality benchmark는 아직 남은 work입니다. Provider별 구현/미구현은 [`PROVIDER_STATUS.md`](PROVIDER_STATUS.md), test detail은 [`TEST_REPORT.md`](TEST_REPORT.md), setup은 [`CLOUD_SETUP.md`](CLOUD_SETUP.md).