# MiniCoder 기존 프로젝트 분석 및 통합 점검

**기준일:** 2026-11-05. 본 문서는 첨부된 `MiniCoder_7B_hardened.zip`을 풀어 기존 구조를 보존한 뒤 현재 작업 트리와 실제 테스트를 확인한 결과입니다. `agent/`, 데이터 관리/생성기, 기존 tokenizer/packed dataset, model, training, provider, controller, workers, dashboard 등 원래 구성 요소를 삭제하지 않았습니다. 요구된 핵심 파일은 아래처럼 확인했습니다.

| 파일 | 현재 동작 | 주요 한계·다음 개선 | 연결·검증 |
|---|---|---|---|
| `providers/base.py` | 5개 클라우드 adapter와 local의 공통 inspection 계약, capability/unsupported 표현 | 공통 계약은 실제 cloud job lifecycle이나 사용자별 quota API를 제공하지 않음 | `providers/registry.py`, `tests/test_provider_*.py` |
| `providers/kaggle.py` | Kaggle 런타임 마커, GPU/스토리지/credential 존재 여부를 검사; 비밀값은 반환하지 않음 | Kaggle account에서 원격 notebook을 시작하거나 quota를 읽지 않음. Kaggle API 호출은 이 작업에서 수행 안 함 | `providers/detector.py`, `providers/credentials.py`, `tests/test_provider_kaggle.py` |
| `training/trainer.py` | 실제 autoregressive 학습·gradient accumulation·AMP·validation·scheduler·clip·control files·resume; step/loss/token/provider/resource 메트릭 저장 | 미니 학습은 검증했지만 언어모델 품질·7B 학습·실제 GPU를 검증하지 않음. OOM은 조언을 기록하고 종료하며 무인 자동 축소/재시작은 하지 않음 | model, packed dataset, `checkpoint.py`, `resource_manager.py`; smoke/integration test |
| `training/checkpoint.py` | 주기/최종 checkpoint에 model·optimizer·scheduler·scaler·RNG·metadata를 한 `.pt`에 저장; 원자 저장, SHA-256 sidecar, manifest, 최근/validation-best 보존, 손상본 건너뛰기 | checkpoint는 신뢰되지 않은 임의 출처에서 load하면 안 됨(`weights_only=False`는 optimizer/RNG 복구용). 별도 DataLoader worker 내부 상태는 저장하지 않음 | `recovery.py`; `tests/test_recovery.py`, smoke resume |
| `training/distributed.py` | 레거시 단일 노드 NCCL/FSDP wrap helper | 현재 통합 trainer가 직접 사용하는 일반 경로는 별도의 `dist_utils.py` DDP이다. FSDP helper에는 CPU fallback·실제 멀티노드 controller 연계가 없음. 멀티 GPU/FSDP 실기기 시험 안 함 | `dist_utils.py`, torch distributed; 현재 보고서에서 live multi-GPU는 NOT TESTED |
| `training/resource_manager.py` | 가능한 경우 RAM, CUDA memory, NVML utilization, step/data-wait 시간을 측정; 불가능 값은 null; probe로 후보 batch/sequence 측정 | auto-tune probe는 독립 유틸이며 모든 후보를 검증/자동적용하지 않음; CPU/GPU 종류별 성능주장으로 확대 불가 | `trainer.py`, `auto_tune.py`; mock/CPU unit tests 및 benchmark |
| `training/auto_tune.py` | 주어진 후보들 forward/backward 실측, 성공·실패·권장 후보를 JSON으로 기록 | trainer 시작 전후의 완전 자동 튜닝·실제 memory margin 기반 online 조절은 미구현; 사용자 검토 없이 구조를 바꾸지 않음 | `resource_manager.probe_training_config`; probe tests |
| `workers/worker.py` | HTTP 등록·heartbeat, GPU 보고, 환경변수 자격 토큰, 비루프백은 HTTPS 강제, bounded heartbeat retry/backoff | agent는 worker 상태만 등록하며 원격 training process를 실행하거나 서로 다른 cloud network를 붙이지 않음 | FastAPI worker endpoints; HTTP mock integration tests |
| `controller/registry.py` | 서버 발급 worker ID, VRAM·address·backend 제약, allowlisted metadata만 저장, heartbeat 만료 | 메모리 내 registry이며 durable database가 아님. 등록 정보는 여전히 remote worker가 주장하므로 실제 네트워크 reachability를 증명하지 않음 | `workers/worker.py`, scheduler; tamper/secret test |
| `controller/scheduler.py` | 최근 heartbeat와 network/runtime/backend 호환성을 확인해 single GPU, 호환 분산 그룹 또는 독립 작업으로 분류; unrelated VRAM 합산 금지 | 가격, quota, 예측 throughput, 종료까지 남은 세션, 저장공간, priority로 점수를 매기는 task queue와 자동 failover는 없음 | `registry.py`; worker scheduler integration test |
| `model/transformer.py` | Transformer/GQA·causal attention·KV cache·generate; 설정에 따라 기존 linear 대신 BitLinear 경로 | BitNet은 전체 tensor/optimizer/checkpoint를 1.58-bit로 저장하지 않음. 정규화·embedding 등 다른 부분은 별도 정밀도이며 현재 BitNet은 교육용 reference path | `model/config.py`, `bitlinear.py`, `quantization.py`; forward/backward·quantization tests |
| `model/config.py` | validation이 있는 모델 구성과 Tiny 10M, 100M, 300M, 1B, 3B, 7B preset; 실제 parameter count 계산 | preset은 구조 크기만 정의하며 학습 완료/pretrained 능력을 뜻하지 않음. `model_tiny.yaml` 361,088 파라미터는 smoke용, 10M preset은 별도 | model YAMLs; parameter-count tests |
| `dataset/pipeline.py` | 사용자 로컬 source ingest, 형식/품질/문법·정확/유사 중복 filtering, license provenance, secret/email 패턴 검사, 결정적 train/val split, shard tokenization과 token-count manifest | 공개 데이터셋 탐색·다운로드·압축해제는 자동화하지 않음. 패턴 스캐너는 DLP/PII 완전 탐지기가 아니며 false positive/negative가 가능. 품질 score는 heuristic | tokenizer, packed dataset; synthetic/license/secret tests |
| `dashboard/backend/main.py` | 실제 FastAPI status/worker/provider/training endpoints, local/token access guard, dashboard control token | quota/remaining credits와 remote session state는 계정 인증 없이는 표시하지 않음. Provider endpoint는 현재 프로세스만 검사 | `dashboard/backend/control.py`, frontend; `tests/test_dashboard.py` |
| `dashboard/backend/control.py` | project-root 경로 검증, train subprocess 시작, 명시적 허용 인자, pause/resume/stop/checkpoint control files, `metrics.jsonl` 상태 | UI/API는 local trainer 제어용이다. 사용자가 로그인한 외부 서비스의 GPU를 시작·중지하지 않으며, 브라우저에서 실제 GPU UI 흐름을 시험하지 않음 | `training/trainer.py`, FastAPI routes; dashboard integration/unit tests |

## 기존 아키텍처에서 유지한 점과 보완한 점

원래 Tiny Transformer/학습 경로, 데이터 파이프라인, Agent, dashboard, model export를 보존했습니다. Provider adapter 다섯 개와 local adapter를 등록하고, 이전 Kaggle module 안에 있던 중복 Colab stub를 실제 Colab adapter 호환 alias로 정리했습니다. Worker registry는 임의 payload/credential 필드를 상태에 저장하지 않고 서버가 identity를 발급합니다. `training/benchmark.py`는 현재 프로세스에서만 측정하며, 원격 provider 작업을 만들지 않습니다.

Checkpoint는 monolithic PyTorch state dictionary와 SHA-256/manifest를 사용합니다. 학습 재개에 필요한 optimizer/RNG 상태를 포함하므로 model-only safetensors export와는 용도가 다릅니다. `providers/storage.py`는 이미 mount된 local/Drive/Kaggle Output/Modal Volume 경로를 원자 복사하고, Hugging Face Hub upload/download는 명시적 confirmation flag가 없으면 차단합니다. Hub 업로드는 로컬 원본 hash와 commit revision을 남기지만 이 과정에서 서버에 저장된 전체 바이트를 다시 받아 검증하지는 않습니다.

## 테스트 재현

프로젝트 루트에서 `python -m unittest discover -s tests -v`, `python -m tests.integration_test`, `python -m compileall -q .`를 실행합니다. cloud transfer 테스트는 임시 디렉터리/mock Hub client를 사용하며 실제 provider 계정 사이에서 전송한 것은 아닙니다. 실제 하드웨어 성능 테스트는 각 사용자의 Colab/Kaggle/Lightning/Modal runtime 내부에서 `python -m training.benchmark --config config/model_tiny_10m.yaml`을 실행해야 합니다.
