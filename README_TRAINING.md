# MiniCoder 학습·복구 안내

MiniCoder는 무작위 초기화 decoder Transformer를 학습하는 코드입니다. 7B YAML은 구조 설정이며 무료 GPU에서 7B 모델을 학습할 수 있다는 뜻이 아닙니다.

## 설치와 CPU smoke

```bash
python -m pip install -r requirements-training.txt
python -m training.trainer --smoke-test
python -m unittest discover -s tests -v
python -m tests.integration_test
```

CPU smoke는 합성 corpus, tokenizer, Tiny forward/backward, evaluation, checkpoint와 resume의 기계적 경로를 확인합니다. 품질·실제 코딩 능력·대규모 학습 또는 클라우드 접근을 검증하지 않습니다.

## 실제 데이터·출처

기본 입력은 사용자가 검토한 local file이며 `config/data.yaml`에서 path, source, 실제 license를 적습니다. 미상/미승인 license와 민감정보 패턴은 기본 거부합니다. pattern scan은 보안/PII를 완전히 검출하지 못하므로 입력을 검토하세요.

```bash
python -m dataset.pipeline build --config config/data.yaml
python -m dataset.tokenizer --input data/processed/train.jsonl --vocab-size 32000 --output dataset/tokenizer.json
python -m dataset.pipeline tokenize --data-dir data/processed --tokenizer dataset/tokenizer.json
```

Public Hugging Face streaming은 opt-in입니다. 해당 dataset의 공식 card에서 실제 license와 field를 확인한 후 아래 dependency를 설치하고 notebook 또는 `dataset.public.load_public_hf_dataset()`에서 source/URL/license를 명시하세요.

```bash
python -m pip install -r requirements-public.txt
```

Loader는 upstream license를 자동 확인하지 않고 caller attestation을 manifest로 남깁니다. `trust_remote_code=False`, streaming, row/sample cap을 사용합니다. 사용자에게 권한이 없는 데이터셋은 사용하지 마세요.

## Tiny fit, resume, measured batch probe

```bash
python -m training.trainer \
  --config config/model_tiny.yaml \
  --data-dir data/processed \
  --run-dir runs/tiny \
  --max-steps 8

python -m training.trainer \
  --config config/model_tiny.yaml \
  --data-dir data/processed \
  --run-dir runs/tiny \
  --max-steps 10 --resume
```

`--max-steps`는 도착 global step입니다. 8에서 10으로 재개하면 2 optimizer update가 더 실행됩니다. checkpoint v3는 model·optimizer·scheduler·scaler·RNG·seed·model/train/tokenizer config·BitNet state·token count·progress를 포함하고 SHA-256 및 manifest를 검증합니다. `training.recovery.transfer_checkpoint()`는 무결성 확인 및 원자 복사를 지원합니다.

GPU에서 forward/backward 후보 batch를 실제 측정하려면 opt-in을 사용합니다.

```bash
python -m training.trainer \
  --config config/model_tiny_10m.yaml --data-dir data/processed \
  --run-dir runs/tiny10m --max-steps 10 \
  --auto-batch-size --max-auto-micro-batch-size 4
```

Probe는 설정한 최대 크기 이내 후보를 실제로 forward/backward하고 성공한 후보의 tokens/sec가 가장 높은 microbatch를 고릅니다. effective batch는 gradient accumulation을 조정해 대략 유지하고 RNG state 및 모델 weight를 보존합니다. report는 `runs/.../applied_config.json` 및 metric에 기록됩니다. 모든 후보가 실패하면 기존 batch 설정을 유지하고 메모리 부족 가능성이 남습니다. 짧은 사전 측정은 장시간 처리량을 보증하지 않으며 다른 GPU로 자동 failover하지 않습니다.

직접 계획만 보고 학습은 시작하지 않으려면:

```bash
python -m training.plan --target auto
python -m training.plan --target 7b
```

이는 현재 runtime의 실제 관찰 가능 메모리 및 manifest token count 기반 휴리스틱입니다. 소형 fallback을 추천하는 경우 사용자가 확인해야 하고, 모델 구조가 자동으로 바뀌지 않습니다. 다른 host/GPU의 메모리를 합산하지 않습니다.

## 비용 추정 안전장치

사용자가 현재 확인한 시간당 GPU 요금과 최대 USD 예산을 모두 지정하고, CLI에 explicit consent를 줘야 비용 가드가 있는 local trainer를 시작합니다.

```bash
python -m training.trainer --config config/model_tiny.yaml \
  --data-dir data/processed --run-dir runs/paid-review --max-steps 100 \
  --gpu-cost-rate-usd-per-hour 0.50 --max-budget-usd 1.00 \
  --confirm-potential-cost
```

Rate는 예시일 뿐이며 실제 provider 가격이 아닙니다. 금액 guard는 사용자 승인 후 training elapsed × user-supplied hourly rate를 계산하고 예산/예약 시점 전에 안전 지점에서 정지·저장합니다. Container startup, disk/network, tax, provider billing rounding, 가격 변경을 포함하지 않으며 공급자가 청구하는 hard cap이 아닙니다. Modal App에도 현재 rate와 budget 제공이 필요합니다.

## Kaggle 공식 CLI job 제출

계정 API token을 Kaggle CLI에 먼저 안전히 설정하세요. `kernel-metadata.json`의 owner/kernel ID, code file, dataset inputs, GPU/internet 선택 등을 검토합니다.

```bash
python -m providers.kaggle_jobs prepare \
  --kernel-dir kaggle_submission --metadata-json kaggle_submission_config.json
# 계정 쿼터·GPU를 직접 검토하고, 새 remote run을 의도할 때만 실행
python -m providers.kaggle_jobs submit --kernel-dir kaggle_submission --confirm-submit
python -m providers.kaggle_jobs status --kernel-id OWNER/KERNEL
python -m providers.kaggle_jobs output --kernel-id OWNER/KERNEL --destination runs/kaggle_output
```

프로젝트는 공식 Kaggle CLI command를 호출하며 private REST endpoint와 브라우저 자동화를 사용하지 않습니다. 제출은 remote action이고 주간 GPU quota를 사용할 수 있으므로 confirmation이 필요합니다. credentials는 코드에 입력하거나 프로젝트가 출력하지 않습니다. Kaggle status/output 및 submit은 실제 사용자 계정에서 미검증입니다. 대시보드에도 인증 endpoint와 쿼터 확인 체크박스가 있습니다.

## Colab/Kaggle notebooks

- [`notebooks/minicoder_colab.ipynb`](notebooks/minicoder_colab.ipynb): 사용자가 Drive 및 GPU runtime을 직접 선택·mount, license-attested dataset, 10M Tiny short run, Drive checkpoint/verified resume/export.
- [`notebooks/minicoder_kaggle.ipynb`](notebooks/minicoder_kaggle.ipynb): 사용자가 project ZIP dataset과 GPU를 직접 선택, `/kaggle/working` 결과 저장, 이전 Notebook Output checkpoint를 연결해 검증 후 재개/export.

노트북은 browser/API credentials를 자동 설정하지 않습니다. Colab은 공식 runtime REST/MCP를 통한 external submission client가 이 코드에 아직 연결되지 않았습니다. Kaggle external submit은 앞 절의 명시적 CLI client를 사용하세요. 상세 단계는 [`CLOUD_SETUP.md`](CLOUD_SETUP.md) 참조.

## BitNet, export, benchmark

```bash
python -m unittest tests.test_bitnet -v
python -m training.benchmark --config config/model_tiny_10m.yaml \
  --output-dir reports --steps 5 --batch-size 1 --seq-len 16
```

BitLinear는 QAT reference(ternary weight fake quantization, token-symmetric activation, STE)이며 master weights는 floating point입니다. 별도의 base-3 packed format은 inference-only이며 integer GEMM accelerator가 아닙니다. 상세 검증: [`BITNET_VALIDATION.md`](BITNET_VALIDATION.md), [`docs/BITNET_REFERENCE.md`](docs/BITNET_REFERENCE.md). Benchmark는 현재 process만 측정하며 report는 [`GPU_BENCHMARK_REPORT.md`](GPU_BENCHMARK_REPORT.md).

## Dashboard와 운영

```bash
python -m pip install -r requirements-agent.txt
python -m dashboard.backend.main
```

기본은 `127.0.0.1:8000`; 원격 접근은 HTTPS reverse proxy와 강한 `MINICODER_DASHBOARD_TOKEN`을 반드시 사용하세요. Dashboard는 실제 local Start/Pause/Resume/Stop/Checkpoint/Restore/Export와 Kaggle CLI endpoints를 제공합니다. Cross-provider provider-switch UI를 흉내내지 않으며, Colab/Lightning 외부 작업은 연결되어 있지 않습니다. Metrics에는 step/token/LR/loss/ETA 및 관측 가능한 자원량이 담깁니다. 이 환경에서 cloud/GPU/browser visual test는 수행하지 않았습니다.

다섯 cloud provider의 quota/가격/인증 공식 조사, provider별 구현 상태 및 테스트 결과는 각각 [`docs/CLOUD_V2_PROVIDER_API_RESEARCH.md`](docs/CLOUD_V2_PROVIDER_API_RESEARCH.md), [`PROVIDER_STATUS.md`](PROVIDER_STATUS.md), [`TEST_REPORT.md`](TEST_REPORT.md)에서 확인하세요. 계정별 무료 GPU, 잔액, 청구, 전화인증, runtime eligibility는 이 프로젝트가 추측하거나 조회하지 않습니다.
