# MiniCoder-7B — Phase 5~6 상태 보고

작성 환경: Linux sandbox, Python 3.12.3, torch 2.15.0+cu132 (CUDA runtime 포함, 실제 GPU 0개), tokenizers 0.23.3, PyYAML 6.0.3, psutil 7.2.2, safetensors 0.8.0.

## 검증 요약

| 항목 | 상태 | 실제 결과 |
|---|---|---|
| `python -m compileall -q .` | 검증됨 | PASS |
| `python -m unittest tests.test_phase2_4 -v` | 검증됨 | 23 tests, OK; 이전 9개 SKIP도 모두 실행됨 |
| `python -m training.trainer --smoke-test` | 검증됨 | `all_passed: true`; synthetic data, eval, loss 감소, resume 계속 진행 |
| Transformer GQA/KV cache/mask/tied parameter tests | 검증됨 | 기존 TestModel 7개 PASS |
| grad accumulation 동등성 | 검증됨 | PASS |
| 실제 BPE tokenizer, 비ASCII 왕복, UNK=0 | 검증됨 | Phase 2~4 전체 테스트 PASS |
| atomic checkpoint + SHA-256 + step ordering | 검증됨 | 왕복 및 손상 파일 감지 테스트 PASS |
| checkpoint metadata mismatch 거부 | 검증됨 | `ValueError` 및 `--force-resume` 경로 구현; 독립 테스트 PASS |
| best checkpoint metadata 보존 | 구현됨 | validation loss가 manifest에 저장되고 보존 대상에 포함 |
| model-only safetensors export | 검증됨 | tied `lm_head.weight` 표기/생략 포함 export PASS |
| ResourceManager | 부분 검증됨 | RAM 측정 실측; GPU 값은 GPU 없음으로 `None` |
| auto-tune 실제 probe | 검증됨 | tiny 모델 forward/backward probe와 `applied_config.json` 생성 |
| CPU 2-process gloo DDP | 검증됨 | rank별 입력과 gradient 동기화 확인 |
| 멀티 GPU CUDA DDP/FSDP | 실행 불가 | CUDA device 0개인 sandbox |
| 실제 라이선스 코드 데이터 학습 품질 | 미검증 | 제공된 데이터가 synthetic corpus뿐; 메커니즘 검증만 수행 |
| 중단/재개 loss 곡선의 중단 없는 baseline과 수치 비교 | 미구현/미검증 | smoke resume 연속성은 확인했으나 독립 baseline 오차 비교는 하지 않음 |
| Windows 직접 실행 | 미검증 | Windows 환경이 없어 PowerShell 명령만 문서로 제공 |

## 구현 내용

### Phase 5 — Checkpoint

`training/checkpoint.py`를 재작성했다.

- 임시 파일 쓰기 → flush/fsync → `os.replace` → 디렉터리 fsync의 원자적 저장
- checkpoint별 `.sha256` 및 `.manifest.json`
- 파일을 step 번호로 선택하며 mtime에 의존하지 않음
- 손상/누락 checksum checkpoint는 이전 정상 step으로 fallback하고 warning log 기록
- model, optimizer, scheduler, scaler, step/epoch/batch, torch/CUDA/NumPy/Python RNG 저장
- model/train config, tokenizer fingerprint, dataset version metadata 저장
- metadata 불일치는 명확히 거부하며 `force_resume=True` 또는 `--force-resume` 사용 시 warning 기록
- validation loss를 manifest에 저장하고 best checkpoint를 prune 시 보존
- `weights_only=False`는 optimizer/RNG를 복구해야 하므로 manager가 관리하는 checkpoint와 명시적 경로에서만 사용

### Model export

`model/export.py`는 FSDP가 import 시 필수가 아니며 일반 모델도 export할 수 있다. safetensors가 설치되면 `model.safetensors`, 미설치면 `pytorch_model.bin`으로 fallback한다. tied embedding은 `config.json`과 `weights_manifest.json`에 표시하고 중복 shared tensor를 저장하지 않는다.

### Phase 6 — 자원 및 자동 설정

- `training/resource_manager.py`: GPU memory/utilization, RAM, data wait/step time, OOM recommendation을 측정한다. 측정할 수 없는 값은 `None`이다.
- `training/auto_tune.py`: 후보 micro-batch/sequence length를 실제 forward/backward probe하고 결과·적용 설정을 JSON으로 저장한다. before/after를 측정하지 않으므로 성능 개선 주장을 하지 않는다.
- trainer는 checkpoint metadata와 `force_resume`를 연결하고, validation loss를 저장한다.

## 재현 명령

### Linux/macOS

```bash
cd mc
python -m pip install -r requirements-training.txt
python -m compileall -q .
python -m unittest tests.test_phase2_4 -v
python -m training.trainer --smoke-test
python -m training.trainer --config config/model_small.yaml --train-config config/train_small.yaml
python -m training.trainer --config config/model_small.yaml --train-config config/train_small.yaml --resume
python -m training.trainer --config config/model_small.yaml --train-config config/train_small.yaml --resume --force-resume
```

### Windows PowerShell

```powershell
Set-Location .\mc
python -m pip install -r .\requirements-training.txt
python -m compileall -q .
python -m unittest tests.test_phase2_4 -v
python -m training.trainer --smoke-test
python -m training.trainer --config .\config\model_small.yaml --train-config .\config\train_small.yaml
python -m training.trainer --config .\config\model_small.yaml --train-config .\config\train_small.yaml --resume
python -m training.trainer --config .\config\model_small.yaml --train-config .\config\train_small.yaml --resume --force-resume
# 안전 중단: New-Item .\runs\small\STOP -ItemType File
# 일시정지:  New-Item .\runs\small\PAUSE -ItemType File
```

위 PowerShell 명령은 문법과 경로를 검토했지만 Windows에서 직접 실행하지 않았으므로 **Windows 미검증**이다.

## 제한 및 다음 단계

7B 랜덤 초기화 학습은 이 sandbox의 GPU 부재와 자원 한계로 실행하지 않았다. synthetic smoke test 성공은 7B 학습 성공이나 모델 품질을 의미하지 않는다. 실제 라이선스가 확인된 코드 데이터, GPU 환경, 독립적인 baseline/resume 곡선 비교가 준비되면 Phase 6 품질 검증을 추가로 수행해야 한다.

Phase 1 문서(`PHASE1_ANALYSIS.md`)는 제공된 압축파일에 포함되어 있지 않아 읽거나 검증할 수 없었다.


> **후속 업데이트:** 이 문서는 2026-11-05 작업의 중간 snapshot입니다. 현재 resource metrics, storage adapter, resume path와 최종 검증 결과는 프로젝트 루트의 `FINAL_IMPLEMENTATION_REPORT.md` 및 `TRAINING_STATUS.md`, `docs/PROJECT_AUDIT.md`가 우선합니다.
