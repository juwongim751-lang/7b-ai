# MiniCoder-7B — Phase 2~4 상태 보고

작성 환경: Linux 샌드박스, Python 3.12, numpy만 설치. **torch / tokenizers / GPU / Windows 없음.**

## 실제로 실행해서 통과한 것 (샌드박스)
| 항목 | 결과 |
|---|---|
| `python -m compileall .` (전체) | PASS |
| `python -m unittest tests.test_phase2_4` | 14개 실행 PASS, 9개 SKIP(torch 필요) |
| 설정 로드/검증 (7B=7,241,732,096 / tiny / small) | PASS |
| 토크나이저 (바이트 fallback) 왕복·특수토큰·저장/로드 | PASS (실제 BPE는 미검증) |
| 데이터 파이프라인: 정확/유사 중복 제거, 문법 오류 제거, 라이선스 기록, 결정적 분할, 누수 방지, 재현 가능한 dataset_version, 토큰 shard 생성 | PASS (합성 데이터 기준) |
| 리밸런싱, 품질점수(길이만으로 고득점 안 됨), rank 분리 에포크 순서, 스케줄러 곡선, EarlyStopper, 메트릭 로그 | PASS |

## 작성만 했고 실행 못 한 것 (UNVERIFIED — torch 필요)
- `model/transformer.py` 전체 (GQA expand, KV cache, generate, 마스크, residual-scaled init, weight tying)
- `training/trainer.py` (AMP, grad accum, 클리핑, 검증/ppl, 조기종료, STOP/PAUSE 파일, OOM 안내, resume)
- `training/dist_utils.py` (DDP, 단일 GPU 자동 전환)
- `training/smoke.py` 종단 스모크 테스트, `TestModel`/`TestTraining` 9개 테스트
- 실제 BPE 학습 (`tokenizers` 패키지 필요), GPU 메모리/속도 수치

## Phase 별 변경 요약
- Phase 2: `model/config.py`(프리셋 tiny/small/7b, tie, 검증), `model/transformer.py`(재작성, 클래스명 호환 유지), `dataset/tokenizer.py`(ByteLevel initial_alphabet, 코드/FIM/언어 특수토큰, BOS/EOS 옵션, fingerprint, 통계)
- Phase 3: `dataset/pipeline.py`, `dataset/packed.py`, `dataset/synthetic.py`, `config/data.yaml`
- Phase 4: `training/trainer.py`(재작성), `training/config.py`, `training/metrics.py`, `training/dist_utils.py`, `training/smoke.py`
- Phase 1 버그: `controller/registry.py` import 수정(실행 확인)

## 아직 안 된 것 / 알려진 한계
- 체크포인트: 기존 `training/checkpoint.py` 그대로(해시 검증·메타데이터·`weights_only` 문제 남음) → Phase 5
- 기존 `dataset/manager.py`, `dataset/generator.py`, `training/distributed.py`(FSDP), `tests/integration_test.py`는 손대지 않음(레거시). 통합 테스트는 여전히 가짜
- 코드 특화 전처리: 특수토큰만 있음. 들여쓰기 인식 pre-tokenizer 없음
- 패킹 데이터는 문서 경계에서 attention을 끊지 않음
- 품질 점수는 휴리스틱이며 모델 성능과의 상관은 검증되지 않음
- 자동 설정 조정(Resource/Training Optimizer), 대시보드, 에이전트 전부 미구현
- 7B 학습은 불가능한 상태(AdamW fp32 기준 약 116GB + 활성화). 소형 모델 검증 성공을 7B 성공으로 간주하지 말 것

## Windows PowerShell 실행 명령
```powershell
pip install -r requirements-training.txt
python -m compileall -q .
python -m unittest tests.test_phase2_4 -v          # torch가 있으면 9개 추가 테스트가 실제 실행됨
python -m training.trainer --smoke-test            # CPU 종단 스모크(합성 데이터)

# 실제 데이터 흐름
python -m dataset.pipeline build --config config/data.yaml
python -m dataset.tokenizer --input data/processed/train.jsonl --vocab-size 32000 --output dataset/tokenizer.json
python -m dataset.pipeline tokenize --data-dir data/processed --tokenizer dataset/tokenizer.json
python -m training.trainer --config config/model_small.yaml --train-config config/train_small.yaml
python -m training.trainer --config config/model_small.yaml --train-config config/train_small.yaml --resume
# 안전 중단: runs\default\STOP 파일 생성 / 일시정지: runs\default\PAUSE 파일 생성
```


> **후속 업데이트:** 이 문서는 초기 phase snapshot입니다. 이후 BitNet·Provider·storage·dashboard·dataset 변경과 2026-11-05 최종 테스트 결과는 프로젝트 루트의 `FINAL_IMPLEMENTATION_REPORT.md`, `TRAINING_STATUS.md`, `docs/PROJECT_AUDIT.md`를 기준으로 보세요.
