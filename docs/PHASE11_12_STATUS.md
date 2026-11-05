# MiniCoder-7B — Phase 11~12 상태 보고

## 구현

### Phase 11 — 실제 학습 대시보드 연결

- `dashboard/backend/control.py`의 `TrainingProcessManager`가 실제 `training.trainer` subprocess를 시작한다.
- 학습 run directory의 `metrics.jsonl`을 읽어 최신 step/loss/lr/tokens-per-second와 최근 metric을 반환한다.
- `STOP`, `PAUSE`, `CHECKPOINT` control file을 실제로 생성한다.
- trainer의 `ControlFiles`가 `CHECKPOINT`를 감지하면 다음 optimizer update 직후 checkpoint를 저장한다.
- FastAPI endpoint:
  - `GET /status`
  - `GET /workers`
  - `POST /train/start`
  - `POST /train/stop`
  - `POST /train/pause`
  - `POST /train/resume`
  - `POST /train/checkpoint`
  - `WS /ws/monitoring`
- 상태 값은 가짜 변수로 변경하지 않고 process 상태·control file·metrics 파일에서 계산한다.

### Phase 12 — 통합 테스트와 문서

- 기존 `tests/integration_test.py`의 무조건 PASS 출력과 존재하지 않는 `MiniCoderTrainer` 의존성을 제거했다.
- 실제 tokenizer, tiny Transformer forward/backward, checkpoint 왕복, RAG, memory, tools, code execution, self-review, dashboard controls, worker scheduler를 실행한다.
- 실패는 traceback과 함께 `FAIL`로 보고하고, 모든 검증이 실제 assertion을 통과한 경우에만 성공한다.

## 실제 검증 결과

| 검증 | 결과 |
|---|---|
| `python -m compileall -q .` | PASS |
| `python -m tests.integration_test` | PASS, 실제 6개 시나리오 실행 |
| dashboard control files | PASS |
| dashboard routes import | PASS |
| `metrics.jsonl` 상태 조회 | PASS |
| `python -m unittest tests.test_phase7_10 tests.test_phase2_4 -v` | PASS, 28개 OK |
| Phase 2~4 CPU smoke + resume | PASS |
| CUDA GPU 대시보드/멀티 GPU | 실행 불가, sandbox GPU 0개 |
| Windows 직접 실행 | 미검증 |
| 외부 API provider live call | 미검증, 인증정보 없음 |

## 실행 명령

```bash
python -m tests.integration_test
python -m dashboard.backend.main
# 별도 터미널에서
curl http://127.0.0.1:8000/status
curl -X POST http://127.0.0.1:8000/train/pause
curl -X POST http://127.0.0.1:8000/train/resume
curl -X POST http://127.0.0.1:8000/train/checkpoint
curl -X POST http://127.0.0.1:8000/train/stop
```

Windows PowerShell에서도 모듈 실행과 REST 호출 형태는 동일하지만 이 환경에서 직접 실행하지 않아 Windows 미검증이다.

## 정직한 제한사항

실제 7B 품질 학습, 외부 provider 인증 호출, CUDA GPU utilization, 브라우저 UI는 이 범위에서 검증하지 않았다. 현재 대시보드는 실제 backend API와 WebSocket monitoring을 제공하며, frontend 화면은 Phase 11의 backend 연결 범위로 남아 있다. synthetic corpus smoke test 성공은 7B 모델 품질을 의미하지 않는다.


> **후속 업데이트 (2026-11-05):** 이 문서의 “frontend 화면은 ... 연결 범위로 남아 있다”는 문장은 현재 상태와 다릅니다. 이후 `dashboard/frontend`를 실제 API에 연결하고 상태표·인증 토큰 입력·worker XSS 방지를 추가했습니다. 현재 API 통합 테스트는 통과했지만 브라우저 UI와 실제 cloud 계정은 별도로 검증하지 않았습니다. 기준 문서는 `FINAL_IMPLEMENTATION_REPORT.md`와 `TRAINING_STATUS.md`입니다.
