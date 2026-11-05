# MiniCoder-7B — Phase 7~10 상태 보고

## 실제 검증 결과

| 항목 | 상태 | 결과 |
|---|---|---|
| Phase 7~10 전용 테스트 | 검증됨 | 5개 PASS |
| 기존 Phase 2~6 회귀 테스트 | 검증됨 | 23개 PASS |
| 전체 테스트 | 검증됨 | `python -m unittest tests.test_phase7_10 tests.test_phase2_4 -v`, 28개 OK |
| Python compileall | 검증됨 | PASS |
| Repository RAG 색인·검색 | 검증됨 | 함수/클래스 AST chunk, 경로/줄 번호, incremental reindex 동작 |
| Persistent Memory | 검증됨 | SQLite scope, 검색, update, 삭제, secret redaction |
| Tool Calling | 검증됨 | 파일 읽기/쓰기, 검색, 디렉터리, 명령, Python, syntax, Git; 위험 작업 승인 요구 |
| Code Execution | 검증됨 | timeout, stdout/stderr/returncode 수집 |
| Automatic Testing | 검증됨 | compileall와 unittest 실행, bounded retry, 실행 여부 기록 |
| Self-review | 검증됨 | syntax 및 hard-coded secret/subprocess timeout 정적 점검 |
| Planning/Context | 검증됨 | repository evidence와 메모리 결정을 포함한 bounded plan 생성 |
| Provider registry | 검증됨 | Ollama/OpenAI-compatible/Anthropic/Gemini/NVIDIA NIM/MiniCoder 등록 |
| 외부 provider 실제 API 호출 | 미검증 | API key와 외부 endpoint가 없어 live call은 실행하지 않음 |
| MiniCoder 실제 생성 품질 | 미검증 | 현재 모델은 랜덤 초기화/소형 검증 모델이며 pretrained intelligence로 표시하지 않음 |
| Windows 직접 실행 | 미검증 | Linux sandbox에서만 실행 |

## 추가된 파일

- `agent/providers.py`: 공통 `BaseProvider`, `ChatResponse`, HTTP adapter와 provider factory
- `agent/rag.py`: Python AST 기반 함수/클래스 chunk 및 변경 파일 incremental index
- `agent/memory.py`: global/project/session/task SQLite memory와 민감정보 redaction
- `agent/tools.py`: 명시적 tool schema, repository-root sandbox, 승인 필요한 write/dangerous command
- `agent/quality.py`: timeout 실행기, test runner, self-review
- `agent/planner.py`: RAG evidence 기반 planning과 context budget
- `agent/cli.py`: `index`, `search`, `plan`, `test`, `review` CLI
- `config/agent.yaml`: provider/모델/timeout/memory/index 설정 예시
- `tests/test_phase7_10.py`: Phase 7~10 재현 테스트

## 실행 예시

```bash
# provider는 config/agent.yaml의 기본값을 참고해 애플리케이션에서 생성
python -m agent.cli --root . index
python -m agent.cli --root . search "checkpoint metadata" --limit 5
python -m agent.cli --root . plan "Add a safe checkpoint export command"
python -m agent.cli --root . test
python -m agent.cli --root . review
python -m unittest tests.test_phase7_10 tests.test_phase2_4 -v
```

PowerShell에서도 동일한 모듈 명령을 사용할 수 있지만, 이 환경에서는 Windows 직접 실행을 하지 않았으므로 Windows 미검증이다.

## 의도적인 제한

1. 외부 API provider는 표준 HTTP 요청과 환경변수 API key를 사용한다. 실제 endpoint 호출 성공은 네트워크·인증정보가 있어야 검증할 수 있다.
2. `MiniCoderProvider`는 `model_runner` callable을 명시적으로 받아야 하며, 랜덤 초기화 MiniCoder의 능력을 외부 모델 능력처럼 표시하지 않는다.
3. RAG의 의미 검색은 의존성 없는 lexical/token-overlap baseline이다. 교체 가능한 embedding/vector backend는 다음 단계에서 추가할 수 있다.
4. 자동 수정은 무한 루프를 피하기 위해 이번 범위에서는 bounded test 결과와 오류 정보를 반환한다. 파일 변경은 tool approval이 필요하다.
5. 대시보드와 실제 학습 프로세스 제어는 Phase 11 범위로 남겨 두었다.


> **후속 업데이트:** 이 문서는 후속 Provider·checkpoint·dashboard 보안 수정 전의 snapshot입니다. 현재 통합 상태와 테스트 수는 `FINAL_IMPLEMENTATION_REPORT.md`, `TRAINING_STATUS.md`, `docs/PROJECT_AUDIT.md`를 기준으로 보세요.
