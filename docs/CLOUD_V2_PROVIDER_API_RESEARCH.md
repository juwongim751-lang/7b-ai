# Cloud V2 제공자 API 조사 요약

**기준일: 2026-11-05.** 아래 내용은 제공된 공식 출처 조사 결과만을 종합한 것이다. 계정 로그인, 실제 제출·실행, 요금제·잔액·자격·쿼터·GPU 가용성은 시험하거나 추정하지 않았다. 특히 Kaggle과 Colab을 구현 우선순위로 둔다. 이 우선순위는 기능 범위와 공식적으로 문서화된 무인 호출 경로에 따른 것으로, 계정별 접근 또는 컴퓨팅 자원을 보장하지 않는다.

## 비교표

| 제공자 | 실행 모델: 사람이 시작하는 노트북 vs 컨트롤러 제출 API | 체크포인트 지속성 | 인증·결제 승인 | 구현 가능한 경계 |
|---|---|---|---|---|
| **Kaggle Notebooks — 우선순위 1** | **컨트롤러 제출 가능:** 공식 Kaggle CLI의 `kaggle kernels push`로 비대화형 노트북 커널 실행을 제출하고, `kernels status`/`kernels output`으로 추적·수집 가능. 직접 REST 노트북 실행 API는 확인된 공식 문서에 없음. 매 작업 브라우저 조작은 불필요하나 계정 및 토큰 사전 준비 필요. | 세션 내 scratchpad는 종료 시 소실. 체크포인트·산출물은 `/kaggle/working`에 쓰고 완료된 노트북 출력으로 저장(공식 사양 최대 20 GB). 이후 노트북 출력 입력 연결 또는 CLI로 Dataset 생성/버전 업로드. 새 버전 실행은 깨끗한 세션이므로 메모리 상태는 이어지지 않음. | 계정 설정에서 발급한 API 토큰을 자동화에 주입하거나 OAuth를 사용(사용자 로그인·동의 필요). Kaggle은 GPU 무료 제공을 안내하지만 공식 안내의 주간 GPU 쿼터 30시간 또는 자원·수요에 따른 그 이상은 계정별 보장이 아님. 가속기 자격·가용성 및 실제 쿼터를 추정하지 않음. 작업별 과금 단가 문서화 없음. | 우선 구현: 사전 발급 토큰 사용, 프로젝트/커널 메타데이터 준비, CLI push → 상태 조회 → 결과 다운로드. 세션은 CPU/GPU 최대 12시간, TPU 최대 9시간. **직접 REST 실행 API가 있다고 가정하지 말 것.** 실제 토큰·쿼터·가속기 확인은 런타임 온보딩 시 별도 필요. |
| **Google Colab 일반판 — 우선순위 2** | **컨트롤러가 런타임·코드를 실행 가능:** REST `POST /v1beta/runtimes`로 런타임 할당, 공식 MCP의 `create_runtime`/`create_session`/`execute_code`로 Jupyter 세션에서 Python 실행. 이는 `.ipynb` 파일을 배치 큐에 제출하는 API가 아님. 호출마다 브라우저 버튼을 누를 필요는 없지만 최초 OAuth·프로젝트 설정은 사람 조치가 필요할 수 있음. | 런타임 로컬 파일은 VM이 살아 있는 동안만 유효하며 VM 삭제 시 소실. Drive 마운트, Drive REST API 또는 Google Cloud Storage로 체크포인트를 주기적으로 외부 저장해야 함. Drive 마운트는 인증 코드를 요구할 수 있어 헤드리스 자격 증명의 자동 승계를 전제하지 말고 별도 설정·검증 필요. | REST API는 `https://www.googleapis.com/auth/colaboratory` OAuth scope 필요. Google Cloud MCP는 프로젝트, API 활성화 및 인증 설정이 필요하며 공식 활성화 절차에 Cloud billing 활성화가 전제됨. Colab 컴퓨팅은 계정 구독·compute unit·동적 정책에 좌우됨. 실제 구독, 가격, GPU 자격 또는 접근성은 시험·추정하지 않음. | 우선 구현: OAuth 준비 → `list_runtime_specs`로 요청 주체의 `eligible` 확인 → 런타임 생성 → 프로젝트 파일·의존성 준비 → Python 학습 엔트리포인트 실행 → 외부 저장소 체크포인트. `execute_code` 스트리밍은 600초 제한이며 `executionId`로 같은 세션의 스트림에 재접속 가능하나 장시간 런타임 보장은 아님. 일반 런타임 최대 12시간(동적·더 짧을 수 있음), Pro+는 충분한 compute unit이 있을 때 최대 24시간. **노트북 파일 배치 제출 API로 표현하지 말 것.** |
| **Lightning AI Studio** | **컨트롤러 제출 가능:** 공식 SDK `Studio.start`, `Studio.run`, `Job.run(...)` 및 CLI로 Studio 명령/배치 Job 실행. 공식 노트북 셀 제출 API는 확인되지 않음. 문서는 Jupyter 노트북을 실행하려면 브라우저 탭을 열어 두어야 한다고 명시. Studio 안에서 실행한 스크립트는 브라우저를 닫아도 실행 지속 가능. | Studio 파일·환경은 sleep 후 지속. Job 제출 시 환경 복제, 산출물은 Teamspace Drive `/teamspace/jobs/<job name>` 또는 `lit://.../jobs/...`에서 확인·다운로드. 체크포인트는 Drive 경로 또는 연결한 S3/GCS에 명시 저장. Job 삭제 시 아티팩트도 삭제. | 공식 외부 인증은 `lightning login`; 시작 안내는 로컬 로그인 시 브라우저 인증을 설명. 사전 인증 상태면 후속 SDK/CLI 호출은 무브라우저 가능하지만, 문서화된 Studio/Jobs headless service-account/API-key 최초 설정은 확인되지 않음. Free 크레딧 및 pay-as-you-go가 있으며 비용 승인·계정 정책 확인 필요. 실제 잔액·권한 미확인. | 인증된 컨트롤러로 원격 Studio에서 `python train.py` 실행 또는 Batch Job 제출·모니터링. Notebook 자체의 무인 제출·실행으로 간주하지 않음. 가격 페이지의 Free persistent storage 50 GB와 Teamspace Drive 문서의 Free 총 200 GB가 일치하지 않으므로 한도를 하드코딩하지 말고 계정에서 확인. 실제 크레딧/GPU 할당량·가용성은 미확인. |
| **Modal** | **컨트롤러가 Function 작업 제출 가능:** 배포된 Function을 공식 SDK로 조회·호출, 예: `Function.from_name(...).spawn(args)` 후 상태·결과 확인. 이는 노트북 파일/셀 제출 API가 아님. Modal Notebook 편집기는 2026-10-23부터 종료되어 생성·편집·실행 불가. | 이름 있는 `modal.Volume`을 Function에 마운트하고 안전 지점에서 저장 후 `volume.commit()`. 수 초 간격 백그라운드 커밋 및 컨테이너 종료 시 최종 스냅샷 문서화. 새 Volume은 v2 권장. 동시 동일 파일 쓰기는 last-write-wins이므로 실행별 경로 분리. | `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` 또는 활성 프로필 API 토큰. Service User는 Team/Enterprise 전용이며 기본 Environment 권한 없음(역할 부여 필요). GPU 사용에는 유효한 결제수단 요구. 가격표의 크레딧·동시성은 계정 잔액·권한을 뜻하지 않으며, 실제 계정 조건은 미확인. | 사전 배포된 App의 Function을 외부 컨트롤러가 호출하는 서버리스 작업으로 구현. GPU는 Function 설정에서 지정하고 Volume에 체크포인트. 비동기 입력 대기열 최대 1,000,000, 결과 7일 보관; 실행시간 기본 5분·최대 24시간. 동기 호출은 큐·연결 종료 제한이 있으므로 비동기와 멱등성 고려. **노트북 API가 아니라 Function API.** |
| **Hugging Face Spaces ZeroGPU** | **제한적 컨트롤러 호출 가능:** 외부 컨트롤러가 Gradio Space에 노출된 API를 호출할 수 있지만, 사전 작성된 앱 함수만 실행. ZeroGPU에는 노트북 시작 또는 임의 학습 명령 제출 API가 문서화되지 않음. Spaces 노트북 문서는 Colab/Kaggle에서 노트북 열기를 설명할 뿐 원격 제출 API가 아님. Hugging Face Jobs는 별도 제품·컴퓨트로서 ZeroGPU와 구분. | Space 로컬 디스크는 restart/stop 사이 ephemeral. Hugging Face Storage Bucket을 Space volume으로 붙이고 프로젝트 `checkpoint_dir`/run 경로를 해당 마운트로 설정해야 지속 가능. 버킷 연결·프로젝트 통합은 미구현·미검증 상태. | ZeroGPU 문서상 일일 할당량은 비인증 2분, 무료 계정 5분, PRO/Team 40분, Enterprise 60분이며 최초 GPU 사용 후 24시간 뒤 초기화; xlarge는 2배 소모. PRO/Team/Enterprise의 할당량 이후 prepaid credits는 10분당 $1. 무료 개인 호스팅은 정상 계정·이메일 인증·계정 연령 30일 초과 등의 정책 조건. 이는 정책 설명이며 계정 자격·잔액·사용 가능성을 확인한 것이 아님. Hugging Face Jobs는 인증과 양수 크레딧이 필요하고 별도 하드웨어 사용량 과금. | ZeroGPU용 Gradio Space의 미리 정의된 GPU 함수만 호출 가능(기본 decorator 런타임 60초, 설정 가능). 일반 학습 job 제출로 구현하지 말 것. 임의 명령이 필요하면 ZeroGPU가 아닌 별도 Hugging Face Jobs 제품을 평가해야 함. 프로젝트의 현재 `huggingface_zerogpu` adapter는 로컬 상태만 검사하며 클라우드 제출·원격 Space 호출·저장소 검증을 하지 않음. |
| **Local GPU (현재 MiniCoder 호스트)** | **컨트롤러 제출 API 없음:** LocalProvider는 런타임/GPU 상태 검사와 단일 프로세스 `torchrun` 명령 생성만 함. Jupyter REST/커널 API는 이미 구동 중인 사용 관리 서버의 세션·커널 제어이며 독립 provider 작업 제출/GPU 할당이 아님. 같은 호스트의 OS 프로세스 실행이나 사전 설정 SSH를 통한 일반 호스트 오케스트레이션은 가능하나 이는 provider API가 아님. | 학습기가 `run-dir`에 체크포인트·메트릭을 쓰고 `checkpoint_dir`로 경로 지정 가능. 실제 보존은 호스트 파일시스템 또는 사용자가 마운트·승인한 공유 저장소의 수명에 달림. LocalProvider는 `persistent_storage_verified=false` 보고. 필요 시 영구 경로에 복사하고 다음 런타임에서 복원. | 로컬 provider 계정·구독·쿼터·GPU 단가는 문서화되지 않음. 같은 호스트 프로세스 권한과 Python/CUDA 환경이 필요. Jupyter는 기본 토큰 인증, 원격 접근 시 도달 가능한 서버·유효 토큰 필요; SSH는 원격 계정 인증·명령 권한 필요. 실제 접속·실행 및 운영비 추정 없음. | 동일 호스트에서 학습 스크립트를 직접 실행하거나 승인된 SSH/Jupyter를 별도로 오케스트레이션할 수 있음. 프로젝트 어댑터 자체는 원격 머신 생성, 실행·감시·재시도, 다중 호스트 통합을 하지 않음. CUDA 검사는 자원 확보·권한·VRAM 적합성·성능·장기 가용성을 보장하지 않음. |

## 우선순위 및 구현 경계

1. **Kaggle 우선:** 이 조사에서 커널 실행을 비대화형으로 제출·추적·수집하는 공식 경로가 가장 분명한 선택이다. 구현은 REST가 아닌 Kaggle CLI 흐름을 중심으로 하고 토큰을 비밀로 취급한다. `/kaggle/working` 출력과 Dataset/노트북 출력 연결을 통해 체크포인트 전달을 설계하되 세션 메모리 재사용을 기대하지 않는다.
2. **Colab 다음:** 공식 REST/MCP로 런타임과 Jupyter 세션의 Python 코드를 원격 실행할 수 있다. 그러나 이는 `.ipynb` 배치 job API가 아니며 Google Cloud 프로젝트·OAuth·billing 준비와 계정별 런타임 적격성 확인을 전제로 한다. 외부 저장소 인증을 포함한 헤드리스 체크포인트 흐름을 검증해야 한다.
3. **나머지 경계:** Lightning과 Modal은 각자의 Studio/Job 또는 Function 작업 모델을 통해 실행하고, notebook 제출 API로 추상화하지 않는다. ZeroGPU는 Space에 구현된 함수 호출에 한정한다. Local GPU adapter는 검사기이지 원격 실행기가 아니다.

## 계정별 미확인 사항 및 주의

모든 계정별 접근·권한·실행·결제 조건은 미확인이다. Kaggle 주간 쿼터·가속기 자격/가용성, Colab 구독·eligible 사양·동적 한도, Lightning 크레딧/머신·스토리지 한도, Modal 토큰·Environment 역할·결제수단 및 GPU 가용성, Hugging Face 계정 연령/인증·할당량·크레딧·저장소 연결은 실제 대상 계정에서 따로 확인해야 한다. 이 문서는 제공자 정책의 명시된 값과 경계를 정리한 것이며 사용자 실제 쿼터를 추론하지 않는다.

**어댑터/노트북에 관한 명시적 경고:** 프로젝트의 Local GPU 어댑터는 로컬 검사 도구이며 원격 실행 API가 아니다. 프로젝트의 Hugging Face ZeroGPU 어댑터도 로컬 상태 검사 도구이며 Space 호출 API가 아니다. Colab MCP는 런타임 내 Python 코드 실행이지 기존 노트북의 배치 제출 API가 아니다. Lightning Jupyter Notebook은 브라우저 탭을 요구하며 제출 API가 아니다. Modal의 과거 Notebook 편집기는 종료되었고 Modal Function 호출은 노트북 제출 API가 아니다. Kaggle의 구현 근거는 직접 REST가 아니라 CLI 커널 제출이다.

## 제공자별 공식 출처

### Kaggle Notebooks

- [Kaggle API 문서](https://www.kaggle.com/docs/api)
- [Kaggle CLI kernels 문서](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)
- [Kaggle kernel metadata 문서](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)
- [Kaggle Notebooks 문서](https://www.kaggle.com/docs/notebooks)
- [Kaggle CLI datasets 문서](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md)
- [Kaggle CLI README](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md)
- [Kaggle Efficient GPU Usage](https://www.kaggle.com/docs/efficient-gpu-usage)

### Google Colab (일반 Colab)

- [Colab API MCP 참조](https://developers.google.com/colab/api/reference/mcp)
- [런타임 생성 REST 참조](https://developers.google.com/colab/api/reference/rest/v1beta/runtimes/create)
- [execute_code 도구](https://developers.google.com/colab/api/reference/mcp/tools_list/execute_code)
- [list_runtime_specs 도구](https://developers.google.com/colab/api/reference/mcp/tools_list/list_runtime_specs)
- [get_subscription 도구](https://developers.google.com/colab/api/reference/mcp/tools_list/get_subscription)
- [create_runtime 도구](https://developers.google.com/colab/api/reference/mcp/tools_list/create_runtime)
- [Google Cloud MCP 인증](https://docs.cloud.google.com/mcp/authenticate-mcp)
- [Google Cloud MCP 활성화/비활성화](https://docs.cloud.google.com/mcp/enable-disable-mcp-servers)
- [Colab I/O 노트북 예제](https://colab.research.google.com/notebooks/io.ipynb)
- [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Google Developers Blog: Colab MCP 서버 발표](https://developers.googleblog.com/announcing-the-colab-mcp-server-connect-any-ai-agent-to-google-colab/)
- [Google Colab colab-mcp 저장소](https://github.com/googlecolab/colab-mcp)

### Lightning AI Studio

- [Studio SDK](https://lightning.ai/docs/platform/developers/sdk/studio)
- [Batch Job 제출](https://lightning.ai/docs/platform/inference/batch-jobs/submit-jobs)
- [Batch CLI](https://lightning.ai/docs/platform/developers/cli/batch)
- [Studio 백그라운드 실행](https://lightning.ai/docs/platform/build/ai-studio/background-execution)
- [Teamspace Drive](https://lightning.ai/docs/platform/team-management/teamspaces/drive)
- [Dataset 가져오기](https://lightning.ai/docs/platform/data/datasets/import-dataset)
- [Lightning 가격](https://lightning.ai/pricing)
- [시작하기](https://lightning.ai/docs/platform/overview/getting-started)

### Modal

- [Modal Notebooks 종료 공지](https://modal.com/changelog/sunsetting-modal-notebooks)
- [Notebook 안내](https://modal.com/docs/guide/notebooks)
- [배포 Function 트리거](https://modal.com/docs/guide/trigger-deployed-functions)
- [Function 호출 방식](https://modal.com/docs/guide/function-invocation-methods)
- [Functions](https://modal.com/docs/guide/functions)
- [지속적 배포](https://modal.com/docs/guide/continuous-deployment)
- [Jupyter Notebooks](https://modal.com/docs/guide/jupyter-notebooks)
- [Volumes](https://modal.com/docs/guide/volumes)
- [Volume Python SDK](https://modal.com/docs/sdk/py/latest/Volume)
- [Service Users](https://modal.com/docs/guide/service-users)
- [CLI 토큰](https://modal.com/docs/cli/latest/token)
- [GPU 설정](https://modal.com/docs/guide/gpu)
- [가격](https://modal.com/pricing)
- [시간 제한](https://modal.com/docs/guide/timeouts)

### Hugging Face Spaces ZeroGPU

- [Spaces ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu)
- [Spaces API endpoints](https://huggingface.co/docs/hub/spaces-api-endpoints)
- [Hub Notebooks](https://huggingface.co/docs/hub/notebooks)
- [Hugging Face Hub Jobs 가이드](https://huggingface.co/docs/huggingface_hub/guides/jobs)
- [Jobs training](https://huggingface.co/docs/hub/jobs-training)
- [Spaces 저장소](https://huggingface.co/docs/hub/spaces-storage)
- [Storage Buckets 접근](https://huggingface.co/docs/hub/storage-buckets-access)
- [Jobs 빠른 시작](https://huggingface.co/docs/hub/jobs-quickstart)

### Local GPU

- [PyTorch CUDA 문서 2.15](https://docs.pytorch.org/docs/2.15/cuda.html)
- [Python subprocess](https://docs.python.org/3/library/subprocess.html)
- [OpenBSD ssh 매뉴얼](https://man.openbsd.org/ssh)
- [Jupyter Server REST API](https://jupyter-server.readthedocs.io/en/latest/developers/rest-api.html)
- [Jupyter Server 보안](https://jupyter-server.readthedocs.io/en/latest/operators/security.html)
- [Jupyter Client 메시징](https://jupyter-client.readthedocs.io/en/stable/messaging.html)
- [PyTorch torchrun](https://docs.pytorch.org/docs/2.14/elastic/run.html)
- [PyTorch 모델 저장/불러오기 튜토리얼](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [Python shutil](https://docs.python.org/3/library/shutil.html)
