# Lightning AI Studio — Python 학습 프로젝트 제공자 검토

**기준일: 2026-11-05.** 공식 Lightning AI 문서와 가격표만 검토했다. 실제 계정, 로그인, GPU 할당 가능 여부 또는 과금은 시험하지 않았다.

## 결론

외부 컨트롤러는 Lightning의 공식 Python SDK 또는 CLI로 Studio를 만들거나 시작하고 GPU 머신을 지정해 Python 명령을 실행할 수 있다. 별도 비대화형 Batch Job 제출도 지원된다. 따라서 학습 코드를 `python train.py ...` 형태로 실행하는 통합은 공식 경로가 있다. 다만 이는 **Jupyter 노트북 셀 제출 API가 있다는 뜻이 아니다**. 문서는 노트북을 브라우저에서 실행하는 것으로 설명하고, 노트북은 탭을 열어 두어야 한다고 명시한다. 공식적으로 문서화된 자동화 경계는 SDK/CLI로 Studio·명령·Batch Job을 제어하는 지점이다. [1] [2] [3] [4]

## 공식 제출 경로와 실행 방식

외부 환경에 `lightning-sdk`를 설치하고 로그인해 SDK의 `Studio(name=..., teamspace=..., user=...)` 객체로 원격 Studio를 지정할 수 있다. `studio.start(Machine.T4)`처럼 시작할 때 GPU 머신을 지정할 수 있고, `studio.run("python train.py ...")`로 명령을 원격 실행할 수 있다. 실행 후에는 `studio.stop()`으로 정지한다. CLI 역시 `lightning studio start ... --machine ...`과 `lightning job run --command="python train.py" --studio=...`를 제공하므로 컨트롤러가 브라우저 UI를 클릭할 필요는 없다. 단, 외부에서 실행하려면 해당 Teamspace와 계정에 접근 권한이 있어야 한다. [1] [2] [3]

노트북을 반드시 유지해야 한다면 공식 안내상 Jupyter Notebook은 브라우저 탭이 열린 상태여야 한다. 브라우저를 닫아도 계속 실행되는 것은 Studio 터미널에서 시작한 스크립트이며, 작업 완료 뒤 Studio가 자동 sleep될 수 있다. 따라서 노트북 파일을 무인 실행에 맡기는 공식 제출 API로 보지 말고, 학습 코드를 재사용 가능한 Python 스크립트/CLI 진입점으로 실행하는 편이 문서화된 경로에 맞는다. [4]

## 인증, 비용, 할당 제한

Studio SDK 문서는 Studio 내부에서는 이미 인증되어 있다고 설명한다. 외부 노트북·CI 환경은 `lightning login`으로 인증하고 SDK가 그 로그인을 이용한다. 공식 시작 안내는 로컬에서 `lightning login`을 실행해 브라우저로 로그인한다고 설명하므로, **최초 자격 증명 설정은 사람의 브라우저 로그인 개입이 필요할 수 있다**. 미리 인증된 CLI 환경이나 Studio 안에서는 이후 명령을 무인 호출할 수 있다. 검토한 Studio/Jobs 문서에는 별도의 공식 headless 서비스 계정/API 토큰 설정 절차가 확인되지 않았다. Sandbox용 API 키 문서는 별도 제품의 인증 안내이므로 Studio/Jobs 인증 방식으로 간주하지 않는다. [1] [3] [8]

가격 페이지는 Free 플랜, 무상 크레딧/일부 무료 GPU 시간, 초 단위 과금 및 사용량을 넘긴 뒤의 pay-as-you-go를 안내한다. 페이지에는 무료 CPU Studio의 4시간 재시작 조건, Free 플랜에서 동시 GPU 최대 2개 및 A100/H100/H200의 4시간 세션 제한도 기재되어 있다. 구체적 GPU 가격은 종류별로 다르며, 예시로 T4는 $0.55/시간, L4는 $0.79/시간으로 표시된다. 따라서 크레딧 잔액·플랜·GPU 선택에 따라 유료 사용이 될 수 있으므로, 실행 전에 사용자가 계정의 결제/지출 정책을 확인하고 GPU 작업을 승인하도록 해야 한다. 이 조사로 특정 계정에 크레딧, 결제수단, 권한 또는 GPU 여유가 있다고 추정하지 않는다. [7]

저장 한도는 공식 페이지 사이에 차이가 있다. 가격 페이지는 Free persistent storage를 50GB로 표시하지만, Teamspace Drive 문서는 Free 총 저장 한도를 200GB(첫 10GB 무료)로 표시한다. 범위·상품 설명 차이인지 문서 갱신 시점 차이인지는 공개 문서만으로 확정할 수 없으므로 한도를 코드에 고정하면 안 된다. 계정별 실제 사용량과 한도는 실행 전에 확인해야 한다. [5] [7]

## 체크포인트와 지속 저장

Studio의 파일·설치 환경은 sleep/재시작 후에도 유지된다. 다만 Studio 자체에 쌓이는 파일이 많으면 시작·종료에 더 오래 걸릴 수 있다. 비대화형 Studio 기반 Job에서는 Studio 환경이 제출 시점에 복제되고, Job 파일/아티팩트는 Teamspace Drive에서 접근 가능하다. SDK는 Job 아티팩트 목록 조회와 다운로드를 제공하며, Job 삭제 시 아티팩트도 삭제된다고 명시한다. 따라서 중요한 체크포인트는 Job 작업 디렉터리에만 남기지 말고 Teamspace 공유 Drive 또는 연결한 S3/GCS 등 내구성 있는 저장소로 기록·복사하고, 보존이 확인되기 전에는 Job을 삭제하지 않도록 해야 한다. [2] [5] [6]

Teamspace Drive는 Studio와 Job이 공유하며, 문서는 Job 파일의 `/teamspace/jobs/<job name>` 경로와 `lit://.../jobs/<job name>` 아티팩트 위치를 안내한다. 연결된 S3 버킷은 Studio/Job에서 `/teamspace/s3_connections` 아래로 접근할 수 있다. 체크포인트 경로를 학습 코드 설정으로 명시하고, 성공 후 SDK의 `artifacts_uri`/`list_artifacts()` 또는 공유 스토리지에서 파일을 확인하는 방식을 권장한다. [5] [6]

## 구현 경계

구현 가능한 범위는 **사용자가 인증과 비용 게이트를 준비한 후**, SDK/CLI로 지정된 Studio를 시작하거나 공식 Batch Job을 제출하고, `python train.py` 실행·상태/로그 확인·결과 다운로드를 수행하는 선택형 백엔드다. Notebook 셀을 Lightning이 제공하는 공식 API로 외부 제출하거나, 이 조사만으로 계정별 GPU 접근·할당·잔액을 보장하는 구현은 근거가 없다. 인증 초기화는 문서화된 `lightning login`을 따르며, 사전 인증 자격 증명이 없는 자동 컨트롤러가 로그인 UI까지 대체한다고 약속하지 않는다. [1] [2] [3] [8]

## References

[1]: https://lightning.ai/docs/platform/developers/sdk/studio "Lightning SDK Studio reference"
[2]: https://lightning.ai/docs/platform/inference/batch-jobs/submit-jobs "Submit Batch Jobs"
[3]: https://lightning.ai/docs/platform/developers/cli/batch "Lightning CLI Batch Jobs"
[4]: https://lightning.ai/docs/platform/build/ai-studio/background-execution "Studio Background Execution"
[5]: https://lightning.ai/docs/platform/team-management/teamspaces/drive "Teamspace Drive"
[6]: https://lightning.ai/docs/platform/data/datasets/import-dataset "Import Datasets and Connect S3"
[7]: https://lightning.ai/pricing "Lightning AI Pricing"
[8]: https://lightning.ai/docs/platform/overview/getting-started "Lightning AI Getting Started"
