# Kaggle Notebooks: 외부 학습 실행 가능 범위

**기준일: 2026-11-05.** 아래 내용은 해당 날짜 기준 공개된 Kaggle 공식 문서와 Kaggle 공식 CLI 저장소를 확인해 정리했다. 실제 Kaggle 계정에 로그인하거나 실행·쿼터를 시험하지 않았으며, 특정 계정의 자격·GPU 잔여량·가용성은 확인하지 않았다.

## 외부 컨트롤러의 실행 제출

**가능하다.** 공식 Kaggle CLI의 `kaggle kernels push -p <폴더>`는 로컬의 노트북/스크립트와 `kernel-metadata.json`을 Kaggle에 올린 뒤 실행한다. `--no-run`은 저장만 하고 실행하지 않는다. 같은 CLI에서 실행 상태를 조회하고(`kernels status`) 완료된 실행의 산출물을 내려받을 수 있다(`kernels output`). 즉, 사전에 유효한 계정 자격정보를 안전하게 주입하면 컨트롤러가 매번 사람의 브라우저 조작 없이 **배치형 노트북 실행**을 제출·추적할 수 있다. [1] [2]

공식으로 확인되는 무인 실행 인터페이스는 이 CLI 경로다. Kaggle의 공개 API 안내는 자동화에서 CLI/API 토큰 사용을 안내하지만, 여기서 확인한 공식 자료에는 임의 노트북 실행을 위한 안정된 직접 REST `run` 엔드포인트 계약이 기재되어 있지 않다. 따라서 구현은 비공개 웹 엔드포인트 호출이나 UI 자동화가 아니라, CLI 호출로 한정하는 것이 정직하다. [1] [2]

일반적인 흐름은 프로젝트 코드와 노트북을 Kaggle 실행 폴더에 준비하고, `kernel-metadata.json`에서 소유자/커널 ID, `notebook` 유형, 입력 데이터, GPU 사용 여부와 `machine_shape`를 지정한 다음 `kaggle kernels push`를 호출하는 것이다. 이후 상태를 폴링하고 성공한 버전의 출력 파일을 가져온다. CLI 문서는 `--accelerator`도 지원하며, 메타데이터는 GPU와 장비 유형을 설정한다. 선택한 가속기 사용 가능 여부는 해당 계정의 쿼터·자격 및 Kaggle의 장비 제한에 달려 있다. [2] [3]

이는 브라우저의 인터랙티브 편집 세션과 다르다. 인터랙티브 노트북은 사용자가 편집기에서 세션을 열고 셀을 실행한다. `Save & Run All`은 별도의 깨끗한 세션에서 처음부터 끝까지 실행해 버전을 만든다. CLI `push`의 실행은 이와 같은 비대화형 버전 실행 경로로 취급해야 하며, 이미 열려 있는 인터랙티브 세션의 메모리 상태를 외부 컨트롤러가 이어받는 기능으로 보면 안 된다. [2] [4]

## 런타임, 체크포인트와 영속성

Kaggle 문서상 CPU/GPU 노트북 세션은 최대 12시간, TPU는 최대 9시간이다. 문서의 기술 사양은 `/kaggle/working`에 자동 저장되는 디스크를 최대 20GB로 제시하고, 세션 중 임시 scratch 공간은 세션 밖에 저장되지 않는다고 구분한다. 이 제한 때문에 긴 학습은 그 안에서 끝난다고 가정하지 말고 단계적으로 재개할 수 있게 구성해야 한다. [4]

후속 실행에 남길 체크포인트·모델·로그는 코드가 `/kaggle/working`에 명시적으로 쓰도록 하고, 완료된 실행의 노트북 버전 산출물로 저장해야 한다. Kaggle은 노트북 출력 파일을 이후 노트북의 입력으로 연결할 수 있고, 출력 파일을 CLI로 회수할 수 있다고 문서화한다. 여러 단계의 장기 보존·재개가 필요하면 산출물을 Dataset으로 버전 업로드하고 다음 실행의 입력 데이터로 연결하는 방식도 공식 CLI에 있다. 다만 이는 별도 저장/업로드 단계이지 실행 세션의 메모리나 임시 scratch를 영속화하는 기능이 아니다. [2] [4] [5]

## 인증, 비용과 계정별 게이트

공식 CLI의 API 토큰은 Kaggle 계정 설정 페이지에서 생성해 환경 변수 `KAGGLE_API_TOKEN` 또는 토큰 파일로 전달할 수 있다. 토큰은 자동화 스크립트에 쓸 수 있으므로, 무인 실행 전에 사람이 계정과 자격정보를 준비·승인해 비밀 저장소에 넣는 운영 절차는 필요하다. OAuth를 사용한다면 문서상 사용자가 Kaggle 로그인 후 요청 권한을 승인해야 한다. 따라서 **최초 계정 등록·토큰 발급/승인까지 사람 없이 가능하다는 뜻은 아니다.** [1] [6]

Kaggle은 GPU를 무료로 제공한다고 안내하며, GPU 사용량은 주간 쿼터를 적용한다. 공식 GPU 안내의 기준값은 주 30시간이며 수요와 자원 상황에 따라 더 높을 수 있고 주마다 재설정된다. 이는 고정 보장량이나 특정 계정의 잔여량이 아니다. CLI 문서는 일부 가속기가 특정 대회 참가자 또는 Kaggle 관리자에게만 제공될 수 있다고 명시한다. 이 문서들에는 각 작업의 과금 단가가 제시되어 있지 않으므로 유료 GPU 시간이나 계정별 이용 가능 여부를 약속하지 않는다. [2] [7]

## 구현 경계와 제한

- **가능한 경계:** 계정이 준비되고 사용 가능한 쿼터/가속기가 있다는 전제 아래, 외부 스케줄러가 공식 CLI로 노트북 버전을 제출하고 상태 확인 및 결과 수집을 자동화한다.
- **불가능하거나 보장되지 않는 경계:** 계정 생성·인증·OAuth 승인 없이 제출하기, 임의 REST 실행 API가 있다고 가정하기, 대화형 세션을 무인 원격 조작 대상으로 취급하기, 12시간 제한을 넘는 단일 세션, 세션 간 메모리/임시 파일 보존, GPU 쿼터·특정 장비·실행 성공을 보장하기.
- Kaggle은 예약된 노트북 실행 기능도 문서화하지만 예약 가능한 노트북 수에 제한이 있고, 공개 문서에서 제한의 수치를 고정해 제시하지 않는다. 외부 이벤트나 프로젝트별 스케줄을 제어해야 한다면, 준비된 자격정보로 컨트롤러가 CLI를 호출하는 쪽이 명확한 경계다. [4]
- Kaggle API는 동적 요청 제한을 적용하며 429 응답이 발생할 수 있으므로, 컨트롤러는 호출량을 제한하고 대기 후 재시도해야 한다. [1]

## References

[1]: https://www.kaggle.com/docs/api "Kaggle Public API, authentication, and rate limits"
[2]: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md "Kaggle CLI kernel commands"
[3]: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md "Kaggle CLI kernel metadata"
[4]: https://www.kaggle.com/docs/notebooks "Kaggle Notebooks documentation"
[5]: https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md "Kaggle CLI dataset commands"
[6]: https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md "Kaggle CLI authentication setup"
[7]: https://www.kaggle.com/docs/efficient-gpu-usage "Kaggle efficient GPU usage tips"
