# Hugging Face Spaces ZeroGPU — 2026-11-05 기준

**결론:** ZeroGPU에는 외부 브라우저 없이 호출할 수 있는 공식 API가 있지만, 대상은 Space가 미리 노출한 Gradio 함수다. 노트북 셀 실행이나 임의의 학습 명령을 제출하는 ZeroGPU 작업 API는 문서화되어 있지 않다. 따라서 짧고 제한된 학습 함수를 Gradio Space에 직접 구현하면 API로 호출할 여지는 있지만, 이 저장소의 학습 실행을 일반적인 장시간 GPU 작업으로 제출하는 용도에는 ZeroGPU가 맞지 않는다. [1] [2] [3]

## 제출 방식과 런타임

Gradio Space는 자동으로 API 엔드포인트를 제공하고 Python 클라이언트 또는 HTTP로 호출할 수 있다. ZeroGPU 런타임에서는 Gradio SDK Space에 `spaces`를 가져와 GPU 작업 함수에 `@spaces.GPU`를 붙인다. 함수 호출 중 GPU가 할당되고 종료 시 반환된다. 기본 GPU 실행시간은 60초이며 `duration`으로 함수별 시간을 설정할 수 있다. 그러므로 외부 컨트롤러는 사전에 작성한 Space 함수만 API로 실행할 수 있고, 노트북이나 임의의 코드/명령을 직접 제출할 수는 없다. Hub의 노트북 문서는 노트북을 렌더링하고 Colab/Kaggle로 여는 흐름을 설명할 뿐, 외부 실행 API를 설명하지 않는다. [1] [2] [3]

이와 구분되는 공식 대안인 Hugging Face **Jobs**는 CLI 및 `huggingface_hub.run_job()`/`run_uv_job()`로 명령·스크립트와 GPU flavor를 제출하고 상태·로그를 조회할 수 있다. 학습 가이드는 이를 학습에 쓰는 예도 제공한다. 이는 ZeroGPU가 아니라 유료 크레딧 게이트가 있는 별도 컴퓨트 제품이다. [4] [5] [8]

## 체크포인트와 계정/비용 게이트

Space의 기본 파일시스템은 재시작·중지 후 사라진다. 영속 저장에는 Storage Bucket을 Space 볼륨으로 연결해 체크포인트 저장·재개 경로로 써야 한다. 버킷은 변경 가능한 저장소이며, ZeroGPU가 자동으로 학습 상태를 보존하는 것은 아니다. [6] [7]

ZeroGPU의 문서상 일일 GPU 할당량은 비인증 사용자 2분, 무료 계정 5분, PRO/Team 사용자 40분, Enterprise 사용자 60분이며 첫 사용 24시간 후 리셋된다. `xlarge`는 할당량을 2배 소모한다. PRO/Team/Enterprise는 포함량 초과 후 선불 크레딧으로 $1/10분에 이용할 수 있다. 무료 개인 계정의 Space 호스팅은 계정 정상 상태, 이메일 인증, 계정 생성 후 30일 경과 조건과 최대 2개 한도가 있고, PRO는 최대 10개, 조직은 최대 50개다. 이 조건은 문서의 정책일 뿐 특정 계정에서 이용 가능한지는 확인하지 않았다. [1]

Jobs는 인증과 양의 크레딧 잔액을 요구하며, hardware 사용량 기준 과금, 기본 30분 timeout, 실행 종료 후 삭제되는 임시 파일시스템을 갖는다. 장시간 작업은 timeout을 설정하고 Bucket 볼륨이나 Hub 저장소로 결과를 내보내야 한다. Jobs 정책을 ZeroGPU 정책과 혼동하지 말아야 한다. [4] [5] [8]

## 이 프로젝트에 적용할 경계

현재 `providers/huggingface_zerogpu.py`는 로컬 Space 환경·GPU를 점검하는 감지기이며, 클라우드 제출·원격 Space 호출·영속 저장 검증은 구현하지 않는다. 저장소의 학습기는 체크포인트 재개를 지원하지만, 이를 ZeroGPU에 올리려면 Gradio 입력·출력 함수, ZeroGPU 데코레이터, 의존성/데이터 제공, Bucket 마운트 경로를 별도 통합해야 한다. 따라서 현재 구현을 “노트북/GPU 작업 제출 기능”으로 간주하면 안 된다. 작은 단계의 학습을 사전 정의 함수로 감싸는 실험은 가능하나, 호출별 실행시간·할당량 제약 때문에 장시간 학습·분산 학습의 적합성을 보장할 수 없다. 이 프로젝트에 일반적인 원격 학습 job 제출이 필요하면 공식적으로 그 흐름을 제공하는 Hugging Face Jobs가 별도 후보이며, 계정 권한·크레딧·실제 하드웨어 적합성은 사용자 환경에서 확인해야 한다. [1] [4] [5] [6]

**실계정 테스트:** 수행하지 않음. 계정 가용성, 현재 할당량/크레딧, 실제 실행 성공 여부를 주장하지 않는다.

## References

[1]: https://huggingface.co/docs/hub/spaces-zerogpu "Spaces ZeroGPU"
[2]: https://huggingface.co/docs/hub/spaces-api-endpoints "Spaces as API endpoints"
[3]: https://huggingface.co/docs/hub/notebooks "Jupyter Notebooks on the Hugging Face Hub"
[4]: https://huggingface.co/docs/huggingface_hub/guides/jobs "Run and manage Jobs"
[5]: https://huggingface.co/docs/hub/jobs-training "Train Models on Jobs"
[6]: https://huggingface.co/docs/hub/spaces-storage "Disk usage on Spaces"
[7]: https://huggingface.co/docs/hub/storage-buckets-access "Storage Bucket access patterns"
[8]: https://huggingface.co/docs/hub/jobs-quickstart "Jobs quickstart"
