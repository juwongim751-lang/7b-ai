# Local GPU — 외부 제출과 실행 경계 (2026-11-05)

## 결론

**Local GPU는 호스팅 서비스나 클라우드 provider가 아니라, MiniCoder 프로세스가 이미 실행 중인 컴퓨터의 GPU를 뜻한다.** 프로젝트의 `LocalProvider`는 현재 Python 프로세스에서 GPU/RAM/디스크를 검사하고, 단일 프로세스용 `torchrun` 명령을 반환한다. 이 어댑터에 원격 작업 제출, 머신 생성, 계정·쿼터 조회 또는 persistent storage 검증 기능은 없다. 따라서 Local GPU만을 대상으로 한 provider의 공식 job-submission API는 없다.

그렇다고 외부 컨트롤러가 무조건 사람 브라우저를 거쳐야 하는 것은 아니다. 컨트롤러가 같은 호스트에서 허가된 OS 사용자로 실행되면 프로세스를 직접 시작할 수 있다. 사용자가 관리하는 원격 호스트라면 사전 설정된 SSH 인증과 접근 권한으로 원격 명령을 실행할 수 있다. 이는 일반 OS/SSH 통제이지, GPU 제공자의 job API도 아니고, 접근 권한을 얻거나 호스트를 할당해 주는 수단도 아니다. 기존 노트북 파일 자체를 서비스에 제출해 관리형 GPU 작업으로 돌리는 공식 API도 조사한 공식 자료에서는 확인되지 않았다. [1] [2] [3]

## 외부 컨트롤러와 노트북 API

PyTorch의 `torch.cuda.is_available()`는 **현재 시스템에서** CUDA를 사용할 수 있는지 나타내고, `device_count()` 등으로 현재 프로세스가 볼 수 있는 장치를 조사한다. 이는 호스트/드라이버/설치 상태를 검사하는 기능이지 GPU를 요청·할당하는 기능이 아니다. Python `subprocess.run()`은 허가된 현재 호스트에서 프로그램을 시작하고 완료·종료 코드를 받을 수 있으며, OpenSSH `ssh destination command`는 인증 후 원격 호스트에서 명령을 실행한다. 원격 경우에는 사용자가 호스트, 계정·키, 네트워크와 실행 권한을 먼저 제공해야 한다. [1] [2] [3]

Jupyter를 사용한다면 **노트북 파일 제출 API와 이미 실행 중인 로컬 Notebook Server의 API를 구분**해야 한다. Jupyter Server는 `POST /api/sessions`로 세션을 만들고 커널을 시작하며, 커널 API는 `/api/kernels`로 커널을 만들 수 있다. 클라이언트는 커널 채널에 `execute_request`를 보내 코드(셀 문자열)를 실행한다. 이에 따라 이미 가동 중인 Jupyter Server의 URL과 접근 자격을 갖춘 컨트롤러는 REST/WebSocket 및 커널 프로토콜로 브라우저 없이 세션/코드 실행을 제어할 수 있다. 로컬 프로세스 권한으로 서버를 먼저 시작하는 것도 가능하지만, 이것은 사용자가 관리하는 호스트에서의 오케스트레이션이다. Jupyter REST API는 Local GPU 클라우드 할당 API가 아니며, `.ipynb` 파일을 독립 원격 큐에 등록하는 API로 설명해서는 안 된다. 사람에게서 브라우저를 통한 최초 GPU 설정·호스트 승인 등이 필요한 환경이면 이 API가 그것을 우회하지 않는다. [4] [5] [6]

이 프로젝트의 실용적인 실행 경로는 노트북 UI 자동화보다 재현 가능한 스크립트다. 저장소와 의존성을 실행 호스트에 준비하고, `torch.cuda.is_available()` 및 GPU·VRAM을 검사한 뒤, 프로젝트 안내에 따라 `python -m training.trainer --config config/model_tiny.yaml --data-dir data/processed --run-dir runs/tiny --max-steps 8`처럼 기존 진입점을 실행한다. 체크포인트가 호스트의 다른 보존 경로에 있어야 하면 train config의 `checkpoint_dir`를 명시하고 이후 `--resume`을 사용한다. 현재 프로젝트 Local adapter의 `run_worker_command`는 rank 0/world size 1만 지원하며 `torchrun`이 설치되어 있을 때 `training/trainer.py` 명령 문자열을 돌려준다. 그 자체가 명령을 시작·감시·재시도하는 제출 API는 아니다. [1] [7]

## 체크포인트와 지속성

MiniCoder 학습 코드는 `runs/tiny`에 메트릭·요약과 체크포인트를 남기고, 프로젝트 문서상 복구 체크포인트에는 model뿐 아니라 optimizer, scheduler, scaler, RNG와 학습 cursor도 포함한다. PyTorch 공식 가이드는 학습 재개용 체크포인트에 모델과 optimizer `state_dict` 등 재개 상태를 함께 직렬화하는 방법을 제시한다. 로컬의 안정된 디스크/별도 마운트 경로가 프로세스 재시작 후에도 보존된다면 거기서 이어 학습할 수 있다. 그러나 디스크 장치·디렉터리의 존재와 보존 수명은 호스트 관리자가 책임지며, 현재 프로젝트 LocalProvider도 `persistent_storage_verified: false`로 표시한다. 경로만 `checkpoints/`라고 해서 영구 저장소가 보장되지는 않는다. [7] [8]

체크포인트를 호스트·마운트 종료 이후에도 보존하려면 사용자가 승인한 공유 파일시스템이나 오브젝트/원격 저장소에 명시적으로 복사하고, 다음 실행에서 복원해야 한다. 프로젝트 저장소 adapter는 이미 마운트된 디렉터리 또는 별도로 허용한 Hub 전송을 지원할 뿐, 디스크를 자동 마운트하거나 전송을 시작하지 않는다. [9]

## 인증, 비용과 자격 게이트

별도의 Local GPU 제공자 계정·구독·GPU 시간 요금은 이 로컬 실행 모델에 문서화되어 있지 않다. 실제 비용은 사용자가 보유·운영하는 장비 및 전력·호스트 계약에 달릴 수 있으나 공개된 고정 요금이나 이 호스트의 요금은 없다. 여기서 클라우드 계정·GPU 또는 잔여 쿼터의 존재를 추정하지 않는다.

같은 호스트에서 직접 실행할 때 필요한 경계는 해당 OS 사용자·프로세스 권한과 실행 가능한 Python/CUDA 설치다. Jupyter Server의 공식 보안 설정은 기본으로 토큰 인증을 사용하며 API 요청은 토큰으로 인증할 수 있다. 원격 Jupyter API를 쓸 때는 도달 가능한 서버와 유효한 토큰/접근 정책을 사전에 안전하게 설정해야 한다. SSH 원격 실행은 원격 계정에 신원을 증명하고 서버 측에서 허가된 명령만 가능하다. GPU 실행 전에 명시적 GPU/VRAM 검사와 짧은 실제 학습 벤치마크가 필요하다. PyTorch의 CUDA 탐지는 실행 환경이 CUDA를 지원하는지를 확인할 뿐, 하드웨어 적합성·장기 가용성·사용 권한·성능을 보증하지 않는다. [1] [3] [5]

## 제한과 구현 경계

- **공식 Local GPU job API:** 없음. 프로젝트 안의 `LocalProvider`는 현재 프로세스 조사기다. 외부 scheduler가 이를 원격 GPU 자원 제출 API처럼 부르면 안 된다.
- **브라우저 없이 실행:** 이미 권한이 있는 같은 호스트에서 OS 프로세스를 시작하거나, 사전 승인된 SSH/Jupyter 자격증명을 통해 관리 호스트에 명령·코드를 보낼 수 있다. 이것은 일반 호스트 오케스트레이션이다. 컨트롤러가 브라우저 없이 새 장비를 프로비저닝하거나 사용자 대신 계정·권한·결제를 생성한다고 뜻하지 않는다.
- **노트북 의미:** Jupyter API는 서버/세션/커널이 존재하는 상태에서 세션 생성과 커널 코드 실행을 제공한다. `execute_request`로 Python 코드를 실행할 수 있지만, 해당 일반 API만으로 노트북 파일이 별도 provider job으로 제출됐다고 볼 수 없다. 순수한 학습 job에는 스크립트 진입점이 더 직접적이다.
- **세션·쿼터:** Local GPU provider의 세션 시간·주간 쿼터·할당량은 정의되지 않는다. 실행 프로세스는 OS/시스템 관리자의 종료, 호스트 절전·재부팅, 자원 경합, 디스크 정책의 영향을 받는다. 임의의 고정 세션 시간 또는 유휴 한도를 붙이지 않는다.
- **자원 범위:** CUDA 장치가 보여야 하며 설치된 PyTorch/CUDA/드라이버 조합이 맞아야 한다. 현재 어댑터는 단일 프로세스만 제안하며 여러 GPU를 합산하거나 다중 호스트를 생성하지 않는다. 프로젝트 전체 성능, 모델 크기 적합성 또는 할당 성공은 검사 결과만으로 확정하지 않는다.
- **정직한 제품 경계:** `inspect → 자원·CUDA 확인 → 사용자 관리 런타임에서 학습 스크립트 실행 → 기존 체크포인트 경로에 저장/재개`까지가 이 프로젝트의 로컬 경로다. 권한 있는 프로세스 실행기와 체크포인트 I/O를 별도로 구현할 수는 있지만, 이는 사용자 호스트에 대한 로컬/SSH/Jupyter 통합이며 provider job API라고 부르지 않는다. 이 조사는 어떠한 실제 GPU·계정·SSH/Jupyter 서버 접속 또는 실행 시험도 하지 않았다.

## References

[1]: https://docs.pytorch.org/docs/2.15/cuda.html "PyTorch torch.cuda API"
[2]: https://docs.python.org/3/library/subprocess.html "Python subprocess management"
[3]: https://man.openbsd.org/ssh "OpenBSD ssh manual"
[4]: https://jupyter-server.readthedocs.io/en/latest/developers/rest-api.html "Jupyter Server REST API"
[5]: https://jupyter-server.readthedocs.io/en/latest/operators/security.html "Jupyter Server security"
[6]: https://jupyter-client.readthedocs.io/en/stable/messaging.html "Jupyter messaging protocol"
[7]: https://docs.pytorch.org/docs/2.14/elastic/run.html "PyTorch torchrun Elastic Launch"
[8]: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html "PyTorch saving and loading models"
[9]: https://docs.python.org/3/library/shutil.html "Python shutil high-level file operations"
