# Modal — Python 학습 프로젝트 제공자 검토

**기준일: 2026-11-05.** 공식 Modal 문서/API 참조만 확인했다. 실제 계정 로그인, 권한, 결제수단, 크레딧, GPU 할당이나 프로젝트 실행은 시험하지 않았다.

## 결론과 제출 경계

Modal Notebook 편집기는 2026-10-23부터 종료되어 새 노트북을 만들거나 실행할 수 없다. 기존 `.ipynb`는 다운로드만 가능하다. 따라서 Modal Notebook을 외부에서 시작하는 공식 제출 API는 확인되지 않는다. 이는 노트북 편집기를 사람이 브라우저에서 시작하는 흐름과도 다르다. [1] [2]

반면, 외부 컨트롤러는 미리 배포된 Modal App의 Function을 공식 SDK로 찾아 호출할 수 있다. Python의 `modal.Function.from_name(app, function).spawn(args)`는 비동기 작업을 시작하고 `FunctionCall` 핸들을 반환한다. 완료 상태/결과는 핸들로 확인할 수 있다. Modal은 JavaScript와 Go SDK도 문서화한다. 그러므로 **사전 배포된 GPU Function을 호출하는 데 매번 사람의 브라우저 조작은 필요하지 않다**. 단, 이는 노트북 파일/셀을 제출하는 API가 아니라 사용자가 앱 코드로 정의한 원격 함수 실행 API다. [3] [4]

## 지원되는 실행 흐름

학습 프로젝트는 `@app.function(gpu="...")`로 GPU 함수를 정의하고, 학습 코드·의존성을 Modal Image에 구성한다. 운영 앱은 `modal deploy`로 먼저 배포하고, 컨트롤러는 배포된 앱/함수 이름을 조회한 뒤 `spawn()`으로 작업을 제출한다. Modal CI 문서는 토큰을 CI 비밀값으로 제공해 `modal deploy`하는 비브라우저 배포 흐름도 안내한다. 제출 API는 SDK 기반이며, 문서화된 별도 노트북 실행 API와 혼동하면 안 된다. [3] [5] [6]

노트북에서 Modal을 개발하는 공식 방법은 별개다. Jupyter에서 `import modal` 후 `with app.run():`로 임시 앱을 실행하고, 셀에서 `.remote()`를 호출할 수 있다. 이때 로컬·원격 Python 버전을 맞춰야 하며, interactive shell/functions는 노트북에서 지원되지 않는다. Modal 자체의 Notebook 편집기/API가 되살아나는 방식은 아니다. [2] [7]

## 체크포인트와 영구 저장

이름 있는 `modal.Volume`을 Function에 마운트해 체크포인트 디렉터리로 쓰는 것이 공식 영구 저장 경로다. 학습 루프에서 안전한 간격으로 체크포인트 파일을 저장하고 `volume.commit()`을 호출하면 다른 실행에서도 보이는 내구 저장 상태가 된다. Modal은 실행 중 수 초 간격의 백그라운드 커밋과 컨테이너 종료 시 최종 스냅샷도 설명하지만, 중요한 체크포인트의 보존 시점을 코드에서 명시적으로 커밋하는 편이 안전하다. 새 볼륨은 Volume v2를 권장한다. 같은 파일에 동시 기록하면 last-write-wins이므로 실행별 경로를 분리해야 한다. [8] [9]

## 인증, 비용 및 한도

Function 조회와 호출은 Modal API 토큰으로 인증된다. 토큰은 `MODAL_TOKEN_ID`와 `MODAL_TOKEN_SECRET` 환경변수 또는 활성 CLI 프로필에서 읽는다. 공유 워크스페이스의 Service User는 자동화용이지만 Team/Enterprise 플랜 기능이며, 기본적으로 어떤 Environment에도 접근 권한이 없으므로 허용된 Environment에 명시적으로 역할을 부여해야 한다. `modal token new`는 인증된 웹 세션을 사용한다. 즉, 이미 발급·전달된 자격 증명으로 작업을 무인 실행할 수는 있지만, 최초 계정/토큰/권한 설정까지 브라우저 없이 가능하다고 단정할 근거는 없다. [3] [10] [11]

Modal 문서는 GPU 사용에 유효한 결제수단이 등록돼 있어야 한다고 명시한다. 가격 페이지상 Starter는 월 $30 컴퓨트 크레딧 및 동시 GPU 10개를 표시하며, 사용량 기반 GPU·CPU·메모리 요금과 Volume 저장료가 별도 기준으로 제시된다. 크레딧이나 결제수단이 이 프로젝트 계정에 실제로 있는지는 확인하지 않았다. [12] [13]

실행 제한도 설계에 반영해야 한다. Function 기본 타임아웃은 5분이며 설정 가능한 최대 실행시간은 24시간이다. 비동기 `spawn()`은 호출한 프로세스가 종료돼도 계속 실행되고 결과는 7일 보관되지만, 동기 호출은 호출자가 연결을 끊으면 2분 안에 취소된다. 동기 입력은 대기열 2,000건·시스템 총 25,000건 제한을 받는 반면, 비동기 입력은 최대 100만 건까지 대기열에 둘 수 있다. 컨테이너 장애/선점 시 재시도될 수 있으므로 학습 체크포인트 저장과 완료 처리는 멱등성을 고려해야 한다. [4] [5]

## 구현 경계

구현 가능한 경계는 **사용자가 준비한 Modal 토큰·Environment 권한·결제 조건을 전제로**, 학습 코드를 GPU Function으로 패키징하고 배포한 다음 SDK로 비동기 제출, 작업 결과 확인, Volume 체크포인트 읽기/쓰기를 연결하는 선택형 백엔드다. Modal Notebook 편집기의 노트북 실행을 외부 API로 자동화하거나, 계정의 GPU 권한·잔액·동시성 여유를 보장하는 기능으로 설명하면 안 된다. [1] [3] [8] [10] [12]

## References

[1]: https://modal.com/changelog/sunsetting-modal-notebooks "Sunsetting Modal Notebooks"
[2]: https://modal.com/docs/guide/notebooks "Modal Notebooks"
[3]: https://modal.com/docs/guide/trigger-deployed-functions "Invoking deployed Functions"
[4]: https://modal.com/docs/guide/function-invocation-methods "Function invocation methods"
[5]: https://modal.com/docs/guide/functions "Modal Functions"
[6]: https://modal.com/docs/guide/continuous-deployment "Continuous deployment"
[7]: https://modal.com/docs/guide/jupyter-notebooks "Jupyter notebooks"
[8]: https://modal.com/docs/guide/volumes "Volumes"
[9]: https://modal.com/docs/sdk/py/latest/Volume "Volume SDK reference"
[10]: https://modal.com/docs/guide/service-users "Service Users"
[11]: https://modal.com/docs/cli/latest/token "modal token"
[12]: https://modal.com/docs/guide/gpu "GPU acceleration"
[13]: https://modal.com/pricing "Modal pricing"
[14]: https://modal.com/docs/guide/timeouts "Timeouts"
