# Google Colab: 외부 자동 실행 경계 (2026-11-05)

**결론:** 외부 컨트롤러는 Google의 공식 Colab API를 통해 런타임을 만들고 Python 코드를 실행할 수 있다. 따라서 이미 인증과 API 설정이 끝난 경우, 매 실행마다 사람이 브라우저에서 노트북의 실행 버튼을 누를 필요는 없다. 다만 공식 인터페이스는 기존 `.ipynb`를 배치 작업으로 제출하는 노트북 큐가 아니다. GPU 런타임과 Jupyter 커널을 제어해 코드를 실행하는 형태다. [1] [2] [3]

## 공식 제출·실행 경로와 인증

공식 Colab REST API의 `POST https://colaboratory.googleapis.com/v1beta/runtimes`는 요청한 사용자의 런타임 할당을 만든다. Colab API 문서에는 런타임 생성·조회 외에 MCP 도구의 `create_session`과 `execute_code`가 명시되어 있고, `execute_code`는 런타임의 Jupyter 커널에서 Python 코드를 실행한다. 런타임 생성 응답의 연결 정보를 사용해 표준 Jupyter REST API 또는 클라이언트 라이브러리로 접속할 수도 있다. REST 생성 API는 `https://www.googleapis.com/auth/colaboratory` OAuth scope를 요구한다. Google Cloud MCP 사용에는 제품 API 활성화와 인증 설정이 필요하며, Google은 사용자·워크로드 등 인증 주체를 지원한다. 그러므로 사전 발급된 허용 자격증명을 가진 컨트롤러라면 실행 요청 자체는 브라우저 없이 호출 가능하다. 단, 최초 OAuth 동의나 프로젝트·자격증명 설정에 사람이 필요할 수 있고, 실제 계정이 Colab 런타임을 쓸 수 있는지는 별도 확인해야 한다. [1] [2] [4] [5]

기존 학습 코드는 컨트롤러가 프로젝트 파일을 런타임에 가져오고 의존성을 설치한 뒤, 세션에서 Python으로 학습 엔트리포인트를 실행하는 방식으로 연결한다. GPU 사양은 요청 주체 기준 `list_runtime_specs`에서 조회하며, 각 사양의 `eligible` 값을 확인한 뒤 런타임을 요청한다. 문서화된 도구는 런타임·세션·Python 실행 단위이므로, 컨트롤러가 `.ipynb` 파일을 그대로 제출해 Colab이 전체 노트북을 독립 배치 작업으로 예약·실행해 주는 API로 취급해서는 안 된다. MCP `execute_code`의 스트리밍 기한은 600초다. `executionId`를 지정하면 같은 세션에서 실행 결과 스트림에 재접속할 수 있지만, 이 기능은 런타임 수명이나 실행 자원 보장을 뜻하지 않는다. [1] [2] [3] [6]

## 체크포인트, 비용 및 제한

Colab VM은 유휴 상태나 서비스 최대 수명 도달 시 삭제될 수 있다. 파일은 세션 종료 후에도 런타임 파일시스템에 남을 수 있지만 런타임이 삭제되면 해당 상태도 제거된다. 학습 체크포인트를 VM 로컬 디스크에만 두지 말고 주기적으로 외부 영구 저장소로 내보내야 한다. 공식 I/O 예제는 Drive를 런타임에 마운트하거나 Drive REST API 및 Google Cloud Storage를 쓰는 방법을 보여 준다. Drive 마운트 예제는 인증 코드 입력을 요구하며, Colab FAQ도 새 런타임에서 Drive 접근에 대한 사용자 승인이 일반적으로 필요하다고 설명한다. 따라서 헤드리스 실행에서 Drive 접근이 자동 승계된다고 가정하지 말고, 저장소 별도 인증을 준비하거나 사전에 검증한다. [7] [8]

Google Cloud MCP를 사용하려면 프로젝트를 선택하고 결제를 활성화한 뒤 Colab API를 활성화해야 한다. 이는 MCP/API 사용을 위한 Cloud 프로젝트 설정 조건이다. Colab 컴퓨팅 접근은 별도로 Colab 계정의 구독·compute unit 및 자원 정책에 좌우된다. API의 `get_subscription`은 구독 등급만 반환하고, 런타임 사양 조회는 사양별 계정 자격 여부를 반환한다. 무료 GPU 자원은 크게 제한되고, 사용량 한도·유휴 제한·최대 VM 수명·GPU 종류는 시점과 사용 패턴에 따라 바뀌며 보장되지 않는다. FAQ상 일반 런타임은 최대 12시간이며 가용성·사용 패턴에 따라 더 짧을 수 있고, Pro+ 연속 실행은 충분한 compute unit이 있을 때 최대 24시간이다. 구체적인 한도나 현재 계정의 GPU 자격을 이 보고서에서 추정하지 않는다. [5] [8] [9] [10]

별도 경로인 Google의 오픈소스 `colab-mcp`는 로컬 에이전트를 **브라우저에서 열어 둔 Colab 노트북**에 연결하는 프록시다. 이 브라우저-연동 방식은 사람이 노트북을 연 상태에서 제어하는 흐름이며, 위의 Google Cloud 호스팅 `colaboratory.googleapis.com/mcp` API 경로와 혼동하지 않는다. 따라서 구현 경계는 “인증된 컨트롤러가 Colab 런타임을 시작하고 Python을 실행할 수 있음”까지다. 장시간·재시작 복구가 필요한 학습은 체크포인트와 재시작 로직을 프로젝트에 넣고, 실제 계정·GPU 자격·요금·시간 제한은 실행 전에 확인해야 한다. 이 조사에서는 어떤 Colab 계정도 시험하지 않았다. [12] [13]

## References

[1]: https://developers.google.com/colab/api/reference/mcp "Colab MCP Reference"
[2]: https://developers.google.com/colab/api/reference/rest/v1beta/runtimes/create "Colab REST API runtimes.create"
[3]: https://developers.google.com/colab/api/reference/mcp/tools_list/execute_code "Colab MCP execute_code"
[4]: https://docs.cloud.google.com/mcp/authenticate-mcp "Google Cloud MCP authentication"
[5]: https://docs.cloud.google.com/mcp/enable-disable-mcp-servers "Enable supported Google MCP products"
[6]: https://developers.google.com/colab/api/reference/mcp/tools_list/list_runtime_specs "Colab MCP list_runtime_specs"
[7]: https://colab.research.google.com/notebooks/io.ipynb "Colab external data and storage examples"
[8]: https://research.google.com/colaboratory/faq.html "Google Colab FAQ and resource limits"
[9]: https://developers.google.com/colab/api/reference/mcp/tools_list/get_subscription "Colab MCP get_subscription"
[10]: https://developers.google.com/colab/api/reference/mcp/tools_list/create_runtime "Colab MCP create_runtime"
[11]: https://developers.googleblog.com/announcing-the-colab-mcp-server-connect-any-ai-agent-to-google-colab/ "Announcing the Colab MCP Server"
[12]: https://github.com/googlecolab/colab-mcp "Google Colab MCP source repository"
