# Google Colab — MiniCoder 클라우드 제공자 조사

**확인일: 2026-11-05**

## 무료 GPU 접근과 크레딧

Google은 Colab을 GPU와 TPU를 포함한 컴퓨팅 자원에 대한 무료 접근을 제공하는 서비스로 안내한다. 따라서 무료 GPU 접근 자체는 공식적으로 존재한다. 다만 무료 이용자가 GPU를 상시·보장받는다는 뜻은 아니다. Google은 무료 등급의 GPU 접근이 크게 제한되고 하드웨어 가용성과 사용 제한을 동적으로 바꾼다고 설명한다. [1] [2]

무료 등급은 유료 구독의 1회 체험 크레딧으로 설명되지 않는다. 확인한 공식 문서는 이를 무료 서비스 등급으로 설명하지만, 무료 사용자에게 정기적으로 지급되는 월간 compute units, 고정 GPU 시간 할당량, 또는 GPU 사용 제한의 정해진 초기화 주기를 공개하지 않는다. Google은 전체 사용량 한도와 유휴 타임아웃, VM 최대 수명, GPU 종류가 시간에 따라 달라지며 한도를 게시하지 않는다고 밝힌다. 따라서 “무료 GPU 시간은 매월 리셋된다”거나 고정량의 무료 크레딧이 있다는 주장은 공식 근거를 확인하지 못했다. [1] [2] [7]

유료 Colab 상품은 별도 범주다. 공식 소개는 Pro와 Pro+의 월간 compute units 및 더 강력한 가속기를 언급하고, FAQ는 유료 사용자의 compute units 잔액이 소진되면 무료 등급 제한으로 돌아간다고 명시한다. 결제 약관은 관련 유료 기능에 일회성 또는 반복 요금이 적용될 수 있다고 하지만, 이 문구는 무료 GPU가 일회성 체험이라는 뜻이 아니다. 가격·단위 수량을 확인하려고 연 공식 가입 페이지는 로그인 화면만 반환해 이 조사에서는 구체적인 가격표를 검증하지 못했다. [1] [4] [5] [6]

## GPU와 런타임

Colab 구독 안내 노트북은 무료 버전이 Nvidia T4 GPU 접근을 제공한다고 적지만, 할당은 사용량 제한과 가용성에 달려 있다고 덧붙인다. FAQ는 GPU·TPU 유형이 시간에 따라 달라진다고 하며 유료 플랜도 프리미엄 GPU 접근이 가용성에 좌우된다고 설명한다. 따라서 T4는 공식 안내에 등장하는 무료 GPU 옵션이지만 모든 세션에서 이를 받을 수 있다고 보장되지는 않는다. 이 자료들은 GPU 모델별 고정 카탈로그나 사용자의 현재 배정을 보장하지 않는다. [1] [3]

무료 런타임은 가용성과 사용 패턴에 따라 최대 12시간 실행될 수 있으며, 유휴 상태면 타임아웃된다. 일반 Colab VM에는 서비스가 정한 최대 수명이 있고, VM은 일정 시간 유휴 후 삭제된다. Colab은 전체 사용량 한도, 유휴 제한, 최대 VM 수명 등 구체 수치를 공개하지 않는다. Pro+는 compute units가 충분한 경우 최대 24시간 연속 코드 실행을 지원한다고 FAQ에 명시돼 있다. 이는 무료 런타임의 보장이 아니다. [1] [3]

## 계정·결제 요구

Colab 공식 화면은 로그인을 요구하고, FAQ는 실행 코드가 사용자 계정 전용 VM에서 동작한다고 설명한다. 유료 Colab 서비스를 구매하려면 Google Payments 계정이 있어야 하고 관련 Google·Colab 약관에 동의해야 한다. 확인한 공식 Colab 자료에는 무료 GPU 이용을 위한 별도 전화번호 인증 또는 카드 등록 의무가 명시돼 있지 않다. 유료 결제에 필요한 구체적인 카드 종류·결제수단은 이 자료들만으로 확정할 수 없다. [1] [5] [6]

## 저장·지속성

Colab 노트북은 Google Drive에 저장되거나 GitHub에서 불러올 수 있다. 반면 코드 실행 VM은 계정별 임시 환경으로, 유휴 상태 이후 삭제될 수 있고 최대 수명도 적용된다. VM에 설치한 사용자 라이브러리와 파일은 노트북을 공유해도 함께 전달되지 않는다. 따라서 모델 체크포인트와 학습 진행 상태를 보존하려면 런타임 VM 바깥의 영속 저장소에 기록해야 한다는 점이 운영상 중요하다. [1]

Drive 마운트 파일 작업은 실행 중인 런타임에 의존한다. Colab을 통한 Drive 파일 이동이 중단되면 전송 중인 데이터가 손실될 수 있고, Drive 저장공간을 추가 구매해도 Colab VM 로컬 디스크가 커지지는 않는다. [1]

## 장시간 사전학습 적합성

**판정: 무료 Colab은 장시간 연속 사전학습의 신뢰 가능한 실행 기반으로 보기 어렵다.** Google은 Colab이 대화형 컴퓨팅을 우선한다고 밝히고, 무료 세션에 최대 12시간(가용성·사용 패턴에 따라 다름)의 실행 상한, 유휴 종료, 동적이며 비공개인 자원 제한을 둔다. 그러므로 작업이 한 세션을 넘어가거나 연속 GPU 점유가 필요한 사전학습은 종료·재시작에 대비해야 하며, 무료 GPU의 연속 배정이나 재현 가능한 할당량을 전제로 설계해서는 안 된다. 이는 공식 제한에 근거한 엔지니어링 적합성 판단이지, Google이 모든 사전학습 작업을 금지한다는 뜻은 아니다. FAQ가 제시하는 Pro+ 최대 24시간 연속 실행도 compute units 잔액 조건이 붙으며 무료 기능이 아니다. [1] [3]

## 계정별 미확인 사항

사용자의 현재 무료 GPU 할당 여부·잔여 사용량·사용량 제한, compute units나 크레딧 잔액, 결제수단/카드 등록 상태, 전화번호 인증 상태, 실제 배정될 GPU 모델 및 해당 시점 가용성은 확인하지 않았다. 이 항목들은 공개 문서만으로 추정할 수 없으며, 특히 계정에 로그인하거나 런타임에 접속하지 않고는 계정별 상태를 알 수 없다. [1] [2] [6]

**References**

[1]: https://research.google.com/colaboratory/faq.html "Google Colaboratory FAQ"
[2]: https://research.google.com/colaboratory/faq.html?hl=ko "Google Colab 자주 묻는 질문"
[3]: https://colab.research.google.com/notebooks/pro.ipynb "Making the Most of your Colab Subscription"
[4]: https://colab.google/ "Google Colab"
[5]: https://research.google.com/colaboratory/tos_v3.html "Colab Paid Services Additional Terms of Service"
[6]: https://colab.research.google.com/signup "Colab Paid Services Pricing"
[7]: https://developers.google.com/colab "Google for Developers — Colab"
