# Lightning AI Studio — 공식 정책 조사

**확인일:** 2026-11-05 · 공개 Pricing/제품/문서 페이지 기준. 계정에는 로그인하지 않았습니다.

## 무료 GPU와 크레딧

Lightning AI의 Free 플랜은 사용량 기반 GPU 요금에 적용되는 **초기 프로모션 크레딧**을 제공합니다. Pricing 페이지는 최대 30 크레딧(대략 50 GPU 시간)을 안내하고, AI Studio 시작 문서는 전화번호 인증 후 최대 80 GPU 시간이라는 별도 수치를 안내합니다. 두 페이지의 수치가 일치하지 않으므로 사용자별 확정량으로 합치지 않습니다. 자료에는 이 GPU 시간이 매월 반복 지급된다고 나와 있지 않습니다. FAQ는 최초 5크레딧과 카드 추가로 받을 수 있는 25크레딧을 설명하고, 미사용 크레딧은 12개월 후 소멸한다고 안내합니다. 크레딧 소진 후 사용량은 초 단위로 과금될 수 있습니다. [1][2][3]

## GPU·세션·저장

Pricing 페이지에는 T4, L4, L40S, RTX 6000, A100, H100, H200의 GPU 및 시간당 가격이 표시됩니다. Free 플랜은 Studio당 GPU 1개, 동시 GPU 최대 2개로 안내되며, A100/H100/H200 Studio에는 4시간 세션 한도가 있습니다. T4/L4/L40S는 표에 session limit “Unlimited”로 표시되지만, 이는 무료 크레딧이 무제한이라는 뜻이 아닙니다. GPU별 당시 가용성은 계정 화면에서 확인해야 합니다. [1]

Studio는 정지 후 재시작할 때 Studio home 및 지정된 Studio 경로의 파일과 환경을 보존한다고 설명합니다. Lightning Drive는 Studio와 Job 사이에서 데이터·checkpoint를 공유하는 저장소입니다. 저장 용량은 공식 자료 간 차이가 있습니다. Pricing은 Free persistent storage 50GB를, Billing/Drive 문서는 Free 총 용량 200GB 및 처음 10GB를 초과한 저장소의 요금(월 $0.10/GB)을 안내합니다. 실제 적용량은 계정의 현재 플랜·청구 화면에서 확인해야 합니다. [1][3][4][6]

## 계정·결제 조건과 장시간 학습 적합성

Studio 계정과 전화번호 인증이 필요하다고 제품 문서는 설명합니다. Studio 시작 문서는 가입 시 카드가 필요하지 않다고 안내하지만, 추가 프로모션 크레딧에는 카드 추가 조건이 있습니다. 지역별 전화번호 지원 및 사용자의 실제 인증 상태는 확인하지 않았습니다. [1][2]

Lightning은 학습·fine-tuning·pretraining과 다중 GPU/다중 머신을 제품 용도로 설명합니다. 다만 초기 크레딧은 한정적이고 일부 GPU 세션은 4시간 제한이 있어, Free 조건에서 장시간 연속 pretraining을 보장한다고 볼 수 없습니다. checkpoint/resume로 세션을 나누어 운영하는 것이 적절합니다. 이는 문서화된 제한을 바탕으로 한 엔지니어링 판단이며, 학습 자체가 금지된다는 의미는 아닙니다.

## 확인하지 않은 계정별 사항

- 현재 지급된 크레딧, GPU 잔액·실제 할당, 카드 등록, 전화 인증, 선택 가능한 GPU 및 대기시간은 조회하지 않았습니다.
- 크레딧의 월별 반복 지급·리셋 주기와 50시간 대 80시간 안내 중 계정에 적용되는 수치는 확정하지 않았습니다.
- Free 저장 용량 50GB 대 200GB 안내의 차이도 공개 문서만으로 해결되지 않았습니다.

## 공식 출처

[1] [Pricing](https://lightning.ai/pricing) · [2] [AI Studio](https://lightning.ai/docs/platform/build/ai-studio) · [3] [Billing FAQ](https://lightning.ai/docs/platform/overview/faq/billing) · [4] [Start and stop Studio](https://lightning.ai/docs/platform/build/ai-studio/start-and-stop-studio) · [5] [Add GPUs](https://lightning.ai/docs/platform/build/ai-studio/add-gpus) · [6] [Lightning Drive](https://lightning.ai/docs/platform/data/drive) · [7] [Account creation FAQ](https://lightning.ai/docs/platform/overview/faq/create-account)
