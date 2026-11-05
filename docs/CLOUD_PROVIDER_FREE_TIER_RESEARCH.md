# 클라우드 GPU 무료 이용 및 크레딧 비교

**비교 기준일: 2026-11-05**

무료 GPU 접근과 무료 크레딧은 다릅니다. 전자는 정해진 한도 내의 제한적 GPU 사용을 뜻하며 GPU 배정·즉시 가용성을 보장하지 않습니다. 크레딧은 사용량 요금에서 차감되는 금액이므로 결제수단 등록이나 소진 후 청구가 따를 수 있습니다.

## 한눈에 비교

| 공급자 | 무료 혜택의 성격 | GPU·사용량 제한 | 계정·결제 조건 및 유의점 | 장시간 사전학습 적합성 |
|---|---|---|---|---|
| **Google Colab** [1] [2] | 무료 등급에서 GPU/TPU 접근. 고정 GPU 시간 또는 월별 무료 크레딧은 공개 확인되지 않음 | 무료 세션 최대 12시간(가용성·사용 패턴에 따라 변동), 유휴 종료 가능. GPU 종류와 사용 한도는 공개된 고정값이 아님. T4는 공식 안내 예시이지 배정 보장이 아님 | 로그인 필요. 조사한 자료에서 무료 사용의 카드·전화 인증 필수 여부를 확인하지 못함. 유료 플랜은 별도 compute units | **낮음** — 지속 세션을 보장하지 않아 외부 저장·재개가 필요 |
| **Kaggle Notebooks** [3] [4] | 무료 GPU의 주간 반복 쿼터. 일반 안내는 주 30시간이나 수요·자원에 따라 달라질 수 있음 | 무료 단일 Tesla P100 안내 및 혼잡 시 대기열. 세션 최대 12시간. 유휴 타임아웃 공식 안내는 20분/60분으로 불일치 | 과거 Kaggle Staff 답변은 GPU/TPU 이용 전 전화 인증을 안내하지만 현행 Notebook 문서에서 재확인되지 않음. 카드 요건 미확인 | **낮음** — 세션·주간 쿼터·대기 제한이 있음 |
| **Lightning AI Studio** [5] [6] | 무제한·정기 GPU 무료 제공이 아니라 초기 크레딧. Pricing은 최대 30크레딧/약 50시간, Quick Start는 최대 80시간으로 서로 다름 | Free 플랜 GPU는 Studio당 1개, 동시 GPU 2개 안내. A100/H100/H200 세션은 4시간 제한. 실제 GPU 선택지는 계정·시점별 | 계정·전화 인증이 요구될 수 있음. 시작 시 카드는 필요 없다는 안내와, 추가 크레딧의 카드 조건이 함께 존재. 미사용 크레딧 12개월 후 만료 안내 | **제한적** — 크레딧이 한정되고 일부 GPU 세션은 4시간. 저장·재개 권장 |
| **Hugging Face Spaces ZeroGPU** [7] [8] | 반복 일일 할당량: 무료 로그인 계정 하루 5분, 비로그인 2분. 무료 GPU 크레딧은 아님 | 요청형 공유 GPU. 기본 함수 한도 60초, xlarge는 할당량 2배 소모. ZeroGPU는 Gradio SDK용 | 자체 ZeroGPU Space 호스팅에는 이메일 인증 및 계정 30일 초과 조건. 무료 이용의 카드·전화 필수 여부는 확인되지 않음 | **매우 낮음** — 무료 계정 하루 5분으로 긴 사전학습에 부적합 |
| **Modal** [9] [10] | Starter에 월 $30 컴퓨트 크레딧. 무료 GPU라기보다 GPU 사용료를 상쇄하는 월간 크레딧 | GPU는 초당 과금. 단일 Function 실행 시도는 최대 24시간 설정 가능. checkpoint·재시도로 장시간 작업을 나누어 재개 가능 | GPU 사용 및 신규 워크스페이스 검증에 유효한 결제수단 필요. 포함 크레딧 초과 시 청구 가능 | **조건부** — 재개형 학습은 지원하지만 $30만으로 장시간 GPU 학습은 제한적 |

## 실제로 의미하는 바

Colab과 Kaggle은 짧은 실험·노트북에 쓸 수 있지만 GPU 배정, 대기시간 또는 세션 지속을 보장하지 않습니다. ZeroGPU는 Gradio 기반의 짧은 추론·데모 함수에 맞습니다. Lightning AI는 초기 크레딧으로 GPU를 시험할 수 있지만 무료 시간 안내가 서로 일치하지 않습니다. Modal은 checkpoint 기반 재개형 실행을 지원하지만 결제수단과 크레딧 초과 요금에 주의해야 합니다. [1] [3] [5] [7] [9] [10]

확인한 공식 문서만으로 무료 장시간 사전학습을 안정적으로 보장하는 서비스는 없었습니다. 실제 사용자 계정의 크레딧 잔액, 남은 무료 시간, 카드·전화 인증, 해당 시점의 GPU 할당은 조사하지 않았습니다.

## 실행 전에 계정에서 확인할 항목

1. 각 서비스에 로그인해 현재 계정에 표시된 GPU 유형과 남은 시간·크레딧을 확인합니다. 이 보고서는 계정 상태를 조회하지 않았습니다.
2. Colab에서는 런타임 메뉴의 실제 가속기와 유료 구독 compute units를 확인합니다. 무료 GPU 시간이나 리셋 주기를 고정값으로 가정하지 않습니다. [1] [2]
3. Kaggle에서는 Notebook 설정과 잔여 주간 쿼터를 확인하고, GPU 대기 여부와 계정 인증 조건을 확인합니다. 리셋 요일·시각은 공개 자료만으로 확정되지 않았습니다. [3] [4]
4. Lightning AI에서는 지급된 크레딧, 카드·전화 인증, 선택 가능한 GPU와 session limit를 계정 화면에서 확인합니다. 50시간과 80시간 중 어느 쪽이 적용되는지 추정하지 않습니다. [5] [6]
5. Hugging Face에서는 계정별 ZeroGPU quota와 리셋을 인증 계정에서 확인하고, Space가 ZeroGPU 함수 방식과 호스팅 조건을 충족하는지 확인합니다. [7] [8]
6. Modal에서는 Billing의 잔액·결제수단과 실행할 GPU 단가·스토리지 비용을 확인합니다. 포함 크레딧을 넘는 사용은 청구될 수 있습니다. [9] [10]
7. 어느 서비스든 실행 전 저장 위치와 checkpoint 복구를 시험하며, 세션 로컬 디스크가 종료 후 보존된다고 가정하지 않습니다.

## 미확인·상충 정보

Colab 유료 요금 페이지는 로그인 화면만 확인되어 구체 가격과 compute unit 수량을 검증하지 못했습니다. Kaggle의 주간 쿼터는 일반 안내치이며 계정별 보장값과 정확한 초기화 시각은 공개 문서에서 확인되지 않았습니다. Kaggle 유휴 타임아웃 안내도 20분과 60분으로 다릅니다. Lightning AI의 무료 GPU 시간 안내는 약 50시간과 최대 80시간으로, 저장 용량도 50GB와 200GB로 공식 페이지 간 차이가 있습니다. Modal 월 크레딧 잔액·갱신일은 사용자 계정별로 확인해야 합니다. [1] [3] [4] [5] [6] [9]

## References

[1]: https://research.google.com/colaboratory/faq.html "Google Colaboratory FAQ"
[2]: https://colab.research.google.com/notebooks/pro.ipynb "Making the Most of your Colab Subscription"
[3]: https://www.kaggle.com/docs/notebooks "Kaggle Notebooks Documentation"
[4]: https://www.kaggle.com/docs/efficient-gpu-usage "Kaggle Efficient GPU Usage Tips"
[5]: https://lightning.ai/pricing "Lightning AI Pricing"
[6]: https://lightning.ai/docs/platform/build/ai-studio "Lightning AI Studio documentation"
[7]: https://huggingface.co/docs/hub/spaces-zerogpu "Hugging Face Spaces ZeroGPU documentation"
[8]: https://huggingface.co/pricing "Hugging Face Pricing"
[9]: https://modal.com/pricing "Modal Pricing"
[10]: https://modal.com/docs/guide/billing "Modal Billing documentation"
