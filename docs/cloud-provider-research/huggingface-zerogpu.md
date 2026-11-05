# Hugging Face Spaces ZeroGPU

**확인일: 2026-11-05** · 대상은 Hugging Face Spaces의 ZeroGPU이며, 계정별 현재 잔여량이나 실제 GPU 배정을 확인한 것은 아니다.

## 무료 GPU와 할당량

ZeroGPU는 공유 GPU를 요청 시 할당하고 함수 실행 뒤 반환하는 Spaces 기능이다. 기존 ZeroGPU Space를 쓰는 것은 모든 사용자에게 무료라고 공식 문서는 설명한다. 이는 1회 체험 크레딧이 아니라 계정 등급별로 되풀이되는 일일 GPU 사용량이다. 문서상 비로그인 사용자는 하루 2분, 무료 계정은 5분을 받고, PRO 및 Team 구성원은 40분, Enterprise 구성원은 60분을 받는다. 무료 계정은 중간 우선순위, 비로그인 사용자는 낮은 우선순위다. 잔여 할당량이 대기열 우선순위에도 영향을 준다. 초기 사용 후 정확히 24시간이 지나면 할당량이 재설정된다. [1]

무료 개인 계정이 직접 호스팅할 수 있는 ZeroGPU Space는 최대 2개이며, 계정은 이메일 인증을 마치고 30일 이상 경과한 양호한 상태여야 한다. PRO 개인 계정은 최대 10개를 호스팅할 수 있다. 기존 Space를 사용하는 무료 접근과 자신의 Space를 호스팅하는 조건은 서로 다르다. [1]

PRO·Team·Enterprise는 일일 기본량을 넘긴 뒤 선불 크레딧으로 이용을 계속할 수 있으며, 공식 ZeroGPU 문서는 GPU 시간 10분당 $1이라고 안내한다. PRO 플랜은 가격 페이지에 월 $9로 표시되며 ZeroGPU 사용량 8배와 높은 대기열 우선순위를 포함한다. PRO 구독료와 사용량 기반 컴퓨트 비용은 별도다. [1] [2] [6]

## GPU와 실행 제약

ZeroGPU가 문서화한 크기는 두 가지다. 기본 `large`는 NVIDIA RTX PRO 6000 Blackwell의 절반 구성으로 VRAM 48GB이며, `xlarge`는 전체 GPU로 VRAM 96GB다. `xlarge`는 일일 할당량을 2배 소모한다. 이는 일반 Spaces 유료 GPU 목록에 있는 A100, L40S 등 선택지와 구별되는 ZeroGPU 전용 구성이다. [1] [3]

ZeroGPU는 Space 안에서 `@spaces.GPU`로 표시한 함수가 호출될 때 GPU를 할당하고 함수가 끝나면 반환하는 방식이다. 기본 함수 GPU 실행 한도는 60초이며, 더 긴 실행은 사용자 지정 `duration`을 지정할 수 있다. 문서는 120초 예시를 제시하지만, 확인한 페이지에는 사용자 지정 한도의 절대 최댓값이 명시되어 있지 않다. ZeroGPU는 Gradio SDK 전용이라고 문서화되어 있으며, 공유 대기열 때문에 GPU 접근이 항상 즉시 가능하다고 보장되지는 않는다. [1]

## 가입·결제

ZeroGPU 호스팅 자격에 필요한 사항으로 공식 문서가 명시하는 것은 이메일 인증과 계정 생성 후 30일 경과다. 이 자료에서 신규 가입 시 전화번호가 필수인지 여부는 확인되지 않았다. 무료 일일 할당량 자체에 결제카드가 필요한지는 문서에서 명시적으로 규정하지 않는다. 다만 PRO 구독은 신용카드로 결제하며, 유료 컴퓨트 사용은 신용카드 결제 또는 조직의 클라우드 마켓플레이스 청구로 처리한다고 결제 문서가 설명한다. [1] [2] [6]

## 저장과 산출물

Space의 기본 로컬 디스크는 50GB이며 영구 저장소가 아니다. Space가 재시작되거나 중지되면 해당 임시 디스크 내용은 사라진다. 지속 보존에는 Storage Bucket을 볼륨으로 연결해 기록해야 한다. 따라서 작업 산출물이 자동으로 보존된다고 가정하지 말고, 버킷 등 지속 저장 위치로 명시적으로 저장해야 한다. Bucket의 구체적인 가격은 이 확인 범위에서 판단하지 않았다. [4] [5]

## 장시간 사전학습 적합성

**무료 ZeroGPU는 장시간 또는 연속 사전학습에 실용적이지 않다.** 무료 계정의 GPU 총량은 하루 5분이고, 기본 GPU 함수 실행은 60초 단위 한도이며, Space는 요청 기반 공유 GPU다. 따라서 ZeroGPU는 짧은 추론·데모 호출에 맞는 구조로 보는 편이 타당하다. 공식 문서는 긴 사전학습 작업을 보장하거나 권장한다고 확인되지 않았고, 사용자 지정 함수 시간의 절대 상한도 해당 페이지에서 찾을 수 없었다. 이는 문서화된 제한을 바탕으로 한 적합성 판단이지, 모든 학습 실행이 기술적으로 불가능하다는 의미는 아니다. [1]

## 계정별 미확인 사항

- 특정 사용자의 현재 일일 잔여량, 리셋 시각, 크레딧 잔액은 계정 인증 정보로 조회하지 않았으므로 알 수 없다. 공식 API 문서는 인증된 사용자의 ZeroGPU 잔여량 및 리셋 시각을 조회하는 기능이 있다고 설명한다. [7]
- 특정 계정이 실제로 어떤 GPU를 배정받을지, 요청 당시 대기시간이나 성공 여부는 확인하지 않았다. 문서화된 GPU 크기는 가능한 사양이지 이 계정의 예약·배정 보장이 아니다. [1]
- 전화번호의 필수 여부 및 무료 계정 가입 과정에서 카드 정보를 요구하는지에 관한 공식 설명은 확인한 자료에 없었다.

## References

[1]: https://huggingface.co/docs/hub/spaces-zerogpu "Spaces ZeroGPU: Dynamic GPU Allocation for Spaces"
[2]: https://huggingface.co/pricing "Hugging Face Pricing"
[3]: https://huggingface.co/docs/hub/spaces-gpus "Using GPU Spaces"
[4]: https://huggingface.co/docs/hub/spaces-storage "Disk usage on Spaces"
[5]: https://huggingface.co/docs/hub/spaces-overview "Spaces Overview"
[6]: https://huggingface.co/docs/hub/billing "Billing"
[7]: https://huggingface.co/docs/huggingface_hub/package_reference/space_runtime "Managing your Space runtime"
