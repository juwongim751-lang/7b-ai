# Modal 선택 실행 안내

`modal_app.py`는 선택적 T4 job입니다. 설치 또는 local tests에서 호출되지 않습니다. 사용자 스스로 현재 Modal GPU 가격·storage 가격·account credit/billing conditions를 확인해야 합니다. 코드에는 단가를 고정하지 않았습니다. 실행자는 현재 확인한 **positive hourly GPU rate**와 **positive USD estimate budget**, 그리고 명시적 `--confirm-potential-cost`를 제공해야 합니다. 이 budget은 사용자가 승인하는 estimate guard이지 공급자 billing hard cap이 아닙니다.

사용자가 직접 생성·준비한 `minicoder-checkpoints` Volume에 `/checkpoints/data/{train.bin,val.bin}`과 metadata sidecars를 저장해야 합니다. Modal runner는 새 volume을 자동 생성하지 않습니다. 학습결과/checkpoints는 volume mounted directory에 저장 후 호출이 끝날 때 commit합니다. 이 프로젝트는 Modal account credit, current rate, GPU eligibility를 조회하지 않습니다.

Modal SDK 설치 및 Volume/data preparation을 마친 후, 실제 가격을 확인해 본인 rate/budget 값으로 명시적으로 시작할 수 있습니다. 아래는 **명령 형식만 보이는 예시이며 실제 단가 추천이 아닙니다**.

```bash
modal run modal_app.py \
  --max-steps 8 \
  --gpu-cost-rate-usd-per-hour 0.01 \
  --max-budget-usd 0.02 \
  --confirm-potential-cost
```

예시의 `0.01`, `0.02`를 실제 Modal price로 사용하지 마세요. 이 작업에서는 위 명령을 실행하지 않았습니다. 계정, GPU allocation, credit balance, billing profile 또는 persistent Volume을 검증하지 않았습니다. Storage, tax, retries, startup overhead 및 billing granularity는 local estimate에 포함되지 않습니다.
