# MiniCoder의 BitNet 스타일 QAT 구현

MiniCoder에는 `use_bitnet: true`로 활성화하는 정확성 우선의 PyTorch 참조 경로를 추가했다. 구현은 absmean 가중치 양자화를 적용해 forward 경로의 선형층 가중치를 `{-1, 0, +1}`로 ternary화하고, token별 symmetric activation quantization을 수행한다. Quantization-aware training에는 identity straight-through estimator(STE)를 쓴다. BitNet b1.58 원 논문은 평균 절댓값으로 가중치를 스케일한 뒤 `round` 및 `clip`하여 ternary 값을 얻는 방법과 8-bit activation quantization을 설명한다.[1]

BitNet b1.58 2B4T 기술 보고서는 표준 선형층을 BitLinear로 바꾸고, forward 동안 ternary weight와 per-token absmax 8-bit activation을 쓰며, SubLN 및 squared-ReLU FFN을 사용한다고 기술한다.[2] 이 구현은 BitLinear, W1.58A8과 유사한 활성화 양자화, squared-ReLU FFN을 코드로 제공한다. 다만 기존 MiniCoder 구조와 호환을 유지하기 위해 RMSNorm·RoPE·GQA를 유지하며 SubLN 구조는 구현하지 않는다. 따라서 이는 작은 규모의 연구/기능 검증용 BitNet 스타일 QAT 구현이지, Microsoft의 공식 BitNet 학습 코드나 BitNet-b1.58-2B-4T 재현으로 간주하면 안 된다.

## 정밀도와 저장 공간

학습 가능한 master weights, gradients, optimizer state는 일반 PyTorch 부동소수점 텐서로 유지된다. Forward에 쓰이는 quantized weight는 ternary 값에 layer scale을 곱한 부동소수점 텐서로 복원된 뒤 일반 `F.linear`를 통과한다. Activation 역시 8-bit 격자로 양자화한 다음 부동소수점 값으로 복원한다. 이 코드는 packed 1.58-bit weight 저장, 정수 GEMM, 전용 CUDA 커널 또는 메모리 절감을 구현하지 않는다. 체크포인트는 기존 PyTorch state-dict 포맷을 사용하므로 float master weights를 저장하며, 1.58-bit가 실제 checkpoint가 파라미터당 1.58비트라는 뜻은 아니다. 공식 구현 저장소는 MIT 라이선스이며 주로 최적화된 추론 프레임워크를 제공한다.[3] MiniCoder는 공식 구현 코드를 복사하지 않고 논문에 기술된 quantization 개념을 참조한 독립적인 PyTorch 참조 코드를 포함한다.

## 실행 및 검증

```bash
python -m unittest tests.test_bitnet -v
python -m training.trainer --config config/model_bitnet_tiny.yaml --data-dir data/processed --run-dir runs/bitnet-tiny --max-steps 5
```

두 번째 명령에는 `data/processed/train.bin`과 호환되는 `val.bin`이 필요하다. 빠른 synthetic end-to-end 데이터→학습→복구 검증은 다음을 실행한다.

```bash
python -m training.smoke
```

유닛 테스트는 ternary 값 범위, activation의 zero/NaN 안정성, STE gradient, BitLinear backward, 작은 고정 synthetic batch의 loss 감소, 체크포인트 출력의 roundtrip 동일성을 검사한다. 이 테스트는 모델의 코드 생성 능력이나 대규모 품질을 평가하지 않는다.

## References

[1]: https://arxiv.org/html/2402.17764 "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"
[2]: https://arxiv.org/html/2504.12285v1 "BitNet b1.58 2B4T Technical Report"
[3]: https://github.com/microsoft/BitNet "Microsoft BitNet inference framework"
