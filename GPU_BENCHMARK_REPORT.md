# GPU·CPU 벤치마크와 메모리 추정 보고서

**기준일: 2026-11-05.** 이 실행 환경은 CUDA device가 없는 CPU sandbox다. 이름에 GPU가 들어간 legacy report도 GPU 성능 측정으로 오해하지 않도록 아래 결과를 기준으로 삼는다.

## 실제 Tiny 측정

실행 명령:

```bash
python -m training.benchmark --config config/model_tiny_10m.yaml \
  --output-dir reports --steps 5 --batch-size 1 --seq-len 16
```

| 항목 | 측정값 |
|---|---:|
| 모델 config / 실제 parameter count | `model_tiny_10m.yaml` / 10,054,272 |
| device | CPU (`torch.cuda.is_available() == False`) |
| batch × sequence / updates | 1 × 16 / 5 |
| tokens / elapsed | 80 / 0.440046 s |
| 실측 처리량 | **181.8 tokens/s** |
| 마지막 synthetic loss | 6.02891 |
| CUDA GPU 이름·VRAM·peak allocation | **N/A — 측정할 GPU 없음** |
| 메모리 bandwidth·provider session quota | 측정 안 함 / 알 수 없음 |
| remote submit | 실행 안 함 |

기계 판독 결과는 [`reports/gpu_benchmark.json`](reports/gpu_benchmark.json)입니다. 이 값은 짧은 10M CPU forward/backward probe로, provider/GPU·7B throughput 또는 언어 품질을 추정하지 않습니다.

## 구성 크기와 거친 memory estimator

파라미터 수는 YAML 구조에서 계산했습니다. memory 열은 **코드의 추정값**이지 실제 allocation/fit test가 아닙니다. Training 열은 BF16 config, microbatch 1, sequence 512, activation checkpointing on인 estimator 설정입니다. 실제 요구량은 model shape, CUDA kernels, allocator fragmentation, activation/workspaces, optimizer와 sharding 등에 따라 달라집니다.

| Preset | Actual parameters | BF16 weight-only estimate | BF16 training estimate (microbatch 1, seq 512, activation checkpointing) |
|---|---:|---:|---:|
| Tiny 10M | 10,054,272 | 0.019 GiB | 0.211 GiB |
| 100M | 98,841,088 | 0.184 GiB | 1.939 GiB |
| 300M | 304,773,888 | 0.568 GiB | 5.808 GiB |
| 1B | 1,003,554,816 | 1.869 GiB | 18.886 GiB |
| 3B | 2,952,416,256 | 5.499 GiB | 55.333 GiB |
| 7B | 7,241,732,096 | 13.489 GiB | 135.433 GiB |

7B BF16 weight-only is about 13.49 GiB, but it excludes gradient, optimizer and activation memory. 7B training estimate uses the project's conservative non-sharded AdamW heuristic; it is not measured on hardware. Multi-GPU/device VRAM is never pooled by the scheduler. FSDP/ZeRO or other partitioned paths need independent tests and estimates; the table does not imply such a run is ready.

## Throughput arithmetic (not a forecast)

100,000,000,000 tokens ÷ 181.8 CPU tokens/s ≈ **17.43 years**. This is a naive extrapolation from a five-step, 10M CPU probe and omits 7B FLOPs, real data loading, sequence/batch variation, optimizer/checkpoint/evaluation overhead, hardware utilization, GPU availability, failures and provider wall-time limits. It is not a 7B training timeline or completion claim.
