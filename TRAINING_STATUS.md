# MiniCoder 학습 상태

**마지막 검증:** 2026-11-05 · **상태: 실제 7B·100B-token 학습 미완료.** 실행 환경은 CPU-only이며 실제 cloud GPU 계정 작업을 시작하지 않았습니다.

## 현재 실행 환경 및 실제 학습

- Python 3.12.3, PyTorch 2.15.0+cu132, CUDA runtime 13.2; `torch.cuda.is_available() == false`. GPU 이름·VRAM은 **N/A**입니다.
- Synthetic CPU smoke test에서 tokenizer, model forward/backward, 유한 loss, validation, loss 감소, checkpoint와 resume(8→10 update)를 통과했습니다.
- 기본 smoke Tiny는 361,088 parameters이며 목표 Tiny 약 10M preset과 다른 구성입니다.
- 10M Tiny는 10,054,272 parameters. 짧은 synthetic/toy data forward/backward, validation, checkpoint를 수행했으나 7B 품질 학습은 아닙니다.
- 가장 최근 10M Tiny CPU microbenchmark: batch 1 × sequence 16, 5 update/80 measured tokens, 0.440046초, **181.8 tokens/s**. GPU·VRAM·bandwidth는 측정되지 않았습니다. 상세: [`GPU_BENCHMARK_REPORT.md`](GPU_BENCHMARK_REPORT.md), JSON [`reports/gpu_benchmark.json`](reports/gpu_benchmark.json).

## Architecture preset 규모 및 추정 메모리

실제 parameter count는 config에서 계산했습니다. Memory 추정은 실측이 아니라 BF16, batch 1, sequence 512, activation checkpointing을 적용한 estimator heuristic입니다.

| Preset | Parameter count | BF16 training estimator | 실제 training |
|---|---:|---:|---|
| Tiny smoke | 361,088 | — | 짧은 합성 CPU smoke |
| Tiny 10M | 10,054,272 | 0.211 GiB | 짧은 CPU toy run |
| 100M | 98,841,088 | 1.939 GiB | 미학습 |
| 300M | 304,773,888 | 5.808 GiB | 미학습 |
| 1B | 1,003,554,816 | 18.886 GiB | 미학습 |
| 3B | 2,952,416,256 | 55.333 GiB | 미학습 |
| 7B | 7,241,732,096 | 135.433 GiB | 미학습 |

Training memory는 hardware requirement 보증이 아니며 GPU RAM, allocator, activation/workspace, distributed sharding, utilization 등을 실측하지 않았습니다. BF16 weight-only estimate for 7B는 13.489 GiB이며 이는 optimizer/gradient/activation을 포함하지 않습니다.

단순 100B-token ÷ 현재 CPU 181.8 tokens/s 산술은 약 17.43년이지만, 이는 10M 5-step probe의 외삽일 뿐 실제 7B 처리량·GPU 실행시간/완료 예상이 아닙니다.

## 최종 자동 테스트

```text
python -m unittest discover -s tests -v     128 tests: OK
python -m tests.integration_test             6/6 scenarios: PASS
python -m compileall -q .                    PASS
node --check dashboard/frontend/app.js       PASS
Colab/Kaggle JSON + code-cell syntax         PASS (7 cells each; notebooks were not provider-executed)
```

세부: [`TEST_REPORT.md`](TEST_REPORT.md), 전체 작업의 상태 분류와 제한: [`FINAL_IMPLEMENTATION_REPORT.md`](FINAL_IMPLEMENTATION_REPORT.md).

## 실제로 하지 않은 작업

Colab/Kaggle/Lightning/ZeroGPU/Modal live account submit·GPU runtime, live quota/credit/billing 조회, cloud-to-cloud restore, GPU benchmark, 100M–7B training, 100B-token training은 미실행입니다. Kaggle 공식 CLI client는 mocked subprocess로 테스트했으며 실제 커널을 제출하지 않았습니다. Dashboard가 runtime provider inspection과 local training control을 제공하지만 모든 provider에 대한 자동 원격 job queue나 failover는 아닙니다. 무료 자원, 세션 시간, 인증 조건은 계정·시점별로 직접 확인해야 합니다.
