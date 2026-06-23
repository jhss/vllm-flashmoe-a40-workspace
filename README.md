# vLLM DeepEP HT MoE EP Fast-Path 최적화 (A100 SXM)

2×A100-SXM 환경에서 vLLM의 **DeepEP High-Throughput MoE Expert-Parallel forward 경로**를
profiling 기반으로 재구성하여, synthetic Qwen3-like MoE forward에서
**paired latency 14.2~15.5% 개선**을 달성한 실험 기록이다.

> 한 줄 요약: 새 GEMM 커널을 만든 게 아니라, **post-routing workload를 프로파일링해 이 shape에
> 맞는 `BLOCK_M=64`를 찾아 W13/W2에 고정 적용**(runtime routing-aware selector가 아니라
> profile-guided, shape-specific 선택)한 것이 개선의 큰 덩어리이고, 그 위에 **non-local
> invalid route 제거**와 **fixed-capacity receive + raw local-expert-id alignment**를 얹은,
> A100-specific DeepEP HT MoE fast-path 재구성이다.

---

## TL;DR 결과

같은 세션 paired 측정 (2×A100-SXM, BF16, DeepEP HT, top-k=8, 128 experts, hidden=2048,
intermediate=768, 3 seed × 4 cycle = 12 paired run/조건):

| tokens | baseline (critical median) | final | 개선 | paired wins |
|---:|---:|---:|---:|:---:|
| 320 | 1728.0 μs | 1461.3 μs | **−15.51%** | 12/12 |
| 448 | 1748.7 μs | 1502.4 μs | **−14.22%** | 12/12 |

- 두 token size 모두 **12/12 paired 승리** + 모든 seed-level median에서 baseline을 이김
  (3 routing seed × 4 cycle; 같은 seed의 반복은 상관되므로 독립 12회 시행은 아님)
- 이 측정에서는 run-level IQR도 86.5 → 21.1 μs(320t)로 감소
- correctness: 모든 케이스 `assert_close` 통과 (rtol=1.6e-2, atol=1.0e-2),
  max_abs = 0.00195312, relative L2 0.0032~0.0038

> 개선율 계산: 각 run은 warmup 20회 후 100 forward의 CUDA-event 평균을 rank별로 재고,
> 두 rank 중 느린 쪽을 critical path로 쓴다. 표의 latency는 12개 run-level 값의 median이며,
> 개선율은 `median(paired delta) / median(baseline)`이라 표의 median을 직접 나눈 값과 약간 다르다.

원본 데이터: [`vllm/benchmarks/results/deepep_ht_final_cumulative_ablation_20260621_3seed_summary.md`](vllm/benchmarks/results/deepep_ht_final_cumulative_ablation_20260621_3seed_summary.md)

---

## 범위와 환경 (먼저 명시)

신뢰도를 위해 일반화 범위를 처음부터 제한해서 읽어야 한다.

- **하드웨어**: 2×A100-SXM4-80GB (SM80), `NCCL_P2P_DISABLE=0`
- **워크로드**: synthetic Qwen3-like MoE forward (실모델/실서빙 아님), BF16, no quant, no LoRA
- **shape**: hidden=2048, intermediate=768, experts=128, top-k=8, tokens ∈ {320, 448}
- **backend**: DeepEP High-Throughput (`deepep_high_throughput`), `VLLM_DEEPEP_HT_NUM_SMS=24`
- **측정**: warmup=20, iters=100, critical-path(느린 rank) median, paired same-session delta

다음은 **주장하지 않는다**:
- ❌ "vLLM 전체를 15% 개선" — MoE forward 한 경로의 microbench다
- ❌ "새 MoE GEMM 커널 개발" — 기존 Triton fused_moe_kernel을 재사용
- ❌ "DeepEP 통신량 자체 감소" — receiver/스케줄링 overhead를 줄인 것
- ❌ "모든 GPU/model에서 일반화" — 위 단일 shape에서 검증

---

## 무엇을 했나 (최종 fast path 구성)

DeepEP HT는 token을 expert 소유 rank로 dispatch하지만, receiver가 받은 데이터를
기존 vLLM expert-kernel 인터페이스에 맞추는 과정에서 overhead가 발생한다. 이를
다음으로 재구성했다.

1. **invalid route 제거** (`VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS`)
   top-k pair의 절반가량이 다른 rank 소유 expert(non-local)라 invalid block으로
   남아 불필요한 GEMM schedule을 만든다 (route stats: valid 2570 / invalid 2534 @320t).
   이를 alignment 단계에서 건너뛴다.
2. **W13/W2 tile 공동 선택** (`VLLM_MOE_TRITON_W1/W2_BLOCK_SIZE_M_OVERRIDE=64`)
   작은 expert workload에 맞춰 `BLOCK_SIZE_M`을 128→64로. 두 GEMM이 같은 BLOCK_M을
   쓰므로 `moe_align_block_size` schedule을 **한 번만 만들어 W13/W2가 공유**.
3. **fixed-capacity receive** (`VLLM_DEEPEP_HT_FIXED_CAPACITY_DISPATCH`)
   CPU metadata dependency를 줄이는 고정 용량 receive.
4. **raw local-expert-id alignment** (`VLLM_DEEPEP_HT_FIXED_CAPACITY_RAW_LOCAL_IDS`)
   DeepEP가 준 local id/`-1`을 global id로 되돌리는 remap을 없애고, CUDA alignment에서
   raw local id와 `-1` invalid slot을 직접 처리.

핵심 systems insight는 **이 단계들이 독립적이지 않다**는 점이다 (아래 ablation 참고).

---

## 누적 ablation (stepwise)

| step | 320t | 448t | 설명 |
|---|---:|---:|---|
| Original → Compute | −12.41% | −10.34% | invalid filtering + BLOCK_M=64 |
| Compute → Fixed remap | **+2.12%** | **+2.30%** | fixed-capacity를 remap과 같이 쓰면 **오히려 회귀** |
| Fixed remap → Raw local | −6.17% | −6.47% | remap 제거(raw local alignment) → 회복 + 추가 이득 |
| **Original → Final** | **−15.51%** | **−14.22%** | 누적 |

→ 두 가지 핵심 관찰:
- 전체 이득의 큰 덩어리(약 10~12%p)는 **compute(filtering+tiling)** 쪽이고, receiver
  fast-path(fixed+raw)의 순기여는 약 3.6~4.3%p다. 즉 **주인공은 receiver가 아니라
  post-routing workload-aware tiling/scheduling**이다.
- fixed-capacity receive는 **단독으로는 손해**(+2.12%/+2.30%)이고, raw local-id
  alignment까지 연결해야 net win이 된다 — "절반만 적용하면 안 되는" 비자명한 의존성.

### Compute 버킷 분해 (별도 세션, commit `001b3cb`)

위 "Compute"는 두 요인의 묶음이라 별도 same-session ablation으로 분리했다
([`deepep_ht_block_m_combined_ablation_20260621_3seed_summary.md`](vllm/benchmarks/results/deepep_ht_block_m_combined_ablation_20260621_3seed_summary.md)):

| setting | 320t | 448t | wins |
|---|---:|---:|:---:|
| filtering only (ignore-invalid, BLOCK_M 기본) | −2.65% | −3.34% | 9/9, 8/9 |
| + BLOCK_M=64 (final) | −12.87% | −10.52% | 9/9 |

→ **compute 이득의 대부분은 `BLOCK_M=64`(tile config)에서 나오고, invalid filtering의
순기여는 약 2.6~3.3%다.** 즉 이 프로젝트의 가장 큰 단일 레버는 expert-shape-aware한
tile 선택이며, 이를 숨기지 않고 명시한다. (절댓값이 세션마다 달라 이 %는 final ablation
표에 그대로 합산하지 말 것 — 같은 세션 내 분해로만 해석.)

> 아직 안 한 것: `ignore-invalid OFF + BLOCK_M=64` 셀. 2×2 factorial의 마지막 칸으로,
> BLOCK_M 단독 효과와 filtering×BLOCK_M interaction을 완전히 분리하려면 이 셀이 필요하다.
> 현재의 조건부 주장("filtering 적용 후 BLOCK_M=64가 추가로 ~7~10%p")에는 위 데이터로 충분.

---

## Correctness

최적화 경로를 같은 2-rank layer instance에서 original 경로와 비교 (worst across ranks):

| case | max_abs | rel_l2 | assert_close |
|---|---:|---:|:---:|
| tokens=320 balanced | 0.00195312 | 0.00375 | ✅ |
| tokens=448 balanced | 0.00195312 | 0.00376 | ✅ |
| rank 128/512 balanced | 0.00195312 | 0.00375 | ✅ |
| rank 128/512 target_rank=0 | 0.00195312 | 0.00322 | ✅ |
| rank 128/512 target_rank=1 | 0.00195312 | 0.00322 | ✅ |

- tolerance: `rtol=1.6e-2, atol=1.0e-2` (benchmark 기본값). 관측 max_abs 0.00195312,
  rel_l2 0.0032~0.0038로 tolerance 안에서 통과
- balanced / 비대칭 rank token / target-rank 집중 routing 등 분포 edge case 커버
- alignment regression: `pytest tests/kernels/moe/test_moe_align_block_size.py` → **477 passed**

데이터: [`deepep_ht_final_correctness_20260621_summary.md`](vllm/benchmarks/results/deepep_ht_final_correctness_20260621_summary.md)

---

## 정직한 한계 (검증되지 않은 부분)

포트폴리오로서 신뢰도를 위해 명시한다.

1. **가장 큰 레버는 tile config(`BLOCK_M=64`)다.** compute 이득의 대부분이 여기서 나오고
   (위 분해 표 참고), invalid filtering은 ~2.6~3.3%로 작다. 이는 "novel kernel"이 아니라
   workload-aware한 tile 선택이므로, 그 한계를 숨기지 않는다. 남은 정밀도 구멍은
   `ignore-OFF + BLOCK_M=64` 셀 미측정(2×2 factorial의 마지막 칸) — BLOCK_M 단독효과와
   interaction의 완전 분리는 아직.
2. **fixed-capacity 일반화 caveat.** capacity는 routing을 관측해 정한 oracle 값이 아니라
   DP 전체 token 수(`tokens × world_size` = 640/896)에서 계산한 **안전한 worst-case 상한**이다.
   다만 본 EP=2, top-k=8 workload는 실제 receive(≈638/892)가 상한에 거의 붙는 **고활용(~99%)**
   조건이라 fixed-capacity에 유리했다. EP가 커지거나 top-k가 작거나 routing이 치우치면
   활용률이 떨어져 메모리·padding 비용이 늘 수 있고, 이 영역은 검증하지 않았다.
3. **synthetic 단일 shape.** 실모델 trace, decode 분포, 다양한 expert 수/world size에서는
   재검증이 필요하다.

---

## 폐기/early attempts (negative results)

방향을 좁히기 위해 시도하고 폐기한 것들. 이 negative result가 최종 경로의 근거다.

| 시도 | 결과 | 결론 |
|---|---|---|
| Triton in-place top-k remap | receiver remap 0.089→0.049 ms, full fwd 1.414→1.386 ms | 작은 승리, 단독으론 약함 |
| AG/RS in-place combine (`out=`) | finalize 0.120→0.101 ms, full fwd 노이즈 | alloc/copy는 병목 아님 (NCCL latency 지배) |
| local-expert-id generic path | 거의 성능 중립 | ID 공간 축소만으로는 schedule/GEMM 비용이 안 줄어듦 |
| DeepEP HT direct-assignment builder | **회귀** (모든 token size에서 baseline보다 느림; experts 구간 ~254us↑, direct+ignore도 ~71us 느림) | 병목은 GEMM이 아니라 prefix-sum·allocation·다수의 작은 CUDA op/launch로 구성된 **builder 고정 비용** |
| W2 atomic epilogue reduce (BF16 direct) | max_abs≈0.295, 정확도 실패 | 폐기. FP32 workspace는 정확하나 더 느림 |
| A100 BF16 specialized kernel-body | 정확하나 더 안 빠름 | generic kernel의 branch가 이미 constexpr 제거됨 |

전체 서술: [`vllm/benchmarks/results/a100_sxm_moe_ep_code_changes.md`](vllm/benchmarks/results/a100_sxm_moe_ep_code_changes.md)

---

## 재현

```bash
# baseline (original) — 이전 shell 환경을 물려받지 않도록 관련 flag를 명시적으로 0
NCCL_P2P_DISABLE=0 VLLM_DEEPEP_HT_NUM_SMS=24 \
  VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS=0 \
  VLLM_MOE_TRITON_W1_BLOCK_SIZE_M_OVERRIDE=0 \
  VLLM_MOE_TRITON_W2_BLOCK_SIZE_M_OVERRIDE=0 \
  VLLM_DEEPEP_HT_FIXED_CAPACITY_DISPATCH=0 \
  python vllm/benchmarks/kernels/benchmark_moe_ep_a40.py \
  --world-size 2 --backend deepep_high_throughput \
  --tokens 320 --hidden-size 2048 --intermediate-size 768 \
  --num-experts 128 --top-k 8 --warmup 20 --iters 100 \
  --seed 7 --rank-distinct-inputs --input-seed-base 1007 --csv

# final fast path
#  주의: IGNORE_INVALID_EXPERTS_MIN_TOKENS 기본값은 1024다. 본 워크로드의 received M은
#  약 638/892라, MIN_TOKENS=0을 주지 않으면 IGNORE_INVALID_EXPERTS=1이어도 filtering이
#  비활성화되어 결과가 재현되지 않는다.
NCCL_P2P_DISABLE=0 VLLM_DEEPEP_HT_NUM_SMS=24 \
  VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS=1 \
  VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS_MIN_TOKENS=0 \
  VLLM_MOE_TRITON_W1_BLOCK_SIZE_M_OVERRIDE=64 \
  VLLM_MOE_TRITON_W2_BLOCK_SIZE_M_OVERRIDE=64 \
  VLLM_DEEPEP_HT_FIXED_CAPACITY_DISPATCH=1 \
  VLLM_DEEPEP_HT_FIXED_CAPACITY_NUM_WORST_TOKENS=640 \
  VLLM_DEEPEP_HT_FIXED_CAPACITY_RAW_LOCAL_IDS=1 \
  python vllm/benchmarks/kernels/benchmark_moe_ep_a40.py \
  --world-size 2 --backend deepep_high_throughput \
  --tokens 320 --hidden-size 2048 --intermediate-size 768 \
  --num-experts 128 --top-k 8 --warmup 20 --iters 100 \
  --seed 7 --rank-distinct-inputs --input-seed-base 1007 --csv
# tokens=448 은 --tokens 448 + VLLM_DEEPEP_HT_FIXED_CAPACITY_NUM_WORST_TOKENS=896 으로 실행
```

전체 ablation 명령: [`...cumulative_ablation_20260621_3seed_commands.log`](vllm/benchmarks/results/deepep_ht_final_cumulative_ablation_20260621_3seed_commands.log)

> 환경 세팅은 [`vllm/AGENTS.md`](vllm/AGENTS.md)의 `uv` 기반 절차를 따른다 (system python/pip 금지).
> (루트 `AGENTS.md`는 워크스페이스 운영 메모이고, 빌드/테스트 절차는 `vllm/AGENTS.md`에 있다.)

---

## 저장소 구조

```text
vllm/                                       # vLLM fork (실험 본체)
  vllm/envs.py                              # opt-in feature flag 정의
  vllm/model_executor/layers/fused_moe/
    prepare_finalize/deepep_ht.py           # DeepEP HT receiver / fixed-capacity / raw-local
    moe_align_block_size.py                 # expert assignment schedule (Python)
    experts/triton_moe.py                   # invalid filtering / BLOCK_M override / assignment 재사용
  csrc/libtorch_stable/moe/moe_align_sum_kernels.cu  # CUDA align: raw local id / -1 직접 skip
  benchmarks/kernels/benchmark_moe_ep_a40.py  # 메인 paired benchmark + section profiler
  benchmarks/results/                       # raw CSV / JSON / summary (모든 측정 근거)
  docs/design/multi_gpu_kernels_ko.md       # vLLM multi-GPU 실행 구조 top-down 설명
architecture.html / architecture.md         # ⚠ 초기 버전 다이어그램 (최종 경로와 다름 — 갱신/삭제 예정)
```

### 주요 코드 변경

kernel-level 기여가 드러나도록 핵심 파일을 정리한다.

| 기여 | 주요 파일 |
|---|---|
| invalid filtering / `BLOCK_M` override / assignment 재사용 | `vllm/.../fused_moe/experts/triton_moe.py` |
| fixed-capacity receive / raw-local contract | `vllm/.../fused_moe/prepare_finalize/deepep_ht.py` |
| raw local id / `-1` 직접 skip (CUDA align) | `vllm/csrc/libtorch_stable/moe/moe_align_sum_kernels.cu` |
| alignment regression test (477 passed) | `vllm/tests/kernels/moe/test_moe_align_block_size.py` |
| paired benchmark + section profiler | `vllm/benchmarks/kernels/benchmark_moe_ep_a40.py` |

### 주요 feature flag

opt-in이 대부분이지만 전부 off는 아니다 (아래 default 열 참고). 좁은 조건에서만 효과가 있다.

| flag | default | 역할 |
|---|---:|---|
| `VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS` | 0 | invalid(non-local) route를 GEMM schedule에서 제거 |
| `VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS_MIN_TOKENS` | 1024 | filtering이 켜지는 received-M 임계값 (본 워크로드선 0으로 낮춰야 활성) |
| `VLLM_MOE_TRITON_W1_BLOCK_SIZE_M_OVERRIDE` | 0 | W13 tile `BLOCK_SIZE_M` override |
| `VLLM_MOE_TRITON_W2_BLOCK_SIZE_M_OVERRIDE` | 0 | W2 tile `BLOCK_SIZE_M` override (W13과 같으면 schedule 공유) |
| `VLLM_DEEPEP_HT_FIXED_CAPACITY_DISPATCH` | 0 | 고정 용량 receive (CPU metadata dependency 축소) |
| `VLLM_DEEPEP_HT_FIXED_CAPACITY_RAW_LOCAL_IDS` | **1** | fixed-capacity 활성 시 raw-local submode (상위 DISPATCH가 off면 무효) |
| `VLLM_DEEPEP_HT_FIXED_CAPACITY_NUM_WORST_TOKENS` | 0 | fixed-capacity 용량 (0이면 자동, 본 실험은 640/896 명시) |
| `VLLM_DEEPEP_HT_TRITON_TOPK_REMAP` | 0 | (early) Triton in-place top-k remap |
| `VLLM_MOE_TRITON_W1_A100_TUNED_CONFIG` / `..._W2_...` | 0 | (early) A100 SM80 전용 meta config |

---

## 다음 단계

1. **2×2 factorial 완성** — `ignore-OFF + BLOCK_M=64` 셀을 추가해 BLOCK_M 단독효과와
   filtering×BLOCK_M interaction까지 완전 분리 (filtering vs BLOCK_M 분해는 `001b3cb`에서 완료)
2. **fixed-capacity over-provisioning 비용 측정** — 높은 EP degree / 낮은 top-k에서 활용률이
   떨어질 때의 메모리·padding 비용
3. **Nsight Systems/Compute** — prefill/decode timeline, tensor-core util, stall reason
   (현 컨테이너는 `ERR_NVGPUCTRPERM`으로 HW counter 차단)
4. **prefill 전용 fusion** — GEMM2 epilogue + top-k reduce/`moe_sum` fuse
