# A100 SXM MoE EP 코드 변경 정리

이 문서는 이번 A100 SXM MoE Expert Parallel 실험에서 어떤 코드를 왜
수정했고, 그 결과가 무엇이었는지 한글로 정리한 문서다.

핵심은 두 가지다.

1. DeepEP HT의 receiver 단계에서 `topk_ids` 변환 비용을 줄였다.
2. AG/RS finalize 단계에서 불필요한 output copy를 줄였다.

## 결과 요약

| 항목 | 전 | 후 | 의미 |
|---|---:|---:|---|
| DeepEP receiver top-k remap | 0.089 ms | 0.049 ms | Triton in-place remap으로 약 40 us 감소 |
| DeepEP HT full forward | 1.414 ms | 1.386 ms | 전체 MoE forward가 약 27 us 빨라짐 |
| AG/RS finalize section | 0.120 ms | 0.101 ms | `reduce_scatterv(out=output)`로 allocation/copy 하나 제거 |
| AG/RS full forward | 1.178 ms | 1.179 ms | 전체 forward에서는 노이즈 수준 |

실험 조건:

- GPU: 2x A100-SXM4-80GB
- MoE shape: Qwen3-like synthetic BF16
- tokens: `128`
- hidden size: `2048`
- intermediate size: `768`
- experts: `128`
- top-k: `8`
- env: `NCCL_P2P_DISABLE=0`

## 1. DeepEP top-k remap 최적화

### 왜 필요했나

DeepEP HT는 token을 각 GPU의 local expert 기준으로 dispatch한다.

예를 들어 전체 expert가 128개이고 GPU가 2개라면:

```text
GPU0: expert 0~63
GPU1: expert 64~127
```

DeepEP가 GPU1에서 처리할 token을 받으면, 그 expert id는 보통 GPU1 내부
기준인 local id로 온다. 예를 들어 global expert 70은 GPU1 안에서는 local
expert 6처럼 볼 수 있다.

그런데 vLLM의 기존 MoE expert kernel 경로는 global expert id를 기대한다.
그래서 DeepEP receiver는 local expert id를 다시 global expert id로 바꿔야
한다.

기존 코드는 대략 이런 식이었다.

```python
torch.where(
    expert_topk_ids == -1,
    invalid_expert_id,
    expert_topk_ids + rank_expert_offset,
)
```

이 작업이 DeepEP receiver 안에서 약 `0.089 ms` 걸렸다.

### 어떻게 바꿨나

수정 파일:

- `vllm/envs.py`
- `vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py`

`vllm/envs.py`에는 feature flag를 추가했다.

```text
VLLM_DEEPEP_HT_TRITON_TOPK_REMAP
```

기본값은 `0`이다. 즉 기본 동작은 기존과 같다.

이 flag를 `1`로 켜면 DeepEP HT receiver에서 `torch.where` 대신 작은
Triton kernel을 사용한다.

새로 추가한 함수는:

```python
remap_deepep_ht_topk_ids(...)
```

이고, 내부에서 조건이 맞으면 Triton kernel을 호출한다.

Triton kernel은 `expert_topk_ids` tensor를 새로 만들지 않고, 기존 tensor를
제자리에서 수정한다.

```python
values = tl.where(values == -1, invalid_expert_id, values + rank_expert_offset)
tl.store(topk_ids + offsets, values, mask=mask)
```

즉 기존 방식은:

```text
새 tensor 생성 + torch.where 결과 저장
```

새 방식은:

```text
기존 topk_ids tensor를 Triton kernel로 직접 수정
```

이다.

### 코드 흐름

DeepEP receiver 쪽에서 원래 `torch.where`를 직접 부르던 부분을:

```python
expert_topk_ids = remap_deepep_ht_topk_ids(
    expert_topk_ids,
    num_experts,
    self.rank_expert_offset,
)
```

로 바꿨다.

flag가 꺼져 있으면 기존 `torch.where` 경로로 돌아간다.

flag가 켜져 있고 tensor가 CUDA/contiguous/non-empty이면 Triton in-place
kernel을 쓴다.

### 결과

| 항목 | 기존 `torch.where` | Triton remap |
|---|---:|---:|
| receiver top-k remap | 0.089 ms | 0.049 ms |
| DeepEP dispatch receiver | 0.207 ms | 0.187 ms |
| DeepEP prepare total | 0.385 ms | 0.364 ms |
| DeepEP HT full forward | 1.414 ms | 1.386 ms |

해석:

- remap 자체는 약 `40 us` 줄었다.
- 전체 MoE forward는 약 `27 us` 줄었다.
- 작은 최적화지만, DeepEP receiver 병목이 어디인지 명확히 보여준다.

## 2. DeepEP receiver 세부 profiling 추가

### 왜 필요했나

처음에는 DeepEP prepare가 AG/RS보다 느리다는 것만 알 수 있었다.

하지만 prepare 안에도 여러 단계가 있다.

```text
dispatch submit
event wait
top-k id remap
metadata 생성
post-dispatch quantization
```

어디가 느린지 모르고 코드를 바꾸면 추측이 된다.

그래서 benchmark에 receiver 내부 단계별 타이머를 넣었다.

### 수정 파일

- `benchmarks/kernels/benchmark_moe_ep_a40.py`

추가한 옵션:

```bash
--section-profile-iters
--section-profile-warmup
--section-profile-output
```

이 옵션을 켜면 benchmark가 실제 vLLM MoE path를 monkey patch해서 구간별
시간을 JSON으로 저장한다.

### 추가한 section 이름

```text
moe_prepare_total
moe_experts_total
moe_finalize_total
deepep_dispatch_submit
deepep_dispatch_receiver
deepep_receiver_wait
deepep_receiver_unpack
deepep_receiver_topk_remap
deepep_receiver_metadata
deepep_receiver_post_quant
deepep_combine_submit
deepep_combine_receiver_copy
```

주의할 점:

이 profiling은 매 section마다 synchronize를 넣는다.
그래서 실제 serving overlap을 그대로 보존하는 profiler는 아니다.

목적은:

```text
정확한 end-to-end latency 측정
```

이 아니라:

```text
어느 코드 구간이 상대적으로 비싼지 병목 위치 찾기
```

이다.

## 3. AG/RS in-place combine 최적화

### 왜 필요했나

AG/RS finalize 경로에서는 expert 계산 결과를 reduce-scatter로 원래 rank에
되돌린다.

기존 흐름은 개념적으로 이랬다.

```python
tmp = get_ep_group().combine(out, ...)
output.copy_(tmp)
```

즉:

```text
1. reduce-scatter 결과 tensor를 새로 할당
2. 그 결과를 이미 준비된 output tensor에 copy
```

를 했다.

그런데 MoE modular kernel은 이미 최종 output buffer를 가지고 있다.
그러면 reduce-scatter가 처음부터 그 output buffer에 쓰면 된다.

### 어떻게 바꿨나

수정 파일:

- `vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py`
- `vllm/distributed/device_communicators/cuda_communicator.py`
- `vllm/distributed/device_communicators/all2all.py`
- `vllm/distributed/parallel_state.py`
- `vllm/distributed/device_communicators/base_device_communicator.py`
- `vllm/distributed/device_communicators/cpu_communicator.py`
- `vllm/distributed/device_communicators/xpu_communicator.py`

핵심 변경은 `combine`과 `reduce_scatterv`가 optional `out` 인자를 받게 한
것이다.

기존 AG/RS finalize:

```python
output.copy_(
    get_ep_group().combine(out, is_sequence_parallel=self.is_sequence_parallel)
)
```

변경 후:

```python
get_ep_group().combine(
    out,
    is_sequence_parallel=self.is_sequence_parallel,
    out=output,
)
```

CUDA communicator에서는 `out`이 들어오면 새 tensor를 만들지 않는다.

```python
if out is None:
    output = torch.empty(...)
else:
    output = out
```

그리고 NCCL reduce-scatter 출력으로 그 buffer를 직접 넘긴다.

```python
pynccl_comm.reduce_scatter(output, input_tensor)
```

### 코드 흐름

변경 후 흐름은 이렇다.

```text
naive_dp_ep.finalize
  -> get_ep_group().combine(out, out=output)
    -> GroupCoordinator.combine(..., out=output)
      -> CudaCommunicator.combine(..., out=output)
        -> AgRsAll2AllManager.combine(..., out=output)
          -> reduce_scatterv(..., out=output)
            -> NCCL reduce_scatter가 output에 직접 기록
```

즉 불필요한:

```text
tmp allocation
tmp -> output copy
```

를 줄였다.

### 결과

| 항목 | 전 | 후 |
|---|---:|---:|
| AG/RS finalize total | 0.120 ms | 0.101 ms |
| AG/RS finalize combine | 0.101 ms | 0.082 ms |
| AG/RS full forward | 1.178 ms | 1.179 ms |

해석:

- finalize 내부에서는 약 `19 us` 줄었다.
- 하지만 full forward에서는 노이즈 수준이다.
- AG/RS 전체 병목은 copy 하나가 아니라 NCCL collective latency 자체다.

그래도 의미는 있다.

이 패치는:

```text
작지만 실제로 불필요한 allocation/copy를 제거한 최적화
```

이고, section profile에서는 그 효과가 확인된다.

## 4. 실행한 benchmark

### DeepEP 기존 경로

```bash
PYTORCH_NVML_BASED_CUDA_CHECK=1 CUDA_VISIBLE_DEVICES=0,1 \
NCCL_P2P_DISABLE=0 VLLM_LOGGING_LEVEL=ERROR \
VLLM_DEEPEP_HT_NUM_SMS=24 VLLM_DEEPEP_HT_TRITON_TOPK_REMAP=0 \
.venv/bin/python benchmarks/kernels/benchmark_moe_ep_a40.py \
  --world-size 2 --backend deepep_high_throughput \
  --tokens 128 --hidden-size 2048 --intermediate-size 768 \
  --num-experts 128 --top-k 8 --warmup 5 --iters 20 --csv
```

### DeepEP Triton remap 경로

```bash
PYTORCH_NVML_BASED_CUDA_CHECK=1 CUDA_VISIBLE_DEVICES=0,1 \
NCCL_P2P_DISABLE=0 VLLM_LOGGING_LEVEL=ERROR \
VLLM_DEEPEP_HT_NUM_SMS=24 VLLM_DEEPEP_HT_TRITON_TOPK_REMAP=1 \
.venv/bin/python benchmarks/kernels/benchmark_moe_ep_a40.py \
  --world-size 2 --backend deepep_high_throughput \
  --tokens 128 --hidden-size 2048 --intermediate-size 768 \
  --num-experts 128 --top-k 8 --warmup 5 --iters 20 --csv
```

### AG/RS 경로

```bash
PYTORCH_NVML_BASED_CUDA_CHECK=1 CUDA_VISIBLE_DEVICES=0,1 \
NCCL_P2P_DISABLE=0 VLLM_LOGGING_LEVEL=ERROR \
.venv/bin/python benchmarks/kernels/benchmark_moe_ep_a40.py \
  --world-size 2 --backend allgather_reducescatter \
  --tokens 128 --hidden-size 2048 --intermediate-size 768 \
  --num-experts 128 --top-k 8 --warmup 5 --iters 20 --csv
```

## 5. 검증

실행한 검증:

```bash
.venv/bin/python -m py_compile \
  benchmarks/kernels/benchmark_moe_ep_a40.py \
  benchmarks/kernels/sweep_moe_ep_a40.py \
  vllm/envs.py \
  vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py \
  vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py \
  vllm/distributed/device_communicators/all2all.py \
  vllm/distributed/device_communicators/base_device_communicator.py \
  vllm/distributed/device_communicators/cuda_communicator.py \
  vllm/distributed/device_communicators/xpu_communicator.py \
  vllm/distributed/device_communicators/cpu_communicator.py \
  vllm/distributed/parallel_state.py
```

```bash
git diff --check
```

추가로 2GPU에서 `reduce_scatterv(..., out=...)` correctness smoke를 돌렸다.

확인한 것:

- 기존 out-of-place reduce-scatter 결과와 값이 같음
- 반환 tensor가 전달한 `out` buffer와 같은 메모리를 가리킴

`ruff`는 현재 `.venv`에 설치되어 있지 않아서 실행하지 못했다.

## 6. 결론

이번 변경으로 알게 된 것은 다음과 같다.

DeepEP 쪽:

```text
DeepEP 자체 통신이 느린 것이 아니라,
vLLM receiver에서 DeepEP output을 기존 expert kernel 인터페이스에 맞추는
비용이 꽤 있다.
```

특히:

```text
local expert id -> global expert id 변환
ExpertTokensMetadata 생성
```

이 receiver overhead의 핵심이다.

AG/RS 쪽:

```text
copy 하나를 줄일 수는 있지만,
전체 병목은 NCCL all-gather / reduce-scatter collective latency다.
```

그래서 다음 최적화 방향은:

1. DeepEP local expert id를 expert kernel이 그대로 먹게 해서 remap 자체 제거
2. `ExpertTokensMetadata.make_from_list` 비용 제거 또는 GPU-side metadata 사용
3. AG/RS는 allocation/copy보다 collective overlap, persistent buffer, fused combine 쪽으로 개선
4. H100/H200에서 DeepEP V2 + FP8/NVFP4 path 재검증

이다.

## 7. 더 깊은 실험: DeepEP HT local expert-id assignment path

위의 Triton remap 최적화는 너무 작은 변경이다.

그래서 다음 단계로 DeepEP HT가 받은 top-k id를 다시 global expert id로 되돌리지 않고,
expert kernel 쪽 assignment를 local expert-id 공간에서 수행하는 실험을 추가했다.

### 기존 경로

기존 DeepEP HT receiver 흐름은 다음과 같다.

```text
DeepEP recv_topk_idx
  local expert id / -1
    -> global expert id로 offset 적용
    -> -1은 현재 rank 밖의 global expert id로 치환
    -> TritonExperts
    -> moe_align_block_size(global_num_experts=128, expert_map=global->local)
```

즉 DeepEP가 이미 local expert id를 만들어 줬는데, vLLM의 기존
`TritonExperts` 인터페이스에 맞추기 위해 다시 global expert id로 바꾼다.

### 새 실험 경로

새 flag:

```text
VLLM_DEEPEP_HT_LOCAL_EXPERT_IDS=1
```

새 경로:

```text
DeepEP recv_topk_idx
  local expert id / -1
    -> -1만 local sentinel id로 치환
    -> TritonExperts
    -> moe_align_block_size(local_num_experts + 1, local_sentinel_map)
```

여기서 local sentinel map은 다음과 같다.

```text
[0, 1, 2, ..., local_num_experts - 1, -1]
```

마지막 entry만 invalid expert로 매핑된다.

이렇게 하면 `moe_align_block_size`가 global 128 experts 전체가 아니라,
현재 rank의 local 64 experts + invalid sentinel만 보게 된다.

### 수정한 코드

수정 파일:

```text
vllm/envs.py
vllm/model_executor/layers/fused_moe/modular_kernel.py
vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py
benchmarks/kernels/benchmark_moe_ep_a40.py
```

핵심 변경:

- `vllm/envs.py`
  - `VLLM_DEEPEP_HT_LOCAL_EXPERT_IDS` 추가
- `modular_kernel.py`
  - `FusedMoEPrepareAndFinalizeModular.expert_assignment_params()` hook 추가
  - prepare/finalize backend가 expert assignment에 사용할 expert id 공간을 바꿀 수 있게 함
- `deepep_ht.py`
  - flag off: 기존처럼 local id를 global id로 remap
  - flag on: `-1`만 local sentinel로 remap
  - local expert map `[0..local_num_experts-1, -1]`를 캐시
- `benchmark_moe_ep_a40.py`
  - section profiler wrapper도 새 remap 함수와 local-id flag를 정확히 따라가도록 수정

### 결과

같은 조건:

```text
2x A100 SXM, DeepEP HT, 24 SM
tokens=128, hidden=2048, intermediate=768, experts=128, top_k=8, BF16
VLLM_DEEPEP_HT_TRITON_TOPK_REMAP=1
```

full forward:

| path | full forward |
|---|---:|
| global expert-id path | 1.355 ms |
| local expert-id path | 1.383 ms |

section profile:

| path | prepare | top-k remap | metadata | experts | finalize |
|---|---:|---:|---:|---:|---:|
| global expert-id path | 0.351-0.358 ms | 0.050 ms | 0.064 ms | 0.695 ms | 0.157-0.163 ms |
| local expert-id path | 0.358-0.364 ms | 0.051 ms | 0.066 ms | 0.700 ms | 0.159-0.163 ms |

### 해석

이 실험은 코드상으로는 더 깊은 변경이다.

하지만 현재 shape에서는 성능이 좋아지지 않았다.

이유는 다음과 같이 보인다.

```text
DeepEP HT recv_x는 expert별로 이미 펼쳐진 tensor가 아니다.
recv_x는 "이 rank에 필요한 token 목록"이고,
recv_topk_idx 안에 local expert id와 -1이 섞여 있다.
```

작은 디버그 dispatch에서 확인한 예:

```text
recv_x rows: token 목록
recv_topk_idx: [[0, 1], [0, -1], [-1, 1], ...]
counts: [expert0_count, expert1_count, ...]
```

즉 `recv_x` 자체가 expert-contiguous layout이 아니므로,
기존 `moe_align_block_size`는 여전히 top-k id를 훑고 token/expert pair를
정렬해야 한다.

local id로 줄여도 이 핵심 정렬/align 비용이 크게 줄지 않는다.

따라서 이 실험의 결론은:

```text
local/global expert id 변환만 줄이는 것은 부족하다.
DeepEP HT 전용 expert-assignment kernel을 새로 만들어야 한다.
```

### 다음 커널 작업

진짜 포트폴리오 핵심으로 만들려면 다음 단계가 맞다.

```text
DeepEP HT recv_topk_idx + topk_weights를 직접 읽는
새 Triton expert-assignment kernel 작성
```

목표:

1. `moe_align_block_size` generic custom op 우회
2. local expert id와 `-1` invalid slot을 native하게 처리
3. `expert_num_tokens_per_expert`를 prefix sum / block scheduling에 직접 사용
4. 가능하면 sorted token id 생성과 padding 계산을 DeepEP HT layout에 맞춰 축소

이 방향은 단순 backend 선택이 아니라,

```text
DeepEP HT receiver layout -> Triton expert GEMM scheduler
```

사이의 kernel contract를 바꾸는 작업이다.

## 8. Prefill / Decode 분리와 kernel-level profile

단순 latency table만으로는 부족해서, prefill-like / decode-like shape를
나누어 CUDA kernel profile을 추가했다.

현재 머신에는 다음 도구가 PATH에 없었다.

```text
ncu
nsys
```

그래서 이번에는 `torch.profiler`의 CUDA event 요약을 benchmark에 추가했다.
Nsight Compute/Systems가 설치되면 같은 shape와 같은 NVTX/section 기준으로
바로 더 깊게 들어가면 된다.

### 추가한 benchmark 옵션

수정 파일:

```text
benchmarks/kernels/benchmark_moe_ep_a40.py
```

추가 옵션:

```text
--phase-name decode|prefill
--torch-profile-iters N
--torch-profile-warmup N
--torch-profile-output PATH
--torch-profile-top-kernels N
--torch-profile-chrome-trace
```

이 옵션은 forward 몇 회를 `torch.profiler`로 감싸고,
CUDA 시간이 큰 event/kernel을 rank별 JSON으로 저장한다.

### 사용한 shape

decode-like:

```text
tokens=16
```

의미:

```text
16개 request가 각각 1 token decode하는 상황에 가까운 synthetic shape
```

prefill-like:

```text
tokens=512
```

의미:

```text
긴 prompt/chunk prefill처럼 한 번에 많은 token이 MoE로 들어가는 상황
```

공통 조건:

```text
2x A100 SXM
hidden=2048
intermediate=768
experts=128
top_k=8
BF16
```

### AG/RS 결과

| phase | full forward | dispatch | experts | finalize | raw dispatch | raw combine |
|---|---:|---:|---:|---:|---:|---:|
| decode-like, tokens=16 | 1.005 ms | 0.126 ms | 0.516-0.536 ms | 0.086-0.107 ms | 0.403 ms | 0.336 ms |
| prefill-like, tokens=512 | 1.493 ms | 0.166-0.172 ms | 0.936-0.959 ms | 0.121-0.145 ms | 0.407 ms | 0.348 ms |

CUDA event 상위 항목:

| phase | 주요 CUDA event |
|---|---|
| decode-like | `fused_moe_kernel` 약 235-254 us/fwd, NCCL all-gather/reduce-scatter, `moe_align_block_size` 약 9 us/fwd, `moe_sum` 약 9 us/fwd |
| prefill-like | `fused_moe_kernel` 약 612-628 us/fwd, NCCL all-gather/reduce-scatter, `moe_sum` 약 31-33 us/fwd, `silu_and_mul` 약 26 us/fwd |

해석:

```text
decode는 collective/launch floor가 너무 크다.
prefill은 expert GEMM 비중이 커지고, fused_moe_kernel 두 번이 주된 compute 비용이다.
```

decode-like routing 통계:

```text
expert_tokens_mean=1.0
expert_tokens_zero=46 / 128
```

즉 decode에서는 expert batch가 매우 작고 sparse하다.
이 상황에서는 큰 GEMM 최적화보다 launch 수, collective floor, ragged scheduling이 더 중요하다.

prefill-like routing 통계:

```text
expert_tokens_mean=32.0
expert_tokens_zero=0 / 128
```

prefill에서는 expert별 token 수가 커져서 tensor core 활용과 expert GEMM scheduling이 중요해진다.

### DeepEP HT 결과

조건:

```text
VLLM_DEEPEP_HT_NUM_SMS=24
VLLM_DEEPEP_HT_TRITON_TOPK_REMAP=1
```

| phase | full forward | prepare | experts | finalize |
|---|---:|---:|---:|---:|
| decode-like, tokens=16 | 1.238 ms | 0.364-0.374 ms | 0.521-0.545 ms | 0.140-0.162 ms |
| prefill-like, tokens=512 | 1.682 ms | 0.403-0.409 ms | 0.934-0.947 ms | 0.202-0.216 ms |

DeepEP HT section breakdown:

| phase | dispatch submit | dispatch receiver | top-k remap | metadata | combine submit | receiver copy |
|---|---:|---:|---:|---:|---:|---:|
| decode-like | 0.137-0.145 ms | 0.192-0.193 ms | 0.052 ms | 0.065 ms | 0.079-0.099 ms | 0.027-0.028 ms |
| prefill-like | 0.171-0.176 ms | 0.196 ms | 0.054 ms | 0.066 ms | 0.139-0.153 ms | 0.028 ms |

CUDA event 상위 항목:

| phase | 주요 CUDA event |
|---|---|
| decode-like | `fused_moe_kernel` 약 232-256 us/fwd, DeepEP dispatch/combine kernels 약 20-30 us/fwd, copy/index/elementwise |
| prefill-like | `fused_moe_kernel` 약 608-620 us/fwd, DeepEP dispatch/combine kernels 약 70 us/fwd, `moe_sum` 약 32 us/fwd, `silu_and_mul` 약 26 us/fwd |

주의:

`torch.profiler`에서 DeepEP `notify_dispatch`가 rank0에서 큰 값으로 보인 run이 있었지만,
동일 run의 synchronized section timing은 `dispatch submit` 약 0.17 ms였다.
DeepEP async/stream boundary 때문에 CUDA event 집계는 rank별 outlier가 섞일 수 있으므로,
DeepEP는 section timing과 Nsight Systems timeline으로 다시 확인해야 한다.

### FlashMoE식 fusion 가능성

FlashMoE는 README 기준으로 다음을 하나의 persistent kernel로 합친다.

```text
MoE Dispatch
Expert Computation
MoE Combine
```

그리고 tile granularity로 communication과 compute를 overlap한다.

하지만 현재 vLLM FlashMoE adapter는 다음 제약이 있다.

```text
BF16
EP 필요
top-1/top-2만 지원
renormalized softmax top-k routing 필요
```

우리 Qwen3-like 실험은:

```text
top_k=8
```

이라서 FlashMoE를 그대로 붙이는 것은 핵심 프로젝트로 맞지 않는다.

대신 가져올 아이디어는 다음이다.

1. decode path

```text
목표: 작은 expert batch에서 launch/collective floor 줄이기
```

후보:

- all-gather / expert compute / reduce-scatter overlap
- top-k metadata, copy, `moe_sum` 같은 작은 kernel들을 persistent path 안으로 흡수
- decode 전용 small-M expert scheduler

단, decode에서는 `fused_moe_kernel` 자체보다 통신/launch floor가 커서,
GEMM microkernel만 바꾸면 효과가 작을 가능성이 높다.

2. prefill path

```text
목표: expert GEMM + activation + top-k weighted reduce의 kernel boundary 축소
```

후보:

- GEMM2 epilogue에서 top-k weight 적용 + `moe_sum`까지 fuse
- `silu_and_mul`과 GEMM2 입력 생성 경계를 줄이기
- expert assignment 결과를 더 직접적으로 GEMM scheduler에 연결
- DeepEP/AG-RS communication을 compute tile과 overlap

prefill에서는 `fused_moe_kernel`이 약 0.61 ms/fwd로 커지므로,
FlashMoE식 tile scheduler/fusion이 더 설득력 있는 target이다.

### 다음 작업 결론

이제 프로젝트 방향은 이렇게 잡는 것이 좋다.

```text
1순위: Nsight Systems로 prefill/decode timeline 확정
2순위: Nsight Compute로 fused_moe_kernel의 tensor core utilization,
       memory throughput, occupancy, stall reason 확인
3순위: prefill 전용 fusion prototype
       - GEMM2 epilogue + top-k reduce/moe_sum fusion
       - 또는 DeepEP HT assignment -> GEMM scheduler direct path
4순위: decode 전용 overlap/small-batch path
```

핵심 메시지:

```text
decode와 prefill의 병목이 다르다.
decode는 communication/launch floor,
prefill은 expert GEMM과 kernel boundary가 더 큰 병목이다.
```

따라서 “FlashMoE처럼 fusion”은 decode보다 prefill에서 먼저 하는 것이 맞다.

## 4. A100 W2 Triton config override 실험

### 배경

`fused_moe_kernel`의 W2 GEMM은 이미 Triton PTX/SASS에서 Tensor Core
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`를 사용하고 있었다.
따라서 먼저 CUDA로 재작성하기보다, 현재 A100/SM80에서 W2에 맞는 Triton
meta config가 있는지 분리 측정했다.

추가한 benchmark:

- `benchmarks/kernels/benchmark_moe_w2_triton_sweep.py`

이 benchmark는 W2 kernel만 호출해서 `BLOCK_SIZE_M/N/K`, `num_warps`,
`num_stages` 후보를 sweep하고, baseline 출력과 correctness를 비교한다.

### W2-only sweep 결과

조건:

- GPU: A100-SXM4-80GB
- dtype: BF16
- hidden size: `2048`
- intermediate size: `768`
- experts: `128`
- top-k: `8`

| tokens | baseline config | best config | W2-only latency | 의미 |
|---:|---|---|---:|---|
| 512 | M64/N128/K64/w8/s3 | M32/N128/K64/w4/s3 | 295.9 us -> 286.8 us | 약 3.1% 개선 |
| 1024 | M128/N128/K64/w8/s3 | M64/N128/K64/w4/s3 | 407.7 us -> 316.8 us | 약 22.3% 개선 |

1024 tokens부터 기본 config가 W2에는 너무 큰 M tile과 많은 warp를 선택하는
것이 확인됐다.

### 코드 변경

수정 파일:

- `vllm/envs.py`
- `vllm/model_executor/layers/fused_moe/experts/triton_moe.py`

추가한 feature flag:

```text
VLLM_MOE_TRITON_W2_A100_TUNED_CONFIG
```

기본값은 `0`이다.

flag를 켜면 아래 조건에서만 W2 config를 override한다.

```text
CUDA SM80
BF16
E=128, hidden=2048, intermediate=768
top-k=8
tokens >= 1024
quantization 없음
LoRA 없음
W2 reduce fusion 꺼짐
```

적용 config:

```python
{
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "SPLIT_K": 1,
    "num_warps": 4,
    "num_stages": 3,
}
```

중요한 구현 세부사항:

W2의 `BLOCK_SIZE_M`을 바꾸면 expert assignment padding도 달라진다.
그래서 기존 W1용 `sorted_token_ids`를 재사용하지 않고, W2 config가 바뀌는
경우에만 W2용 `_prepare_expert_assignment(...)`를 별도로 수행한다.

### End-to-end 결과

AG/RS, world size 1 synthetic MoE path에서 측정했다.

| tokens | baseline | tuned flag on | 개선 |
|---:|---:|---:|---:|
| 1024 | 1209.3 us | 1129.0 us | 약 6.6% |
| 2048 | 1727.3 us | 1602.3 us | 약 7.2% |

정확성 확인:

```text
correctness ok max_abs=0.0 mean_abs=0.0
```

### 해석

이 결과는 “sum kernel 교체”나 “atomic epilogue reduce fusion”보다 W2 GEMM
자체 config가 더 직접적인 개선 지점임을 보여준다. 다만 W2-only 22%가
end-to-end 6~7%로 줄어든 이유는 W1/activation/top-k 비용과 W2용 assignment
재생성 비용이 남아 있기 때문이다.

다음 단계는 이 config override를 더 넓은 token bucket과 실제 prefill trace에
대해 검증하거나, 같은 shape에 대해 CUTLASS/CuTe W2 전용 kernel을 만들어
W2용 assignment 재생성 비용까지 줄일 수 있는지 확인하는 것이다.

## 5. W2 tuned config Nsight Systems 확인

`VLLM_MOE_TRITON_W2_A100_TUNED_CONFIG=1`이 실제로 W2 `fused_moe_kernel` 시간을
줄이는지 확인하기 위해 `world_size=1`, `tokens=1024` 조건에서 짧은 Nsight
Systems trace를 수집했다.

수집 방식:

```text
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none
```

2-GPU 전체 trace는 `torch.multiprocessing.spawn` + NCCL + fork tracing 때문에
export 비용이 커서 사용하지 않았다. W2 kernel 자체 확인은 `world_size=1`에서
하고, 2-GPU EP는 latency/section timing으로 판단했다.

### world_size=1 Nsight 결과

파일:

- `benchmarks/results/nsys_w1_moe_t1024_base.nsys-rep`
- `benchmarks/results/nsys_w1_moe_t1024_tuned.nsys-rep`
- `benchmarks/results/a100_sxm_w2cfg_nsys_summary.json`

`fused_moe_kernel`은 forward마다 W1, W2 순서로 두 번 실행된다.
Nsight kernel event를 시간순으로 파싱해서 짝수 번째를 W1, 홀수 번째를 W2로
분리했다.

| kernel | baseline | tuned | 변화 |
|---|---:|---:|---:|
| W1 `fused_moe_kernel` | 710.3 us | 711.3 us | 변화 없음 |
| W2 `fused_moe_kernel` | 408.7 us | 307.4 us | 약 24.8% 개선 |

추가로 tuned 경로에서는 W2용 `_prepare_expert_assignment(...)` 때문에
`moe_align_block_size_kernel`이 W2 앞에 한 번 더 실행된다. 이 비용은 약
`10-11 us`다. 그래도 W2 GEMM 감소폭이 더 커서 net win이다.

Nsight에서 확인된 W2 launch shape 변화:

```text
baseline W2: grid=(3056,1,1), block=(256,1,1), regs=164
 tuned W2:  grid=(4064,1,1), block=(128,1,1), regs=168
```

해석:

```text
W1은 그대로이고 W2만 빨라졌다.
즉 env flag가 의도한 W2 call-site config override에 정확히 걸렸다.
```

### 2-GPU EP 판단

2-GPU AG/RS는 Nsight Systems 전체 trace 대신 latency와 section timing으로
판단했다.

조건:

```text
world_size=2
backend=allgather_reducescatter
tokens=1024
hidden=2048
intermediate=768
experts=128
top_k=8
BF16
NCCL_P2P_DISABLE=0
```

| 항목 | baseline | tuned | 변화 |
|---|---:|---:|---:|
| full forward | 1763.5 us | 1711.9 us | 약 2.9% 개선 |
| raw dispatch | 371.4 us | 367.5 us | 거의 동일 |
| raw combine | 296.7 us | 296.4 us | 거의 동일 |

해석:

```text
W2 kernel 자체는 world_size=1에서 약 25% 줄지만,
2-GPU EP 전체에서는 all-gather/reduce-scatter와 나머지 MoE 비용이 섞여
end-to-end 개선이 약 3%로 희석된다.
```

따라서 이 변경은 유효하지만, 2-GPU EP 전체를 크게 줄이려면 다음 병목은
W2 단독 config가 아니라 communication/compute overlap 또는 dispatch/combine
경로다.

## 6. W1/W13 tuned config 추가

W2 tuned config가 유효했기 때문에, 같은 방식으로 W1/W13 `fused_moe_kernel`도
분리 sweep했다.

추가한 benchmark:

- `benchmarks/kernels/benchmark_moe_w1_triton_sweep.py`

조건:

```text
A100-SXM4-80GB
BF16
tokens=1024
hidden=2048
intermediate=768
top_k=8
```

### W1-only sweep 결과

| local experts | baseline | best | 개선 |
|---:|---:|---:|---:|
| 128 | 707.0 us | 569.1 us | 약 19.5% |
| 64 | 516.2 us | 385.9 us | 약 25.2% |

best config는 W2와 동일했다.

```python
{
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "SPLIT_K": 1,
    "num_warps": 4,
    "num_stages": 3,
}
```

### 코드 변경

추가 flag:

```text
VLLM_MOE_TRITON_W1_A100_TUNED_CONFIG
```

기본값은 `0`이다. 조건은 W2 tuned config와 같이 좁게 잡았다.

```text
CUDA SM80
BF16
E=64 또는 128
W1 shape=(E, 1536, 2048)
top-k=8
tokens >= 1024
quantization 없음
LoRA 없음
```

W1과 W2가 같은 tuned config를 쓰는 경우 W2용 expert assignment는 W1용
assignment를 재사용한다. 다만 현재 구조상 기존 base assignment는 아직 먼저
한 번 만들어진다. 이 부분은 다음 micro-optimization 후보이다.

### Correctness

동일 입력/weight/router logits에서 baseline과 비교했다.

```text
w1 correctness ok max_abs=0.0 mean_abs=0.0
w2 correctness ok max_abs=0.0 mean_abs=0.0
w1w2 correctness ok max_abs=0.0 mean_abs=0.0
```

### world_size=1 end-to-end

| 설정 | full forward | 개선 |
|---|---:|---:|
| baseline | 1210.3 us | - |
| W1 tuned only | 1078.4 us | 약 10.9% |
| W1+W2 tuned | 999.8 us | 약 17.4% |

짧은 Nsight Systems trace로 W1+W2 kernel time도 확인했다.

| kernel | tuned 평균 |
|---|---:|
| W1 `fused_moe_kernel` | 566.5 us |
| W2 `fused_moe_kernel` | 315.3 us |

파일:

- `benchmarks/results/nsys_w1w2_moe_t1024_tuned.nsys-rep`
- `benchmarks/results/nsys_w1w2_moe_t1024_tuned.sqlite`
- `benchmarks/results/a100_sxm_w1w2cfg_summary.json`

### 2-GPU EP 결과

AG/RS, `world_size=2`, `tokens=1024`, `NCCL_P2P_DISABLE=0` 조건이다.

| 설정 | full forward | 개선 |
|---|---:|---:|
| baseline | 1753.0 us | - |
| W2 tuned only | 1701.8 us | 약 2.9% |
| W1+W2 tuned | 1612.8 us | 약 8.0% |

section timing에서도 expert compute가 줄었다.

| 설정 | rank0 experts | rank1 experts |
|---|---:|---:|
| baseline | 1247.2 us | 1232.0 us |
| W2 tuned only | 1181.6 us | 1177.3 us |
| W1+W2 tuned | 1089.8 us | 1088.7 us |

해석:

```text
W1+W2 config tuning은 단순 Triton meta config 변경만으로 2-GPU EP에서 약 8%를 만든다.
남은 병목은 여전히 prepare/finalize 통신과 expert compute의 residual이다.
다음 후보는 base assignment 중복 제거 또는 AG/RS 통신 overlap이다.
```

## 7. A100 BF16 전용 kernel-body fork 실험

단순 config tuning과 분리하기 위해 generic `fused_moe_kernel`을 우회하는
A100/SM80 BF16 전용 MoE Triton kernel-body fork를 추가했다.

추가 파일:

- `vllm/model_executor/layers/fused_moe/a100_moe_kernels.py`

추가 flag:

```text
VLLM_MOE_A100_BF16_SPECIALIZED_KERNELS
```

조건은 의도적으로 좁게 잡았다.

```text
CUDA SM80
BF16
top-k=8
tokens >= 1024
quantization 없음
LoRA 없음
bias 없음
W1 shape=(E, 1536, 2048), E=64 또는 128
W2 shape=(128, 2048, 768)
```

Correctness:

```text
baseline vs specialized max_abs=0.0 mean_abs=0.0
```

world_size=1, AG/RS benchmark shape:

| 설정 | full forward |
|---|---:|
| baseline Triton | 1209.7 us |
| W1+W2 tuned config | 999.2 us |
| A100 BF16 specialized kernel-body | 1035.4 us |

해석:

```text
전용 kernel-body fork는 정확하지만, 현재 body는 generic Triton kernel보다 빠르지 않다.
generic kernel의 quant/bias/reduce branch가 constexpr로 대부분 제거되기 때문에,
단순 branch 제거만으로는 portfolio-grade 개선이 나오지 않는다.
다음 성능 개선은 schedule/layout/reduce 방식 자체를 바꿔야 한다.
```

## 8. W2 epilogue fused top-k reduce 재검증

W2 GEMM epilogue에서 top-k reduce까지 수행해 `intermediate_cache3` write/read와
별도 `moe_sum` launch를 없애는 경로를 다시 검증했다.

기존 flag:

```text
VLLM_MOE_TRITON_W2_REDUCE_FUSION
```

이번 변경:

- W2 reduce-fusion 경로에서도 A100 W2 tuned config를 사용할 수 있게 했다.
- BF16 final output에 직접 atomic add하는 direct-output 실험은 정확도 실패로 폐기했다.
- 정확한 경로는 FP32 workspace atomic reduce 후 output copy를 유지한다.

Correctness:

```text
FP32 workspace fused reduce max_abs=0.0 mean_abs=0.0
BF16 direct-output atomic max_abs≈0.295, norm mismatch -> 폐기
```

world_size=1 결과:

| 설정 | full forward |
|---|---:|
| baseline Triton | 1209.7 us |
| W1+W2 tuned config | 999.2 us |
| W1+W2 tuned + W2 fused reduce | 1060.8 us |
| W1 tuned + W2 fused reduce | 1161.4 us |

해석:

```text
atomic epilogue reduce는 정확하지만, atomic 비용과 FP32 workspace/copy 비용이
sum launch 제거 이득보다 크다. 따라서 현재 승리 경로는 W1+W2 tuned config이고,
진짜 fusion 승리 경로는 atomic 없는 token-major/direct scheduler 쪽이어야 한다.
```

## 9. 2-GPU EP 재측정과 overlap 후보

같은 코드 상태에서 2-GPU AG/RS를 다시 측정했다.
이번 런의 절대 latency는 이전 기록보다 높았지만, baseline과 tuned를 같은 조건에서
재서 상대 개선을 확인했다.

조건:

```text
world_size=2
backend=allgather_reducescatter
tokens=1024
hidden=2048
intermediate=768
num_experts=128
top_k=8
```

end-to-end:

| 설정 | full forward |
|---|---:|
| baseline | 2197.5 us |
| W1+W2 tuned | 2008.8 us |

상대 개선은 약 8.6%다.

section timing:

| 설정 | rank0 experts | rank1 experts | rank0 dispatch | rank0 combine |
|---|---:|---:|---:|---:|
| baseline | 1246.1 us | 1229.8 us | 479.0 us | 451.4 us |
| W1+W2 tuned | 1088.6 us | 1089.0 us | 468.7 us | 448.0 us |

파일:

- `benchmarks/results/a100_sxm_ep_baseline_sections.rank0.json`
- `benchmarks/results/a100_sxm_ep_baseline_sections.rank1.json`
- `benchmarks/results/a100_sxm_ep_w1w2_tuned_sections.rank0.json`
- `benchmarks/results/a100_sxm_ep_w1w2_tuned_sections.rank1.json`

해석:

```text
W1+W2 tuned config는 2-GPU에서도 expert compute를 약 140-160 us 줄인다.
prepare/finalize AG/RS 통신은 거의 그대로라, 다음 decisive overlap 후보는
hidden_states all-gather를 top-k와 겹치도록 AG/RS dispatch API를 split하는 것이다.
현재 API는 hidden/topk tensors를 한 번에 dispatch하므로, 실제 overlap을 넣으려면
prepare 단계의 dispatch contract를 먼저 쪼개야 한다.
```

## 10. W2 token-major reduce kernel-body prototype

config tuning만으로는 portfolio-grade 변경이라고 보기 어렵기 때문에,
W2 계산 구조 자체를 바꾸는 decode/small-batch 전용 kernel-body prototype을 추가했다.

추가 flag:

```text
VLLM_MOE_A100_BF16_W2_TOKEN_MAJOR_REDUCE
VLLM_MOE_A100_BF16_W2_TOKEN_MAJOR_REDUCE_MAX_TOKENS
```

기본값은 둘 다 안전하게 잡았다.

```text
VLLM_MOE_A100_BF16_W2_TOKEN_MAJOR_REDUCE=0
VLLM_MOE_A100_BF16_W2_TOKEN_MAJOR_REDUCE_MAX_TOKENS=1
```

핵심 변경:

```text
기존:
  W2 expert-major GEMM -> intermediate_cache3 write
  -> moe_sum launch -> output

prototype:
  token-major W2 kernel에서 top-k=8 expert contribution을 직접 누산
  -> output에 바로 store
  -> intermediate_cache3 write/read 및 moe_sum launch skip
```

코드:

- `vllm/model_executor/layers/fused_moe/a100_moe_kernels.py`
  - `a100_bf16_w2_token_major_reduce_kernel`
  - `invoke_a100_bf16_w2_token_major_reduce_kernel`
- `vllm/model_executor/layers/fused_moe/experts/triton_moe.py`
  - `_use_a100_bf16_w2_token_major_reduce_kernel`

조건:

```text
CUDA SM80
BF16
W2 shape=(128, 2048, 768)
top-k=8
tokens <= max_tokens
quantization 없음
LoRA 없음
bias 없음
apply_router_weight_on_input=False
```

Correctness:

```text
tokens=8 baseline vs token-major:
max_abs=0.0178
mean_abs=0.00040
allclose(atol=5e-2, rtol=5e-2)=True
```

world_size=1:

| tokens | baseline | token-major W2 reduce | 결과 |
|---:|---:|---:|---:|
| 1 | 485.8 us | 464.2 us | 약 4.4% 개선 |
| 8 | 522.7 us | 502.7 us | 약 3.8% 개선 |
| 16 | 527.2 us | 598.5 us | 악화 |

world_size=2 AG/RS:

| tokens | baseline | token-major W2 reduce | 결과 |
|---:|---:|---:|---:|
| 1 | 972.0 us | 953.5 us | 약 1.9% 개선이지만 통신 noise 대비 작음 |
| 8 | 957.5 us | 980.0 us | 악화 |

해석:

```text
이 변경은 config tuning이 아니라 실제 W2 kernel body/schedule 변경이다.
single-token decode에서는 intermediate_cache3와 moe_sum을 제거한 효과가
나오지만, token-major schedule은 expert-major tensor-core GEMM보다 산술 효율이
낮아서 tokens가 커지면 불리하다.

따라서 현재 이 path는 2-GPU EP 개선으로 주장하면 안 된다. 코드에서는
EP local expert shard(E=64)에서 이 path가 타지 않도록 막고, single-GPU
full-expert(E=128) decode prototype으로만 남긴다. prefill 및 multi-token decode의
승리 경로는 여전히 expert-major tensor-core 효율을 유지하면서 reduce를 합치는
scheduler가 필요하다.
```

## 11. EP invalid expert assignment skip prototype

추가 개선점을 찾는 과정에서 AG/RS EP의 expert assignment 낭비를 확인했다.

기존 AG/RS EP Triton path는 global expert id 공간에서 assignment를 만든 뒤,
현재 rank에 없는 expert block은 `expert_ids=-1`로 표시하고 kernel 안에서 zero-store
후 return한다. 즉 off-rank expert에 대해서도 assignment/padding/block launch가
발생한다.

`moe_align_block_size`에는 이미 `ignore_invalid_experts=True` 옵션이 있었지만,
TritonExperts EP path에서는 사용하지 않았다.

추가 flag:

```text
VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS
```

기본값은 `0`이다.

변경 내용:

```text
1. expert_map이 있는 EP Triton path에서 invalid expert를 assignment 단계에서 제거
2. W2에서 router weight multiply를 하지 않음
3. final sum을 expert_map-aware `moe_fused_mul_sum`으로 바꿔
   invalid top-k slot을 mask하고 weight multiply + sum을 한 kernel에서 처리
```

조건:

```text
EP enabled (expert_map != None)
CUDA SM80 BF16
top-k=8
tokens >= 1024
quantization 없음
LoRA 없음
apply_router_weight_on_input=False
W2 reduce fusion off
```

Correctness, 2-GPU:

```text
rank0 max_abs=0.00390625 mean_abs=7.05e-05 allclose_1e_2=True
rank1 max_abs=0.00390625 mean_abs=7.05e-05 allclose_1e_2=True
```

2-GPU AG/RS, W1+W2 tuned config 위에 추가:

| 설정 | full forward |
|---|---:|
| W1+W2 tuned | 2046.6 us |
| W1+W2 tuned + ignore invalid experts | 1944.4 us |

추가 개선은 약 5.0%다.

section timing:

| 설정 | rank0 experts | rank1 experts | rank0 prepare | rank1 prepare |
|---|---:|---:|---:|---:|
| W1+W2 tuned | 1088.6 us | 1089.0 us | 489.1 us | 491.8 us |
| tuned + ignore invalid | 970.9 us | 957.2 us | 510.4 us | 522.2 us |

파일:

- `benchmarks/results/a100_sxm_ep_tuned_ignore_invalid_sections.rank0.json`
- `benchmarks/results/a100_sxm_ep_tuned_ignore_invalid_sections.rank1.json`

해석:

```text
invalid expert block을 assignment 단계에서 제거하면 expert compute가 약 118-132 us 줄어든다.
대신 expert_map-aware align/final masked sum 때문에 prepare가 약 20-30 us 늘어난다.
순효과는 2-GPU end-to-end 약 5% 추가 개선이다.

다음 개선은 invalid slot을 W1/W2에서만 제거하는 데서 끝내지 않고,
activation도 compact local-routed rows에 대해서만 실행하게 만드는 것이다.
현재는 W1/W2 invalid block은 줄였지만 activation은 여전히 dense (tokens * top_k)
buffer 전체에 대해 실행된다.
```

## 12. EP masked activation prototype

`VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS=1` 이후에도 남는 낭비는 activation이다.
W1/W2 assignment에서는 off-rank expert slot을 제거하지만, 기존
`torch.ops._C.silu_and_mul`은 여전히 `(tokens * top_k, 2 * intermediate)` 전체
row를 처리한다.

추가 flag:

```text
VLLM_MOE_TRITON_EP_MASKED_ACTIVATION
```

기본값은 `0`이다.

변경 내용:

```text
1. EP ignore-invalid path에서만 동작하는 masked SiLU+mul Triton kernel 추가
2. topk_ids + expert_map으로 현재 rank가 처리하지 않는 route row 판정
3. invalid row에서는 gate/up activation input을 읽지 않고 output도 쓰지 않음
4. valid row만 SiLU(gate) * up 계산
```

조건:

```text
VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS=1
EP enabled (expert_map != None)
CUDA SM80 BF16
activation=silu
top-k=8
gemm1_clamp_limit 없음
quantization 없음
LoRA 없음
```

Correctness, 2-GPU:

```text
rank0 max_abs=0.0029296875 mean_abs=1.03e-04 allclose_1e_2=True
rank1 max_abs=0.0029296875 mean_abs=1.03e-04 allclose_1e_2=True
```

activation microbenchmark, tokens=1024/top_k=8/intermediate=768:

| kernel | latency |
|---|---:|
| torch `silu_and_mul` dense | 28.0 us |
| masked SiLU+mul | 24.5 us |

2-GPU AG/RS, W1+W2 tuned + ignore-invalid 위에 추가:

| 설정 | full forward |
|---|---:|
| ignore invalid experts | 1478.4 us |
| ignore invalid + masked activation | 1464.6 us |

추가 개선은 약 `13.8 us`, `0.9%`다.

section timing:

| 설정 | rank0 experts | rank1 experts |
|---|---:|---:|
| ignore invalid | 975.5 us | 960.8 us |
| ignore invalid + masked activation | 963.8 us | 944.2 us |

파일:

- `benchmarks/results/a100_sxm_ep_ignore_invalid_no_mask_sections.rank0.json`
- `benchmarks/results/a100_sxm_ep_ignore_invalid_no_mask_sections.rank1.json`
- `benchmarks/results/a100_sxm_ep_ignore_invalid_masked_activation_sections.rank0.json`
- `benchmarks/results/a100_sxm_ep_ignore_invalid_masked_activation_sections.rank1.json`

해석:

```text
activation skip은 실제로 experts 구간을 약 12-17 us 줄인다.
다만 dense activation 자체가 약 28 us라 전체 MoE latency 관점의 상한이 작다.
따라서 이 path는 보조 개선이고, 큰 다음 과제는 W1 epilogue에서 activation까지
직접 생성하거나 local-routed compact buffer를 만들어 activation/W2/final reduce를
같은 compact route space에서 처리하는 것이다.
```

추가로 `ignore-invalid + W2 reduce fusion` 조합도 확인했다. 이론적으로는
`intermediate_cache3`와 final sum을 없앨 수 있지만, 현재 Triton W2 reduce path는
atomic add 비용이 커서 느렸다.

| 설정 | full forward |
|---|---:|
| ignore invalid experts | 1483.0 us |
| ignore invalid + W2 reduce fusion | 1586.0 us |

따라서 현재 추천 경로는 W2 atomic reduce fusion이 아니라
`ignore-invalid + masked activation`이다.

## 13. 2026-06-20 A100 SXM 재현 실험

`AGENTS.md`의 즉시 실험 큐를 기준으로 현재 워크스페이스의 최신 EP Triton
커널 후보를 다시 검증했다. 대상은 `VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS`와
`VLLM_MOE_TRITON_EP_MASKED_ACTIVATION` 조합이다.

환경:

```text
commit=3f556a52b61b5dd685366bc4c92382ca73d1d923
GPU=2x NVIDIA A100-SXM4-80GB, NV12
driver=570.172.08
CUDA toolkit=12.8.93
torch=2.11.0+cu128
torch CUDA=12.8
triton=3.6.0
vllm=0.23.1.dev0
```

환경 메모:

```text
VLLM_USE_PRECOMPILED=1 editable install을 사용했다.
precompiled wheel에는 vllm._C_stable_libtorch와 vllm._moe_C_stable_libtorch는
있지만 vllm._C가 없어 CUDA platform import가 실패했다.
이번 벤치마크에서는 파일 변경 없이 /tmp/vllm_c_stub/sitecustomize.py로
vllm._C import만 stub 처리했고, MoE stable extension들은 정상 로드됐다.
```

Correctness smoke:

```text
tests/kernels/moe/test_deepep_ht_expert_assignment.py
52 passed
```

성능 조건:

```text
NCCL_P2P_DISABLE=0
world_size=2
backend=allgather_reducescatter
tokens=1024
hidden=2048
intermediate=768
num_experts=128
local_experts=64
top_k=8
dtype=BF16
warmup=10
iters=50
```

end-to-end:

| 설정 | full forward | baseline 대비 |
|---|---:|---:|
| baseline | 1869.2 us | - |
| W1+W2 tuned | 1831.6 us | 2.0% 개선 |
| W1+W2 tuned + ignore invalid experts | 1567.5 us | 16.1% 개선 |
| W1+W2 tuned + ignore invalid + masked activation | 1590.4 us | 14.9% 개선 |

해석:

```text
ignore-invalid expert assignment는 이번 재현 런에서도 명확히 이겼다.
W1+W2 tuned 위에 추가했을 때 full forward가 1831.6 us -> 1567.5 us로
약 14.4% 더 줄었다.

반면 masked activation은 이번 50-iteration end-to-end 런에서는
1567.5 us -> 1590.4 us로 소폭 악화됐다. 단일 실행 기준으로는
전체 latency에서 안정적인 이득이라고 보기 어렵다.
```

짧은 section profile도 비교했다. 조건은 같은 shape에서 warmup=5/iters=20,
section-profile-warmup=2/section-profile-iters=10이다.

| 설정 | rank0 experts | rank1 experts | rank0 prepare | rank1 prepare |
|---|---:|---:|---:|---:|
| ignore invalid experts | 1028.0 us | 1007.4 us | 195.7 us | 200.3 us |
| ignore invalid + masked activation | 1007.4 us | 1001.3 us | 211.9 us | 190.0 us |

section 해석:

```text
masked activation은 experts 구간만 보면 rank0 약 20.6 us, rank1 약 6.0 us를
줄였다. 하지만 prepare/finalize 변동과 전체 실행 노이즈에 묻혀 end-to-end
latency 개선으로는 안정적으로 나타나지 않았다.

따라서 이번 재현 결과 기준의 추천 경로는
VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS=1 단독이다.
VLLM_MOE_TRITON_EP_MASKED_ACTIVATION=1은 experts 구간 개선은 있으나
전체 latency 효과가 작으므로 보조/후속 실험 후보로 둔다.
```
