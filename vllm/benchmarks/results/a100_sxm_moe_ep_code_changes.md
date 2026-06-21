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

## 14. DeepEP HT direct assignment 성능 측정

`deepep_high_throughput` backend에서 direct assignment 경로를 실제 forward 기준으로
측정했다. 이번 측정은 section profiler가 direct assignment의 raw local expert id와
`DeepEPHTLocalRaw` metadata를 보존하도록 수정한 뒤 수행했다.

환경:

```text
commit=d4968a611 이후 로컬 benchmark_moe_ep_a40.py 수정 포함
GPU=2x NVIDIA A100-SXM4-80GB, NV12
DeepEP=deep-ep==1.1.0+be8053d
DeepEP build=SM80, DISABLE_SM90_FEATURES=1, DISABLE_NVSHMEM=1
vLLM DeepEP detection: has_deep_ep=True, has_deep_ep_v2=False
backend=deepep_high_throughput
VLLM_DEEPEP_HT_NUM_SMS=24
VLLM_DEEPEP_HT_DIRECT_ASSIGNMENT_DEBUG=0
NCCL_P2P_DISABLE=0
```

공통 shape:

```text
world_size=2
hidden=2048
intermediate=768
global experts=128
local experts=64
top_k=8
dtype=BF16
warmup=20
iters=100
independent runs=5
대표 latency=critical_path_us=max(rank0_forward_us, rank1_forward_us)
```

측정 파일:

```text
benchmarks/results/deepep_ht_direct_assignment_20260620_raw.csv
benchmarks/results/deepep_ht_direct_assignment_20260620_summary.csv
benchmarks/results/deepep_ht_direct_assignment_20260620_commands.log
```

CSV/log는 `.gitignore`의 `*.csv`/`*.log` 대상이라 git에는 추가하지 않았다.

end-to-end median critical path:

| tokens | baseline | local-ID | direct assignment | direct + ignore-invalid |
|---:|---:|---:|---:|---:|
| 128 | 1447.1 us | 1443.7 us | 1657.8 us | 1638.7 us |
| 512 | 1722.7 us | 1723.1 us | 1914.7 us | 1915.2 us |
| 1024 | 2054.5 us | 2051.0 us | 2248.0 us | 2102.2 us |
| 2048 | 2717.7 us | 2690.9 us | 2938.9 us | 2739.9 us |

min/max/IQR:

| tokens | 설정 | median | min | max | IQR |
|---:|---|---:|---:|---:|---:|
| 128 | baseline | 1447.1 | 1441.3 | 1491.4 | 25.3 |
| 128 | local-ID | 1443.7 | 1423.5 | 2016.9 | 312.8 |
| 128 | direct | 1657.8 | 1650.7 | 1661.5 | 8.5 |
| 128 | direct + ignore-invalid | 1638.7 | 1627.0 | 1662.0 | 28.6 |
| 512 | baseline | 1722.7 | 1715.7 | 1747.3 | 25.5 |
| 512 | local-ID | 1723.1 | 1695.4 | 2224.8 | 282.8 |
| 512 | direct | 1914.7 | 1896.3 | 2036.1 | 95.8 |
| 512 | direct + ignore-invalid | 1915.2 | 1851.7 | 1981.2 | 85.2 |
| 1024 | baseline | 2054.5 | 1986.9 | 2058.7 | 43.0 |
| 1024 | local-ID | 2051.0 | 2042.3 | 2078.1 | 23.2 |
| 1024 | direct | 2248.0 | 2186.4 | 2267.3 | 51.2 |
| 1024 | direct + ignore-invalid | 2102.2 | 2091.4 | 2125.8 | 27.7 |
| 2048 | baseline | 2717.7 | 2629.3 | 2767.5 | 82.2 |
| 2048 | local-ID | 2690.9 | 2688.7 | 2745.4 | 34.3 |
| 2048 | direct | 2938.9 | 2841.6 | 2966.5 | 112.4 |
| 2048 | direct + ignore-invalid | 2739.9 | 2657.4 | 2793.9 | 81.9 |

해석:

```text
direct assignment 단독은 모든 token size에서 baseline보다 느렸다.
local-ID ablation은 baseline과 거의 동률이거나 약간 빠른 수준이며, direct
assignment의 손실을 설명하지 못한다.

direct + ignore-invalid는 direct 단독의 손실을 줄인다. 특히 tokens=1024에서
2248.0 us -> 2102.2 us로 약 146 us 회복했다. 하지만 baseline 2054.5 us와
local-ID 2051.0 us보다 여전히 느리다.

따라서 현재 A100/DeepEP HT/BF16 synthetic shape에서는 direct assignment를
성능 최적화로 유지할 근거가 없다. benchmark/debug용 경로로 남기거나,
전용 W1/W2 scheduler가 expert compute 구간의 손실을 없애기 전까지는
기본 활성화하지 않는 것이 맞다.
```

tokens=1024 section profile도 확인했다. 표는 rank0/rank1 중 느린 쪽의 평균이다.

조건:

```text
warmup=5
iters=20
section-profile-warmup=2
section-profile-iters=10
```

| 설정 | prepare | experts | finalize | dispatch submit | dispatch receiver | top-k remap | metadata | post quant | combine submit | combine copy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 477.3 us | 1322.1 us | 261.6 us | 227.2 us | 220.6 us | 99.8 us | 53.8 us | 9.5 us | 198.3 us | 30.5 us |
| local-ID | 647.8 us | 1498.4 us | 317.4 us | 251.6 us | 336.0 us | 150.3 us | 73.0 us | 16.1 us | 216.7 us | 46.3 us |
| direct | 418.1 us | 1576.4 us | 294.0 us | 244.0 us | 141.8 us | 11.9 us | 63.3 us | 10.2 us | 227.9 us | 31.6 us |
| direct + ignore-invalid | 396.7 us | 1393.3 us | 255.4 us | 219.9 us | 141.7 us | 12.0 us | 64.4 us | 8.8 us | 191.0 us | 31.4 us |

section 해석:

```text
direct assignment는 receiver prepare를 줄인다. top-k remap은 baseline
99.8 us에서 direct 약 12 us로 줄고, dispatch receiver도 220.6 us에서
141.8 us로 줄었다.

하지만 direct assignment 단독은 experts 구간이 1322.1 us -> 1576.4 us로
약 254 us 늘어 전체 성능을 잃는다. direct + ignore-invalid는 experts 구간을
1393.3 us까지 낮춰 손실을 줄이지만, 여전히 baseline보다 약 71 us 느리다.

결론적으로 현재 direct assignment의 문제는 receiver remap 제거가 아니라
raw local assignment layout 이후의 expert compute/scheduling 효율처럼 보였다.
하지만 이후 Section 15의 assignment/GEMM 분리 benchmark에서 experts 증가의
원인은 W1/W2 GEMM 효율이 아니라 direct assignment builder의 고정 비용으로
확인되었다. 따라서 이 해석은 "전용 W1/W2 scheduler 필요"가 아니라
"DeepEP raw/local route-space를 빠른 generic align 경로에 연결"하는 방향으로
수정한다.
```

## 15. DeepEP HT assignment builder와 prebuilt GEMM 분리 측정

위 section profile의 `experts` 증가는 `_fused_experts()` 전체 시간이다. 즉 direct
assignment schedule 생성, W1, activation, W2, reduce가 모두 섞여 있다. 그래서
아래 진단 벤치를 추가해 schedule 생성과 prebuilt schedule GEMM을 분리했다.

추가한 스크립트:

```text
benchmarks/kernels/benchmark_deepep_ht_assignment_gemm.py
```

측정 조건:

```text
GPU=1x NVIDIA A100-SXM4-80GB on 2x A100 SXM NV12 host
vLLM commit at measurement=dd9f951b98990e31f65b40326eeec039b5eee1ba
torch=2.11.0+cu128
CUDA=12.8
shape=BF16, hidden=2048, intermediate=768, global experts=128,
      local experts=64, top_k=8
tokens=128,512,1024,2048
BLOCK_SIZE_M sweep=16,32,64,128
warmup=10
iters=50
repeats=3
VLLM_DEEPEP_HT_DIRECT_ASSIGNMENT_DEBUG=0
```

실행 명령:

```bash
PYTHONPATH=/tmp/vllm_c_stub:$PYTHONPATH \
VLLM_DEEPEP_HT_DIRECT_ASSIGNMENT_DEBUG=0 \
.venv/bin/python benchmarks/kernels/benchmark_deepep_ht_assignment_gemm.py \
  --output-prefix benchmarks/results/deepep_ht_assignment_gemm_20260620
```

측정 파일:

```text
benchmarks/results/deepep_ht_assignment_gemm_20260620_assignment.csv
benchmarks/results/deepep_ht_assignment_gemm_20260620_components.csv
benchmarks/results/deepep_ht_assignment_gemm_20260620_gemm.csv
benchmarks/results/deepep_ht_assignment_gemm_20260620.metadata.json
benchmarks/results/deepep_ht_assignment_gemm_20260620.commands.log
```

CSV/log/json은 `.gitignore` 대상이라 git에는 추가하지 않았다. 이 벤치는 실제
분산 DeepEP dispatch를 다시 실행하지 않고, 동일한 synthetic global top-k를
DeepEP HT receiver-local raw id와 local sentinel id로 투영해 assignment/GEMM
입력 분포를 고정한다. 따라서 목적은 end-to-end latency가 아니라 병목 분해다.

### A. Assignment-only 결과

actual GEMM config의 `BLOCK_SIZE_M`은 tokens 128/512에서 64, 1024/2048에서
128이었다.

| tokens | BM | generic global | local-ID generic | direct | direct-local | generic ignore | local-ID ignore | direct ignore | direct-local ignore |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 64 | 43.1 | 43.0 | 235.0 | +192.0 | 26.0 | 24.1 | 161.9 | +137.8 |
| 512 | 64 | 43.0 | 43.2 | 232.6 | +189.4 | 24.1 | 25.3 | 162.0 | +136.7 |
| 1024 | 128 | 42.6 | 43.6 | 232.9 | +189.3 | 25.2 | 24.1 | 161.0 | +137.0 |
| 2048 | 128 | 43.1 | 43.2 | 231.8 | +188.6 | 24.5 | 24.6 | 159.9 | +135.4 |

전체 `BLOCK_SIZE_M=16,32,64,128` sweep에서도 같은 패턴이었다.

```text
ignore-invalid=0: direct - local-ID generic = +174.4 ~ +192.0 us
ignore-invalid=1: direct - local-ID generic = +135.4 ~ +140.5 us
```

즉 이전 end-to-end에서 보였던 direct의 약 190~220 us 고정 penalty는 대부분
assignment builder 자체에서 재현된다.

direct helper를 쪼개 잰 isolated component timing은 아래와 같다. 이 값들은
각 component를 따로 반복 측정한 것이므로 정확히 더해지는 end-to-end breakdown은
아니지만, 비용 위치를 보여준다.

| tokens | BM | ignore | counts | prefix_sum | alloc_init | fill_ids | scatter_ids | component sum |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 64 | 0 | 51.6 | 74.8 | 20.1 | 14.5 | 20.6 | 181.6 |
| 128 | 64 | 1 | 0.4 | 74.3 | 20.0 | 14.1 | 20.6 | 129.4 |
| 512 | 64 | 0 | 52.8 | 74.5 | 20.1 | 13.3 | 20.7 | 181.5 |
| 512 | 64 | 1 | 0.4 | 73.6 | 20.1 | 14.6 | 20.4 | 129.1 |
| 1024 | 128 | 0 | 51.1 | 73.6 | 20.2 | 13.3 | 20.6 | 178.8 |
| 1024 | 128 | 1 | 0.4 | 73.5 | 20.1 | 14.6 | 20.5 | 129.0 |
| 2048 | 128 | 0 | 51.0 | 76.9 | 20.9 | 14.7 | 21.2 | 184.7 |
| 2048 | 128 | 1 | 0.4 | 76.1 | 20.6 | 15.3 | 20.9 | 133.3 |

가장 큰 고정 비용은 `prefix_sum` 계열의 작은 PyTorch GPU op 누적이고,
`ignore-invalid=0`에서는 invalid count 생성을 위한 `counts.sum`, scalar 생성,
`torch.cat` 경로가 약 51 us 추가된다. allocation/init와 두 Triton kernel도 각각
작지만 고정 비용으로 누적된다.

### B. Prebuilt schedule GEMM 결과

schedule을 timing loop 밖에서 한 번만 만든 뒤 W1, activation, W2, reduce,
full expert path를 측정했다. 아래 표는 `W1 + activation + W2 + reduce` 전체다.

| tokens | ignore | generic global | local-ID generic | direct | direct-local |
|---:|---:|---:|---:|---:|---:|
| 128 | 0 | 409.2 | 425.4 | 411.7 | -13.8 |
| 128 | 1 | 398.0 | 397.1 | 397.6 | +0.6 |
| 512 | 0 | 448.7 | 448.4 | 448.3 | -0.1 |
| 512 | 1 | 437.6 | 436.5 | 436.5 | -0.0 |
| 1024 | 0 | 591.0 | 584.6 | 584.6 | -0.0 |
| 1024 | 1 | 567.9 | 566.1 | 567.5 | +1.4 |
| 2048 | 0 | 830.9 | 818.8 | 818.7 | -0.1 |
| 2048 | 1 | 768.3 | 765.3 | 765.1 | -0.1 |

512 tokens 이상에서는 prebuilt direct가 local-ID generic과 사실상 같다. 즉 direct
schedule의 내부 순서나 block 구성이 W1/W2 GEMM을 느리게 만든다는 증거는 없다.
1024/2048에서 generic global이 조금 느린 것은 global invalid expert들을 더 많이
schedule에 남기는 구조 차이와 일치한다.

대표 phase breakdown:

| tokens | ignore | phase | generic global | local-ID generic | direct |
|---:|---:|---|---:|---:|---:|
| 1024 | 0 | W1 | 339.4 | 339.6 | 339.3 |
| 1024 | 0 | W2 | 204.2 | 198.2 | 197.5 |
| 1024 | 0 | reduce | 29.7 | 29.7 | 29.6 |
| 1024 | 1 | W1 | 333.1 | 333.1 | 333.3 |
| 1024 | 1 | W2 | 184.3 | 183.9 | 185.6 |
| 1024 | 1 | reduce | 33.2 | 34.6 | 34.4 |
| 2048 | 0 | W1 | 451.5 | 446.4 | 446.1 |
| 2048 | 0 | W2 | 285.4 | 278.3 | 278.4 |
| 2048 | 0 | reduce | 53.3 | 53.3 | 53.3 |
| 2048 | 1 | W1 | 433.4 | 432.8 | 432.6 |
| 2048 | 1 | W2 | 259.8 | 257.9 | 257.9 |
| 2048 | 1 | reduce | 34.4 | 33.8 | 34.7 |

### C. Schedule 구조 비교

local-ID generic과 direct는 모든 actual BM 조건에서 동일한 schedule 구조를 만든다.
generic global만 off-rank expert들을 global expert space에 따로 padding해서 더 큰
schedule이 된다.

| tokens | BM | ignore | generic post | generic valid/invalid blocks | local/direct post | local/direct valid/invalid blocks |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 64 | 0 | 8128 | 63 / 64 | 4544 | 63 / 8 |
| 128 | 64 | 1 | 4032 | 63 / 0 | 4032 | 63 / 0 |
| 512 | 64 | 0 | 8192 | 64 / 64 | 6208 | 64 / 33 |
| 512 | 64 | 1 | 4096 | 64 / 0 | 4096 | 64 / 0 |
| 1024 | 128 | 0 | 16384 | 64 / 64 | 12416 | 64 / 33 |
| 1024 | 128 | 1 | 8192 | 64 / 0 | 8192 | 64 / 0 |
| 2048 | 128 | 0 | 24320 | 98 / 92 | 20736 | 98 / 64 |
| 2048 | 128 | 1 | 12544 | 98 / 0 | 12544 | 98 / 0 |

### 결론 수정

이번 분리 측정 결과, 이전 section profile의 `experts` 증가를 W1/W2 GEMM
효율 저하로 해석하면 안 된다.

```text
prebuilt direct schedule GEMM ~= local-ID generic GEMM
direct assignment builder ~= local-ID generic align + 135~192 us
```

따라서 다음 우선순위는 새 W1/W2 scheduler가 아니라 direct assignment builder의
고정 비용 제거다.

구체적으로는 다음이 더 유효하다.

```text
1. expert count 확장, padded count, prefix sum, expert_offsets 생성을
   하나의 CUDA/Triton helper로 fusion
2. sorted_token_ids, expert_ids, write cursor buffer를 재사용하거나 workspace에서
   받아 allocation/init 비용 제거
3. include-invalid 경로의 counts.sum + torch.cat 제거
4. fill_expert_ids와 scatter_token_ids를 하나의 schedule-build kernel로 합치거나,
   적어도 host launch 수를 줄이기
```

`ignore-invalid`는 여전히 긍정적인 방향이다. builder 고정 비용이 제거되면 큰
prefill에서는 local/direct route-space의 작은 schedule과 invalid block 제거가
end-to-end win으로 이어질 가능성이 높다.

## 16. DeepEP HT generic ignore-invalid missing 조건 측정

Section 14의 end-to-end 표에는 다음 두 조건이 빠져 있었다.

```text
global generic + ignore-invalid
local-ID generic + ignore-invalid
```

Section 15에서 `local-ID generic + ignore-invalid`가 direct+ignore와 같은
schedule을 약 24 us에 만든다는 것을 확인했으므로, direct builder 최적화보다
이 두 조건의 실제 2-GPU forward 성능을 먼저 측정했다.

측정 전 환경 메모:

```text
uv run ruff 시도 후 CUDA 13 runtime wheel들이 venv에 섞여 PyNccl이
"CUDA driver version is insufficient for CUDA runtime version"으로 실패했다.
cu13 wheel을 제거하고 cu12 wheel을 재설치해 torch/NCCL/FlashInfer import를
복구한 뒤 측정했다.

torch=2.11.0+cu128
CUDA driver=570.172.08
nvidia-smi CUDA=12.8
nvidia-nccl-cu12=2.28.9
nvidia-cuda-runtime-cu12=12.9.79
cuda-python=12.9.7
cuda-bindings=12.9.7
cuda-tile=1.4.0
```

측정 조건:

```text
commit=acef97424
GPU=2x NVIDIA A100-SXM4-80GB, NV12
backend=deepep_high_throughput
world_size=2
hidden=2048
intermediate=768
global experts=128
local experts=64
top_k=8
dtype=BF16
warmup=20
iters=100
independent runs=5
VLLM_DEEPEP_HT_NUM_SMS=24
VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS=1
VLLM_DEEPEP_HT_DIRECT_ASSIGNMENT=0
VLLM_DEEPEP_HT_DIRECT_ASSIGNMENT_DEBUG=0
NCCL_P2P_DISABLE=0
```

조건별 추가 flag:

```text
global generic + ignore:
  VLLM_DEEPEP_HT_LOCAL_EXPERT_IDS=0

local-ID generic + ignore:
  VLLM_DEEPEP_HT_LOCAL_EXPERT_IDS=1
```

측정 파일:

```text
benchmarks/results/deepep_ht_ignore_missing_20260620_raw.csv
benchmarks/results/deepep_ht_ignore_missing_20260620_commands.log
```

CSV/log는 `.gitignore` 대상이라 git에는 추가하지 않았다.

새로 측정한 missing 조건의 critical-path median:

| tokens | global generic + ignore | local-ID generic + ignore |
|---:|---:|---:|
| 128 | 1441.9 us | 1439.8 us |
| 512 | 1711.8 us | 1701.8 us |
| 1024 | 1941.6 us | 1930.1 us |
| 2048 | 2605.3 us | 2615.7 us |

min/max/IQR:

| tokens | 설정 | median | min | max | IQR |
|---:|---|---:|---:|---:|---:|
| 128 | global + ignore | 1441.9 | 1425.0 | 1478.7 | 23.6 |
| 128 | local-ID + ignore | 1439.8 | 1430.3 | 1456.2 | 13.2 |
| 512 | global + ignore | 1711.8 | 1672.2 | 1714.2 | 11.3 |
| 512 | local-ID + ignore | 1701.8 | 1633.8 | 1727.8 | 55.5 |
| 1024 | global + ignore | 1941.6 | 1934.3 | 1983.0 | 28.1 |
| 1024 | local-ID + ignore | 1930.1 | 1908.1 | 1978.8 | 23.1 |
| 2048 | global + ignore | 2605.3 | 2496.8 | 2636.9 | 91.1 |
| 2048 | local-ID + ignore | 2615.7 | 2445.4 | 2631.5 | 124.6 |

Section 14의 baseline/direct+ignore medians와 나란히 보면 다음과 같다.
baseline과 direct+ignore는 Section 14의 기존 5-run median이고, global/local
ignore는 이번 새 측정이다.

| tokens | baseline | global + ignore | local-ID + ignore | direct + ignore |
|---:|---:|---:|---:|---:|
| 128 | 1447.1 | 1441.9 | 1439.8 | 1638.7 |
| 512 | 1722.7 | 1711.8 | 1701.8 | 1915.2 |
| 1024 | 2054.5 | 1941.6 | 1930.1 | 2102.2 |
| 2048 | 2717.7 | 2605.3 | 2615.7 | 2739.9 |

baseline 대비:

| tokens | global + ignore | local-ID + ignore | direct + ignore |
|---:|---:|---:|---:|
| 128 | -5.2 us (-0.4%) | -7.3 us (-0.5%) | +191.6 us (+13.2%) |
| 512 | -10.9 us (-0.6%) | -20.9 us (-1.2%) | +192.5 us (+11.2%) |
| 1024 | -112.9 us (-5.5%) | -124.4 us (-6.1%) | +47.7 us (+2.3%) |
| 2048 | -112.4 us (-4.1%) | -102.0 us (-3.8%) | +22.2 us (+0.8%) |

해석:

```text
1. direct+ignore가 느린 이유는 direct schedule 품질이 아니라 direct builder
   고정 비용이라는 Section 15 결론이 end-to-end에서도 확인됐다.

2. global generic + ignore와 local-ID generic + ignore는 모두 direct+ignore보다
   훨씬 빠르다.

3. 1024/2048에서는 ignore-invalid generic 경로가 기존 baseline보다 4~6%
   빠르다. 작은 token에서는 baseline과 거의 같은 수준이다.

4. local-ID generic + ignore는 128/512/1024에서 global+ignore보다 약간 빠르고,
   2048에서는 noise 범위에서 global+ignore가 약간 빠르다. 둘의 차이는
   direct builder penalty와 비교하면 작다.
```

따라서 다음 우선순위는 direct Triton builder 확장이 아니다. 현재 가장 좋은
실용 경로는 다음이다.

```text
DeepEP HT + generic moe_align_block_size + ignore-invalid
```

그중 receiver remap까지 줄이려면 다음 단계가 맞다.

```text
1. 기존 CUDA moe_align_block_size에 raw local -1 skip mode 추가
2. DeepEP HT receiver에서 -1 -> local sentinel remap을 생략
3. raw local IDs를 generic align custom op가 직접 histogram/align
4. 1024-token 부근부터 runtime threshold로 활성화 여부 검증
```

## 17. DeepEP HT generic ignore-invalid same-session paired 재측정

Section 16은 missing 조건을 채우는 데 충분했지만, baseline/direct+ignore와
측정 세션이 달랐다. 이번에는 같은 세션에서 baseline, global-ignore,
local-ID-ignore를 cycle별로 섞어 재측정했다.

추가로 벤치마크 스크립트에 다음 필드를 기록했다.

```text
input_tokens
received_tokens_rank0/rank1
ep_ignore_enabled_rank0/rank1
valid_route_pairs_rank0/rank1
invalid_route_pairs_rank0/rank1
assignment_stats_rank0/rank1
```

`assignment_stats` 형식은 다음과 같다.

```text
block_m:num_tokens_post_padded/valid_blocks/invalid_blocks
```

측정 조건:

```text
base_commit=b439cdf47
GPU=2x NVIDIA A100-SXM4-80GB, driver=570.172.08, 81920 MiB each
torch=2.11.0+cu128
CUDA=12.8
backend=deepep_high_throughput
world_size=2
hidden=2048
intermediate=768
global experts=128
local experts=64
top_k=8
dtype=BF16
warmup=20
iters=100
cycles=10
VLLM_DEEPEP_HT_NUM_SMS=24
VLLM_DEEPEP_HT_DIRECT_ASSIGNMENT=0
VLLM_DEEPEP_HT_DIRECT_ASSIGNMENT_DEBUG=0
NCCL_P2P_DISABLE=0
```

cycle 순서:

```text
cycle 1: baseline -> global-ignore -> local-ID-ignore
cycle 2: local-ID-ignore -> global-ignore -> baseline
cycle 3: global-ignore -> baseline -> local-ID-ignore
위 순서를 10 cycles까지 반복
```

설정별 flag:

```text
baseline:
  VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS=0
  VLLM_DEEPEP_HT_LOCAL_EXPERT_IDS=0

global-ignore:
  VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS=1
  VLLM_DEEPEP_HT_LOCAL_EXPERT_IDS=0

local-ID-ignore:
  VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS=1
  VLLM_DEEPEP_HT_LOCAL_EXPERT_IDS=1
```

측정 파일:

```text
benchmarks/results/deepep_ht_paired_ignore_20260620_raw.csv
benchmarks/results/deepep_ht_paired_ignore_20260620_commands.log
benchmarks/results/analyze_deepep_ht_paired.py
```

CSV/log는 원래 `.gitignore` 대상이지만, paired difference 재계산을 위해
`git add -f`로 커밋했다. 분석 스크립트는 raw CSV에서 아래 표를 재생성한다.
CSV는 header 포함 121줄, 즉 10 cycles x 4 token sizes x 3 settings =
120 rows이며 누락은 없다.

```text
vllm/.venv/bin/python \
  vllm/benchmarks/results/analyze_deepep_ht_paired.py
```

아래 latency 값은 각 run 안에서 `iters=100` 전체 CUDA event 시간을 100으로
나눈 steady-state 평균이고, 표의 median은 이 run-level 평균 10개의 median이다.

critical path 절대값:

| tokens | baseline median us (IQR/min/max) | global-ignore median us (IQR/min/max) | local-ID-ignore median us (IQR/min/max) |
|---:|---:|---:|---:|
| 128 | 1451.0 (18.8/1423.1/1482.8) | 1447.9 (7.9/1434.5/1520.3) | 1458.0 (28.1/1427.6/2076.2) |
| 512 | 1728.4 (16.3/1712.1/1756.4) | 1711.1 (20.5/1690.3/1734.4) | 1711.1 (31.2/1690.4/1758.8) |
| 1024 | 2042.1 (31.6/2003.7/2075.6) | 1957.8 (16.5/1909.1/2001.5) | 1954.6 (25.6/1927.2/1980.6) |
| 2048 | 2765.7 (45.8/2638.1/2789.3) | 2618.2 (26.5/2503.5/2784.5) | 2610.6 (69.5/2469.2/2777.7) |

같은 `(cycle, tokens)` 안에서 baseline을 뺀 paired 차이:

| tokens | global-ignore - baseline | local-ID-ignore - baseline | local-ID - global |
|---:|---:|---:|---:|
| 128 | -0.6 us (-0.04%, IQR 27.7, min/max -48.3/+68.2) | +5.0 us (+0.34%, IQR 54.0, min/max -26.2/+626.4) | +5.4 us (IQR 27.9, min/max -16.0/+629.9) |
| 512 | -20.8 us (-1.20%, IQR 23.1, min/max -47.7/-0.5) | -21.1 us (-1.22%, IQR 29.8, min/max -39.1/+8.8) | +3.5 us (IQR 18.9, min/max -26.8/+46.4) |
| 1024 | -80.9 us (-3.96%, IQR 27.6, min/max -101.4/-50.6) | -77.6 us (-3.80%, IQR 35.0, min/max -123.4/-40.4) | -0.6 us (IQR 32.4, min/max -49.3/+50.7) |
| 2048 | -137.2 us (-4.96%, IQR 55.6, min/max -266.9/-4.8) | -142.6 us (-5.16%, IQR 28.9, min/max -278.7/-11.6) | -9.0 us (IQR 42.1, min/max -85.5/+127.6) |

ignore-invalid 실제 활성 여부:

| tokens | setting | rank0 true/total | rank1 true/total |
|---:|---|---:|---:|
| 128 | baseline | 0/10 | 0/10 |
| 128 | global-ignore | 0/10 | 0/10 |
| 128 | local-ID-ignore | 0/10 | 0/10 |
| 512 | baseline | 0/10 | 0/10 |
| 512 | global-ignore | 0/10 | 10/10 |
| 512 | local-ID-ignore | 0/10 | 10/10 |
| 1024 | baseline | 0/10 | 0/10 |
| 1024 | global-ignore | 10/10 | 10/10 |
| 1024 | local-ID-ignore | 10/10 | 10/10 |
| 2048 | baseline | 0/10 | 0/10 |
| 2048 | global-ignore | 10/10 | 10/10 |
| 2048 | local-ID-ignore | 10/10 | 10/10 |

critical rank와 rank별 latency median:

| tokens | setting | critical rank r0/r1/tie | rank0 median us | rank1 median us | recv r0/r1 median |
|---:|---|---:|---:|---:|---:|
| 128 | baseline | 5/5/0 | 1451.0 | 1450.5 | 256/256 |
| 128 | global-ignore | 5/5/0 | 1447.9 | 1447.6 | 256/256 |
| 128 | local-ID-ignore | 2/8/0 | 1458.0 | 1457.5 | 256/256 |
| 512 | baseline | 6/4/0 | 1727.8 | 1728.4 | 1016/1024 |
| 512 | global-ignore | 5/5/0 | 1710.8 | 1710.9 | 1016/1024 |
| 512 | local-ID-ignore | 6/4/0 | 1711.1 | 1710.7 | 1016/1024 |
| 1024 | baseline | 6/4/0 | 2042.1 | 2041.5 | 2044/2044 |
| 1024 | global-ignore | 6/4/0 | 1957.3 | 1957.8 | 2044/2044 |
| 1024 | local-ID-ignore | 5/5/0 | 1954.3 | 1954.2 | 2044/2044 |
| 2048 | baseline | 4/6/0 | 2765.2 | 2765.7 | 4078/4080 |
| 2048 | global-ignore | 6/4/0 | 2618.0 | 2618.0 | 4078/4080 |
| 2048 | local-ID-ignore | 5/5/0 | 2610.6 | 2610.2 | 4078/4080 |

rank별 received tokens와 route pair 통계 median:

| input tokens | received tokens r0/r1 | valid route pairs r0/r1 | invalid route pairs r0/r1 |
|---:|---:|---:|---:|
| 128 | 256/256 | 1032/1016 | 1016/1032 |
| 512 | 1016/1024 | 4032/4160 | 4096/4032 |
| 1024 | 2044/2044 | 8236/8148 | 8116/8204 |
| 2048 | 4078/4080 | 16310/16458 | 16314/16182 |

대표 assignment stats:

| tokens | baseline r0/r1 | ignore r0/r1 |
|---:|---|---|
| 128 | `64:5120/64/16` / `64:5184/64/17` | `64:5120/64/16` / `64:5184/64/17` |
| 512 | `128:12288/64/32` / `128:12288/64/32` | `128:12288/64/32` / `128:8192/64/0` |
| 1024 | `128:20736/98/64` / `128:19968/91/65` | `128:12544/98/0` / `128:11648/91/0` |
| 2048 | `128:36352/156/128` / `128:36864/161/127` | `128:19968/156/0` / `128:20608/161/0` |

해석:

```text
1. 128 tokens에서는 ignore env를 켜도 runtime threshold 때문에 실제 ignore
   경로가 켜지지 않는다. 성능 차이도 noise다.

2. 512 tokens에서는 rank1만 ignore가 켜진다. paired로 약 -21 us, -1.2%가
   보이지만 partial activation 조건이므로 큰 결론은 내리지 않는다.

3. 1024/2048 tokens에서는 양 rank 모두 ignore가 켜진다. paired 기준
   global-ignore는 -3.96%/-4.96%, local-ID-ignore는 -3.80%/-5.16%로
   baseline보다 안정적으로 빠르다.

4. global-ignore와 local-ID-ignore의 차이는 작고 방향이 일정하지 않다.
   local-ID가 더 빠르다고 결론낼 근거는 없다. 이번 결과에서 유의미한
   효과는 route-space가 아니라 invalid block을 schedule에서 제거하는
   ignore-invalid 경로다.
```

따라서 Section 16의 결론은 다음처럼 보강한다.

```text
현재 최선의 실용 경로:
DeepEP HT + generic moe_align_block_size + ignore-invalid

추가 direct Triton builder 개발은 우선순위가 낮다.
다음 최적화 후보는 generic CUDA align에 raw -1 skip mode를 넣어
receiver의 -1 -> local sentinel remap 비용을 줄이는 것이다.
```

## 18. Rank-distinct input seed와 512-token threshold sweep

Section 17의 paired 재측정은 같은 rank input/routing을 반복한 workload였다.
이번에는 weight seed는 고정하고 rank별 hidden/router seed를 다르게 주는
benchmark 옵션을 추가했다.

추가된 benchmark 옵션과 필드:

```text
--rank-distinct-inputs
--input-seed-base
rank_distinct_inputs
weight_seed
input_seed_base
input_seed_rank0/rank1
ep_ignore_num_tokens_rank0/rank1
```

또한 512-token cliff를 직접 sweep하기 위해 다음 env를 추가했다. 기본값은
기존 동작과 같은 `1024`다.

```text
VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS_MIN_TOKENS=1024
```

### A. Rank-distinct input/routing seed 재측정

측정 조건:

```text
GPU=2x NVIDIA A100-SXM4-80GB
backend=deepep_high_throughput
world_size=2
hidden=2048
intermediate=768
global experts=128
local experts=64
top_k=8
dtype=BF16
warmup=20
iters=100
weight_seed=7
input_seed_base=1007,2007,3007,4007,5007
rank0_input_seed=input_seed_base
rank1_input_seed=input_seed_base + 1
cycles_per_seed=3
tokens=1024,2048
VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS_MIN_TOKENS=1024
VLLM_DEEPEP_HT_NUM_SMS=24
VLLM_DEEPEP_HT_DIRECT_ASSIGNMENT=0
NCCL_P2P_DISABLE=0
```

측정 파일:

```text
benchmarks/results/deepep_ht_rank_distinct_seed_20260620_raw.csv
benchmarks/results/deepep_ht_rank_distinct_seed_20260620_commands.log
```

CSV는 header 포함 91줄, 즉 5 input seeds x 3 cycles x 2 token sizes x
3 settings = 90 rows이며 누락은 없다.

critical path 절대값:

| tokens | setting | median | IQR | min | max |
|---:|---|---:|---:|---:|---:|
| 1024 | baseline | 2036.7 | 32.4 | 1995.2 | 2076.1 |
| 1024 | global-ignore | 1962.7 | 21.2 | 1933.1 | 1992.7 |
| 1024 | local-ID-ignore | 1961.2 | 23.5 | 1921.9 | 1994.3 |
| 2048 | baseline | 2733.6 | 82.0 | 2632.4 | 2982.4 |
| 2048 | global-ignore | 2613.5 | 32.9 | 2541.6 | 2665.9 |
| 2048 | local-ID-ignore | 2596.2 | 81.5 | 2474.6 | 2656.9 |

같은 `(input_seed, cycle, tokens)` 안에서 baseline을 뺀 paired 차이:

| tokens | setting | median delta | pct | IQR | min | max | wins |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1024 | global-ignore | -77.4 | -3.80% | 30.7 | -124.1 | -30.1 | 15/15 |
| 1024 | local-ID-ignore | -72.2 | -3.55% | 27.5 | -104.8 | -22.0 | 15/15 |
| 2048 | global-ignore | -117.5 | -4.30% | 109.1 | -369.1 | -20.3 | 15/15 |
| 2048 | local-ID-ignore | -147.9 | -5.41% | 109.9 | -386.3 | -11.1 | 15/15 |

seed별 paired median delta:

| tokens | input_seed_base | global-ignore | local-ID-ignore |
|---:|---:|---:|---:|
| 1024 | 1007 | -88.7 | -72.2 |
| 1024 | 2007 | -61.2 | -85.6 |
| 1024 | 3007 | -79.1 | -72.2 |
| 1024 | 4007 | -78.9 | -99.4 |
| 1024 | 5007 | -44.1 | -71.2 |
| 2048 | 1007 | -168.1 | -213.9 |
| 2048 | 2007 | -179.8 | -129.7 |
| 2048 | 3007 | -56.8 | -93.1 |
| 2048 | 4007 | -83.9 | -133.5 |
| 2048 | 5007 | -114.1 | -240.7 |

ignore 활성 여부:

| tokens | setting | rank0 true/total | rank1 true/total | num_tokens r0/r1 median |
|---:|---|---:|---:|---:|
| 1024 | baseline | 0/15 | 0/15 | 2040/2041 |
| 1024 | global-ignore | 15/15 | 15/15 | 2040/2041 |
| 1024 | local-ID-ignore | 15/15 | 15/15 | 2040/2041 |
| 2048 | baseline | 0/15 | 0/15 | 4085/4083 |
| 2048 | global-ignore | 15/15 | 15/15 | 4085/4083 |
| 2048 | local-ID-ignore | 15/15 | 15/15 | 4085/4083 |

해석:

```text
1. rank별 input/routing을 다르게 해도 1024/2048에서 ignore-invalid 효과는
   유지된다.

2. 5개의 독립적인 rank-distinct routing seed에서 모두 개선됐고, 각 seed를
   3회 반복한 총 15회의 paired run에서 global-ignore와 local-ID-ignore가
   모두 baseline보다 빨랐다.

3. global-ignore와 local-ID-ignore의 차이는 여전히 일관되지 않다. 핵심
   효과는 local route-space가 아니라 invalid block 제거다. 포트폴리오
   대표 결과는 더 단순한 global generic + ignore-invalid로 잡는 편이
   안전하다.
```

### B. 512-token threshold sweep

측정 조건:

```text
tokens=512
weight_seed=7
input_seed_base=1007,2007,3007,4007,5007
threshold=512,768,896,1024,1280
cycles=1
warmup=20
iters=100
```

측정 파일:

```text
benchmarks/results/deepep_ht_threshold_sweep_20260620_raw.csv
benchmarks/results/deepep_ht_threshold_sweep_20260620_commands.log
```

CSV는 header 포함 76줄, 즉 5 thresholds x 5 input seeds x 3 settings =
75 rows이며 누락은 없다.

threshold별 paired delta:

| threshold | setting | median delta | pct | IQR | min | max | wins |
|---:|---|---:|---:|---:|---:|---:|---:|
| 512 | global-ignore | -53.9 | -3.11% | 14.7 | -64.9 | -37.1 | 5/5 |
| 512 | local-ID-ignore | -48.1 | -2.78% | 17.6 | -71.1 | -19.0 | 5/5 |
| 768 | global-ignore | -50.0 | -2.92% | 3.3 | -73.9 | -26.8 | 5/5 |
| 768 | local-ID-ignore | -47.8 | -2.79% | 32.8 | -70.4 | -2.3 | 5/5 |
| 896 | global-ignore | -61.6 | -3.57% | 460.6 | -74.0 | +525.0 | 3/5 |
| 896 | local-ID-ignore | -41.5 | -2.41% | 26.9 | -74.4 | +298.1 | 4/5 |
| 1024 | global-ignore | -8.1 | -0.47% | 12.9 | -15.6 | +27.2 | 3/5 |
| 1024 | local-ID-ignore | -11.0 | -0.64% | 32.3 | -35.7 | +433.6 | 4/5 |
| 1280 | global-ignore | -7.6 | -0.44% | 17.0 | -26.6 | +9.8 | 3/5 |
| 1280 | local-ID-ignore | -5.2 | -0.30% | 9.4 | -23.8 | -1.1 | 5/5 |

threshold별 ignore 활성 여부:

| threshold | setting | rank0 true/total | rank1 true/total | num_tokens r0/r1 median |
|---:|---|---:|---:|---:|
| 512 | global-ignore | 5/5 | 5/5 | 1020/1019 |
| 512 | local-ID-ignore | 5/5 | 5/5 | 1020/1019 |
| 768 | global-ignore | 5/5 | 5/5 | 1020/1019 |
| 768 | local-ID-ignore | 5/5 | 5/5 | 1020/1019 |
| 896 | global-ignore | 5/5 | 5/5 | 1020/1019 |
| 896 | local-ID-ignore | 5/5 | 5/5 | 1020/1019 |
| 1024 | global-ignore | 1/5 | 0/5 | 1020/1019 |
| 1024 | local-ID-ignore | 1/5 | 0/5 | 1020/1019 |
| 1280 | global-ignore | 0/5 | 0/5 | 1020/1019 |
| 1280 | local-ID-ignore | 0/5 | 0/5 | 1020/1019 |

input seed별 512-token received tokens:

| input_seed_base | received r0/r1 |
|---:|---:|
| 1007 | 1024/1017 |
| 2007 | 1021/1019 |
| 3007 | 1020/1021 |
| 4007 | 1019/1018 |
| 5007 | 1018/1020 |

해석 수정:

```text
1. 이 sweep은 threshold=512/768/896의 kernel 성능 차이를 비교한 것이 아니다.
   512 input tokens에서 실제 received tokens는 약 1017~1024였으므로,
   threshold=512/768/896은 모두 양 rank에서 같은 ignore path를 실행한다.
   세 값 사이의 차이는 threshold 효과가 아니라 실행 노이즈다.

2. threshold=1024는 rank0 1/5만 켜지고 rank1은 0/5만 켜지는 비대칭
   partial activation 조건이다. threshold=1280은 양 rank 모두 꺼진다.
   따라서 1024/1280 결과는 mostly baseline/off 경로에 가깝다.

3. 896 global/local-ID와 1024 local-ID에는 여러 transient outlier가 있다.
   seed당 cycle=1이고 실행 순서도 baseline -> global -> local로 고정됐으므로
   순서/thermal/background noise를 분리하기 어렵다.

4. 다음 실험은 threshold 값 자체가 아니라 실제 received-M을 변화시키면서
   ignore OFF와 forced ON을 paired로 비교해 break-even 지점을 찾는 것이다.
```

## 19. Received-M break-even: ignore OFF vs forced ON

Section 18B의 threshold sweep은 `threshold=512/768/896`이 모두 같은 실행
경로를 켜는 조건이었기 때문에 threshold 값 자체의 비교가 아니었다. 이번에는
실제 received token 수를 바꾸면서 global generic ignore-invalid를 강제로 켰다.

분석 스크립트도 기존 paired CSV뿐 아니라 rank-distinct, threshold sweep,
break-even CSV를 모두 처리하도록 확장했다.

```text
benchmarks/results/analyze_deepep_ht_paired.py
```

스크립트는 다음 key를 자동으로 사용한다.

```text
basic paired:
  (cycle, tokens, setting)

rank-distinct:
  (input_seed_group, cycle, tokens, setting)

threshold sweep:
  (threshold, input_seed_group, cycle, tokens, setting)
```

paired table에는 다음을 함께 출력한다.

```text
min(received_tokens_rank0, received_tokens_rank1)
critical rank의 received tokens
median(delta)
median(delta) / median(baseline)
median(pairwise delta / baseline)
wins / total
positive delta outlier rows
rank별 activation
```

측정 조건:

```text
base_commit=5b2044614
GPU=2x NVIDIA A100-SXM4-80GB
backend=deepep_high_throughput
world_size=2
hidden=2048
intermediate=768
global experts=128
local experts=64
top_k=8
dtype=BF16
warmup=20
iters=100
weight_seed=7
input_seed_base=1007,2007,3007,4007,5007
rank0_input_seed=input_seed_base
rank1_input_seed=input_seed_base + 1
cycles_per_seed=3
tokens=128,192,256,320,384,448,512,640,768
VLLM_DEEPEP_HT_NUM_SMS=24
VLLM_DEEPEP_HT_LOCAL_EXPERT_IDS=0
VLLM_DEEPEP_HT_DIRECT_ASSIGNMENT=0
NCCL_P2P_DISABLE=0
```

설정:

```text
baseline:
  VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS=0

global-ignore forced ON:
  VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS=1
  VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS_MIN_TOKENS=0
```

측정 파일:

```text
benchmarks/results/deepep_ht_break_even_m_20260620_raw.csv
benchmarks/results/deepep_ht_break_even_m_20260620_commands.log
```

CSV는 header 포함 271줄, 즉 5 seeds x 3 cycles x 9 token sizes x
2 settings = 270 rows이며 누락은 없다.

critical path 절대값:

| tokens | baseline median | forced ON median | min/max baseline | min/max forced ON |
|---:|---:|---:|---:|---:|
| 128 | 1446.3 | 1410.4 | 1437.4/1517.6 | 1398.0/1566.5 |
| 192 | 1475.4 | 1429.2 | 1443.9/1651.4 | 1412.3/1507.3 |
| 256 | 1496.1 | 1452.0 | 1479.1/2075.1 | 1429.6/1544.1 |
| 320 | 1671.0 | 1629.8 | 1648.9/1681.6 | 1599.4/1749.7 |
| 384 | 1694.7 | 1636.2 | 1671.1/1753.3 | 1618.5/1675.6 |
| 448 | 1715.9 | 1651.1 | 1691.9/2307.1 | 1636.3/2167.9 |
| 512 | 1732.9 | 1669.0 | 1713.1/1773.2 | 1651.9/1702.9 |
| 640 | 1762.8 | 1681.6 | 1704.6/2287.1 | 1670.6/1887.3 |
| 768 | 1805.0 | 1712.5 | 1765.2/1874.3 | 1692.7/2221.1 |

paired delta. `delta/median baseline`은 `median(delta) / median(baseline)`,
`pair pct`는 각 pair의 `(ON - OFF) / OFF`를 계산한 뒤 median을 낸 값이다.

| input tokens | min recv median | critical recv median | ON-OFF median | pair pct | IQR | wins |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 254 | 255 | -32.3 | -2.24% | 18.9 | 14/15 |
| 192 | 382 | 383 | -44.5 | -3.04% | 26.3 | 13/15 |
| 256 | 509 | 510 | -44.1 | -2.97% | 33.4 | 13/15 |
| 320 | 637 | 638 | -43.6 | -2.61% | 16.4 | 14/15 |
| 384 | 765 | 765 | -55.3 | -3.27% | 21.9 | 15/15 |
| 448 | 890 | 892 | -53.6 | -3.12% | 36.4 | 13/15 |
| 512 | 1018 | 1019 | -67.5 | -3.92% | 26.7 | 15/15 |
| 640 | 1273 | 1274 | -82.0 | -4.67% | 31.5 | 14/15 |
| 768 | 1528 | 1529 | -97.1 | -5.39% | 39.2 | 14/15 |

seed별 paired median delta:

| tokens | 1007 | 2007 | 3007 | 4007 | 5007 |
|---:|---:|---:|---:|---:|---:|
| 128 | -26.2 | -28.1 | -32.3 | -35.1 | -44.7 |
| 192 | -60.8 | -53.6 | -38.1 | +18.4 | -58.1 |
| 256 | -34.0 | -41.9 | -44.1 | -56.3 | -64.6 |
| 320 | -50.3 | -36.1 | -42.5 | -46.3 | -29.4 |
| 384 | -69.7 | -59.1 | -45.2 | -42.6 | -55.3 |
| 448 | -65.1 | -45.2 | -53.6 | -45.8 | -52.5 |
| 512 | -54.2 | -67.5 | -47.9 | -77.3 | -70.3 |
| 640 | -82.0 | -67.0 | -85.3 | -89.9 | -72.9 |
| 768 | -103.7 | -53.7 | -99.4 | -64.7 | -125.7 |

activation sanity:

| tokens | baseline active r0/r1 | forced ON active r0/r1 | recv r0/r1 median |
|---:|---:|---:|---:|
| 128 | 0/15, 0/15 | 15/15, 15/15 | 255/255 |
| 192 | 0/15, 0/15 | 15/15, 15/15 | 383/383 |
| 256 | 0/15, 0/15 | 15/15, 15/15 | 510/509 |
| 320 | 0/15, 0/15 | 15/15, 15/15 | 638/637 |
| 384 | 0/15, 0/15 | 15/15, 15/15 | 766/765 |
| 448 | 0/15, 0/15 | 15/15, 15/15 | 892/892 |
| 512 | 0/15, 0/15 | 15/15, 15/15 | 1020/1019 |
| 640 | 0/15, 0/15 | 15/15, 15/15 | 1276/1274 |
| 768 | 0/15, 0/15 | 15/15, 15/15 | 1532/1529 |

대표 assignment stats:

| tokens | baseline mode | forced ON mode |
|---:|---|---|
| 128 | `64:5120/64/16` / `64:5184/64/17` | `64:4096/64/0` / `64:4096/64/0` |
| 192 | `64:5632/64/24` / `64:5632/64/24` | `64:4096/64/0` / `64:4096/64/0` |
| 256 | `64:6144/64/32` / `64:6144/64/32` | `64:4096/64/0` / `64:4096/64/0` |
| 320 | `128:10752/64/20` / `128:10752/64/20` | `128:8192/64/0` / `128:8192/64/0` |
| 384 | `128:11264/64/24` / `128:11264/64/24` | `128:8192/64/0` / `128:8192/64/0` |
| 448 | `128:11776/64/28` / `128:11776/64/28` | `128:8192/64/0` / `128:8192/64/0` |
| 512 | `128:12288/64/32` / `128:12288/64/32` | `128:8192/64/0` / `128:8192/64/0` |
| 640 | `128:13312/64/40` / `128:13312/64/40` | `128:8192/64/0` / `128:8192/64/0` |
| 768 | `128:14336/64/48` / `128:14336/64/48` | `128:8192/64/0` / `128:8192/64/0` |

positive delta outliers:

| tokens | input_seed | cycle | delta |
|---:|---:|---:|---:|
| 128 | 5007 | 2 | +94.6 |
| 192 | 4007 | 2 | +28.4 |
| 192 | 4007 | 3 | +18.4 |
| 256 | 1007 | 2 | +0.7 |
| 256 | 2007 | 2 | +42.1 |
| 320 | 5007 | 2 | +72.2 |
| 448 | 3007 | 3 | +451.9 |
| 448 | 5007 | 2 | +5.7 |
| 640 | 2007 | 3 | +108.1 |
| 768 | 4007 | 2 | +433.2 |

해석:

```text
1. 이번 범위에서는 break-even M이 관측되지 않았다. 가장 작은 input=128,
   min received ~=254에서도 forced ON은 median -32.3 us, -2.24%로 이겼다.

2. 384와 512 tokens에서는 15/15 paired run이 모두 이겼고, 128/320/640/768도
   14/15로 대부분 이겼다.

3. Raw CSV 재계산 기준으로도 paired median과 win count가 위 table과
   일치한다. Seed별 median은 대부분 음수였고, 192-token의 seed=4007만
   양수였으므로 특정 routing seed 하나에만 의존한 결과로 보이지 않는다.

4. 다만 실행 순서는 cycle 1/3이 baseline -> ON, cycle 2가 ON -> baseline인
   2:1 구조라 완전히 균형적이지 않다. ON-first cycle에서도 대체로 이득은
   유지되지만, 다음 confirmation run은 4-cycle balanced order로 잡는다:
   OFF->ON, ON->OFF, ON->OFF, OFF->ON.

5. 몇 개 큰 positive transient outlier가 있다. Median 결론은 안정적이지만
   default 변경 전에는 GPU clock 고정이 가능하면 고정하고, correctness와
   balanced confirmation run을 추가한다.

6. 현재 데이터가 말할 수 있는 정확한 결론은 "이 workload의 break-even은
   M ~=254보다 낮고, 기존 1024 threshold는 지나치게 높다"이다.
   input=128의 min/critical received M이 254~255라서 threshold=256은
   이 최소 케이스의 한쪽 또는 양쪽 rank에서 ignore-invalid를 끌 수 있다.
   input=128의 이득까지 보존하려면 threshold 후보는 192 또는 128 이하가
   더 일관된다.

7. 반대로 전역 기본값을 바로 0으로 내리는 것도 아직 이르다.
   `_use_ep_ignore_invalid_experts()`는 DeepEP HT 전용 정책이 아니라
   `expert_map`을 쓰는 Triton EP 경로 전반에 영향을 준다. 안전한 순서는
   env override로 DeepEP HT에서 항상 ON 검증 -> DeepEP HT 전용 default
   검토 -> 다른 backend/shape 측정 뒤 generic default 검토다.

8. 다음 코드는 global generic + ignore-invalid를 주력 경로로 유지하되,
   local-ID나 direct assignment를 default 후보로 삼을 근거는 여전히 없다.
```

다음 측정:

```text
1. 진짜 break-even 확인

   input tokens:
     8,16,32,48,64,96,128

   5 seeds x 4 balanced cycles로 OFF/ON을 paired 측정한다.

2. ignore ON 이후 BLOCK_SIZE_M matrix

   tokens:
     256,320,384,448

   settings:
     A. ignore ON + 현재 default config
     B. ignore ON + W1 BLOCK_M=64
     C. ignore ON + W2 BLOCK_M=64
     D. ignore ON + W1/W2 BLOCK_M=64
     E. ignore ON + W1/W2 BLOCK_M=32

   필요하면 후보 BLOCK_M=32/64/128을 별도 sweep한다. W1과 W2는 padding
   감소와 GEMM 효율 tradeoff가 다를 수 있으므로 분리해서 측정한다.

3. 기록할 값

   W1 latency, activation latency, W2 latency, reduce latency,
   full critical-path latency, W1/W2 num_tokens_post_padded,
   W1/W2 BLOCK_SIZE_M, valid blocks, padding ratio.
```

후속 구현 후보:

```python
padded_rows(block_m) = sum(
    ceil_div(expert_count[e], block_m) * block_m
    for e in local_experts
)
```

invalid filtering 이후 expert histogram을 기준으로 후보 `BLOCK_M=32/64/128`
각각의 padded rows를 계산하고, GEMM 효율 LUT를 얹어 W1/W2 config를 고르는
routing-aware selector가 다음 큰 성능 기회다. 이후 ignore activation도
단순 received-M threshold 대신, 제거되는 invalid GEMM program 수를 추정하는
`saved_programs` 기준으로 바꾸는 쪽이 더 일반적이다.

추가 검증:

```text
- uniform, moderately skewed, hot-expert 집중, 실제 model router trace 비교
- 2-rank full output correctness 비교
```

## 20. Balanced runner and BLOCK_M sweep hook

리뷰에서 지적한 두 후속 실험을 반복 가능하게 만들기 위해 runner를 추가했다.

```text
benchmarks/results/run_deepep_ht_paired_matrix.py
```

지원하는 matrix:

```text
1. break-even:
   tokens 8,16,32,48,64,96,128
   OFF/ON 5 seeds x 4 balanced cycles

2. block-m-sweep:
   tokens 256,320,384,448
   ignore ON + default/W1/W2/both BLOCK_SIZE_M overrides
```

full break-even command:

```bash
python benchmarks/results/run_deepep_ht_paired_matrix.py \
  --mode break-even \
  --output benchmarks/results/deepep_ht_break_even_sub254_20260621_raw.csv
```

full BLOCK_M sweep command:

```bash
python benchmarks/results/run_deepep_ht_paired_matrix.py \
  --mode block-m-sweep \
  --output benchmarks/results/deepep_ht_block_m_sweep_20260621_raw.csv
```

W1/W2 각각의 tile만 바꿀 수 있도록 benchmark-only env hook도 추가했다.

```text
VLLM_MOE_TRITON_W1_BLOCK_SIZE_M_OVERRIDE
VLLM_MOE_TRITON_W2_BLOCK_SIZE_M_OVERRIDE
```

이 hook은 A100/SM80, BF16, top-k=8, Mixtral-like synthetic shape에만 걸리고,
env가 0이면 기존 config 선택을 그대로 둔다. 기존
`VLLM_MOE_TRITON_W1_A100_TUNED_CONFIG`와
`VLLM_MOE_TRITON_W2_A100_TUNED_CONFIG` default behavior는 바꾸지 않는다.

Smoke verification:

```text
benchmarks/results/deepep_ht_break_even_sub254_20260621_smoke_raw.csv
benchmarks/results/deepep_ht_block_m_sweep_20260621_smoke_raw.csv
```

break-even smoke는 `tokens=8`, `seed=1007`, `cycle=1`, `warmup=1`,
`iters=2`로 runner/output/analyzer path만 확인했다.

```text
baseline critical path:      1891.7 us
global_ignore critical path: 1773.9 us
delta:                       -117.9 us (-6.23%)
received M:                  15/16
```

이 수치는 짧은 smoke라 결론용 데이터는 아니지만, `received M<254`에서도
measurement path가 동작하고 paired analyzer가 정상적으로 읽는다는 점을
확인한다.

BLOCK_M smoke는 `tokens=320`, `seed=1007`, `cycle=1`, `warmup=1`,
`iters=2`로 default/W1_64/W2_64/both_32를 확인했다.

```text
block_default: 2139.2 us
block_w1_64:  2196.0 us
block_w2_64:  2260.5 us
block_both_32:2025.0 us
```

대표 assignment stats:

```text
default:
  0:m128:p8192:b64/0:sr3.19:vr3.19

W1 BLOCK_M=64:
  0:m128:p8192:b64/0:sr3.19:vr3.19;
  1:m64:p4096:b64/0:sr1.59:vr1.59

W2 BLOCK_M=64:
  0:m128:p8192:b64/0:sr3.19:vr3.19;
  1:m64:p4096:b64/0:sr1.59:vr1.59

W1/W2 BLOCK_M=32:
  0:m128:p8192:b64/0:sr3.19:vr3.19;
  1:m32:p3872:b121/0:sr1.51:vr1.51
```

즉 override가 실제 W1/W2 assignment를 새 `BLOCK_SIZE_M`으로 다시 만들고,
padding rows도 `8192 -> 4096 -> 3872`로 바뀐다. Smoke latency 자체는
`iters=2`라 noisy하지만, `both_32`가 default보다 빠르게 나온 점은 full
paired sweep을 돌릴 가치가 있다는 신호다.

3-seed break-even pilot:

```text
benchmarks/results/deepep_ht_break_even_sub254_20260621_3seed_raw.csv
benchmarks/results/deepep_ht_break_even_sub254_20260621_3seed_summary.md
```

Run shape:

```text
input seeds: 1007, 2007, 3007
cycles:      4 balanced cycles per seed
tokens:      8, 16, 32, 48, 64, 96, 128
settings:    baseline vs forced global_ignore
warmup/iters:20 / 100
rows:        168 measurements, 84 paired comparisons
```

Paired critical-path delta, `global_ignore - baseline`:

```text
tokens  recv-M median  median delta  pair pct  wins
8       16             -28.8 us      -2.23%    9/12
16      31/32          -21.1 us      -1.58%    9/12
32      63/64          -19.3 us      -1.38%    10/12
48      95             -47.8 us      -3.41%    10/12
64      127            -48.9 us      -3.41%    12/12
96      191            -30.9 us      -2.19%    9/12
128     254/255        -46.3 us      -3.22%    8/12
```

3-seed pilot에서도 모든 measured received-M 구간에서 median delta가 음수다.
즉 이번 workload에서는 `received M=16`까지 내려가도 forced generic
`ignore-invalid`의 break-even이 관측되지 않았다. 다만 `tokens=128`은
seed 1007에서 median delta가 `+12.4 us`로 뒤집힌 반면 seed 2007/3007은
각각 `-83.6 us`, `-39.7 us`였다. 따라서 global default를 0으로 낮추는
결론보다는, DeepEP HT + 이 synthetic shape에서는 threshold 1024가 너무
보수적이라는 결론이 더 안전하다.

## 21. Lazy BLOCK_M assignment and 54-run screening

Section 20의 BLOCK_M smoke 이후 `TritonExperts.apply()` 흐름을 다시 봤다.
기존 코드는 먼저 기본 `config`로 expert assignment를 만든 뒤 W1/W2 override
config assignment를 다시 만들었다. 따라서 `both_32` 같은 override는 실제로
W1/W2에서 쓰지 않는 `BLOCK_M=128` assignment 비용까지 포함해 측정될 수
있었다.

이를 W1/W2 config 선택 후 필요한 assignment만 lazy하게 생성하도록 바꿨다.
같은 `BLOCK_M`은 기존처럼 cache를 공유하므로 `both_32`와 `both_64`에서는
W1/W2가 같은 assignment를 재사용한다.

Lazy assignment smoke:

```text
benchmarks/results/deepep_ht_block_m_screening_20260621_lazy_assignment_smoke_raw.csv
```

Representative rank0 assignment stats:

```text
block_default: 0:m128:p8192:b64/0:sr3.19:vr3.19
block_both_64: 0:m64:p4096:b64/0:sr1.59:vr1.59
block_both_32: 0:m32:p3872:b121/0:sr1.51:vr1.51
```

이제 override 설정에서 불필요한 M128 assignment가 보이지 않는다.

54-run screening:

```text
benchmarks/results/deepep_ht_block_m_screening_20260621_3seed_raw.csv
benchmarks/results/deepep_ht_block_m_screening_20260621_3seed_summary.md
```

Run shape:

```text
tokens:       320, 448
settings:     block_default, block_both_64, block_both_32
input seeds:  1007, 2007, 3007
cycles:       3-way Latin square
warmup/iters: 20 / 100
rows:         54 measurements, 36 paired comparisons
```

Cycle order:

```text
cycle 1: default -> both_64 -> both_32
cycle 2: both_64 -> both_32 -> default
cycle 3: both_32 -> default -> both_64
```

Paired critical-path delta, `setting - block_default`:

```text
tokens  setting        recv-M median  median delta  pair pct  wins
320     block_both_32  638            -124.7 us     -7.74%    9/9
320     block_both_64  638            -145.7 us     -8.91%    9/9
448     block_both_32  892            -93.4 us      -5.69%    9/9
448     block_both_64  892            -146.1 us     -8.80%    9/9
```

판정 기준이었던 `>=2% median improvement`, `>=7/9 wins`, `한 token size에서
>=3%`를 모두 넉넉히 넘었다. 특히 `both_64`가 320/448 양쪽에서
`both_32`보다 더 빠르다. 즉 단순히 padded rows를 최소화하는 것보다 GEMM
efficiency까지 같이 봐야 한다.

다음 실험은 full 400-run sweep이 아니라 W1/W2 기여 분리다.

```text
tokens:   320, 448
settings: block_default, block_w1_64, block_w2_64, block_both_64
seeds:    1007, 2007, 3007
cycles:   4
```

여기서 W1/W2 중 어느 쪽의 `BLOCK_M=64`가 실제 이득을 만드는지 확인한 뒤,
winner만 256/320/384/448 범위에서 5-seed final validation으로 확장한다.

## 22. BLOCK_M correctness, W1/W2 split, combined ablation

Section 21 이후 세 가지를 보완했다.

1. benchmark-only `BLOCK_M` override validation
2. final output correctness smoke
3. W1/W2 contribution split과 same-session combined ablation

Override validation:

```text
VLLM_MOE_TRITON_W1_BLOCK_SIZE_M_OVERRIDE
VLLM_MOE_TRITON_W2_BLOCK_SIZE_M_OVERRIDE
```

위 두 env hook은 이제 `32,64,128`만 허용한다. `48` 같은 값은 바로
`ValueError`를 낸다.

Correctness smoke:

```text
benchmarks/results/verify_deepep_ht_block_m_correctness.py
benchmarks/results/deepep_ht_block_m_correctness_tokens320_20260621.json
benchmarks/results/deepep_ht_block_m_correctness_tokens448_20260621.json
```

두 token size에서 `block_default` output을 reference로 두고 `block_both_64`,
`block_both_32`를 비교했다. tolerance는 BF16용 `rtol=1.6e-2`,
`atol=1.0e-2`다.

```text
tokens  rank  setting        max_abs  mean_abs  rel_l2  assert_close
320     0     block_both_64  0.0      0.0       0.0     true
320     0     block_both_32  0.0      0.0       0.0     true
320     1     block_both_64  0.0      0.0       0.0     true
320     1     block_both_32  0.0      0.0       0.0     true
448     0     block_both_64  0.0      0.0       0.0     true
448     0     block_both_32  0.0      0.0       0.0     true
448     1     block_both_64  0.0      0.0       0.0     true
448     1     block_both_32  0.0      0.0       0.0     true
```

W1/W2 split:

```text
benchmarks/results/deepep_ht_block_m_w1_w2_split_20260621_3seed_raw.csv
benchmarks/results/deepep_ht_block_m_w1_w2_split_20260621_3seed_commands.log
benchmarks/results/deepep_ht_block_m_w1_w2_split_20260621_3seed_summary.md
```

Run shape:

```text
tokens:       320, 448
settings:     block_default, block_w1_64, block_w2_64, block_both_64
input seeds:  1007, 2007, 3007
cycles:       4-way cyclic order
warmup/iters: 20 / 100
rows:         96 measurements, 72 paired comparisons
```

Paired critical-path delta, `setting - block_default`:

```text
tokens  setting        median delta  pair pct   wins
320     block_w1_64    -93.6 us      -5.73%     12/12
320     block_w2_64    -29.5 us      -1.82%     10/12
320     block_both_64  -175.0 us     -10.77%    12/12
448     block_w1_64    -69.5 us      -4.20%     11/12
448     block_w2_64    -26.3 us      -1.59%     11/12
448     block_both_64  -143.6 us     -8.62%     12/12
```

W1-only가 대부분의 단독 이득을 설명하지만, W2-only도 median 기준 이득이다.
다만 W2-only에는 positive transient가 있었다. `block_both_64`는 두 token
size 모두 `12/12` wins로 가장 안정적이며, 단독 W1/W2보다 크다. 따라서 다음
policy 후보는 W1/W2 양쪽 M64다.

Same-session combined ablation:

```text
benchmarks/results/deepep_ht_block_m_combined_ablation_20260621_3seed_raw.csv
benchmarks/results/deepep_ht_block_m_combined_ablation_20260621_3seed_commands.log
benchmarks/results/deepep_ht_block_m_combined_ablation_20260621_3seed_summary.md
```

Run shape:

```text
tokens:       320, 448
settings:     original, filtering, final_both_64
input seeds:  1007, 2007, 3007
cycles:       3-way Latin square
warmup/iters: 20 / 100
rows:         54 measurements, 36 paired comparisons
```

Settings:

```text
original:      ignore-invalid OFF, default BLOCK_M
filtering:     ignore-invalid ON, default BLOCK_M
final_both_64: ignore-invalid ON, W1/W2 BLOCK_M=64
```

Paired critical-path delta, `setting - original`:

```text
tokens  setting        median delta  pair pct   wins
320     filtering      -43.6 us      -2.65%     9/9
320     final_both_64  -213.4 us     -12.87%    9/9
448     filtering      -56.7 us      -3.34%     8/9
448     final_both_64  -178.6 us     -10.52%    9/9
```

`final_both_64 - filtering` 추가 이득:

```text
tokens  median delta  pair pct  wins
320     -160.3 us     -9.99%    9/9
448     -137.3 us     -8.37%    9/9
```

따라서 같은 세션에서도 contribution split이 명확하다. `ignore-invalid`는
기존 경로 대비 약 2.6-3.3%를 만들고, W1/W2 M64는 그 위에서 약 8.4-10.0%를
추가한다. 최종 후보는 original 대비 약 10.5-12.9% 빠르다.

다음은 모든 setting을 넓게 돌리는 full sweep이 아니라 M64/M128 crossover
탐색이다. 우선 `default` vs `final_both_64`만 비교해 token range
`256,320,384,448,512,640,768,1024`에서 상한을 찾고, 경계 주변만 5 seeds
× 4 cycles로 재검증한다.

## 23. DeepEP HT fixed-capacity dispatch prototype

Section 22 이후에는 compute padding이 줄어든 최종 설정에서 DeepEP HT prepare
orchestration이 얼마나 남는지 다시 봤다. `nsys`는 현재 환경에 없고 `ncu`만
있어서, 기존 section profiler로 먼저 재측정했다.

Final setting profile:

```text
benchmarks/results/deepep_ht_final_both64_profile_20260621_raw.csv
benchmarks/results/deepep_ht_final_both64_profile_20260621/
```

`final_both_64`의 대표 section 평균:

```text
tokens  rank  prepare  dispatch_submit  receiver  topk_remap  metadata  experts  finalize  combine_copy
320     0     411.9us  169.0us          205.6us   95.1us      43.8us    772.6us  195.1us   29.6us
320     1     402.5us  154.4us          209.9us   96.5us      45.5us    782.6us  179.8us   29.7us
448     0     398.5us  158.0us          204.5us   93.3us      43.5us    805.4us  186.2us   31.2us
448     1     400.5us  160.3us          204.5us   93.9us      43.3us    800.9us  193.2us   31.2us
```

M64로 expert compute를 줄인 뒤에도 prepare는 약 400us 남고, receiver remap
plus metadata가 약 137-142us다. combine receiver copy도 약 30us 남는다. 즉
다음 최적화 축은 통신량 압축보다 CPU-sync/metadata/remap/copy 제거가 맞다.

Prototype:

```text
VLLM_DEEPEP_HT_FIXED_CAPACITY_DISPATCH=1
VLLM_DEEPEP_HT_FIXED_CAPACITY_NUM_WORST_TOKENS=<explicit capacity>
```

이 flag는 DeepEP intranode dispatch에 `num_worst_tokens`를 넘긴다. 이후
safety patch에서는 `local_input_tokens * dp_size` 자동 계산을 제거했고,
`VLLM_DEEPEP_HT_FIXED_CAPACITY_NUM_WORST_TOKENS > 0`을 필수로 만들었다.
권장값은 scheduler/static bucket이 알고 있는
`max_tokens_per_dp_rank * dp_size`다. 실행 시에는 forward context의
`num_tokens_across_dp_cpu.sum()`과 비교해 capacity가 작으면 dispatch 전에
명확히 실패한다.

DeepEP는 실제 receive row 뒤를 `topk_id=-1`로 채우고 per-expert count
list를 비워 반환한다. vLLM 쪽은 dispatch 시점의 `num_worst_tokens > 0`을
receiver에 명시적으로 넘기고, fixed mode일 때
`ExpertTokensMetadata.make_from_list()`를 생략한 뒤 generic ignore-invalid
assignment로 넘긴다. 더 이상 빈 expert-count list 길이로 mode를 추론하지
않는다.

현재 prototype은 의도적으로 좁게 gated 되어 있다.

```text
requires:
  DeepEP HT intranode
  TritonExperts backend
  BF16, unquantized experts
  no LoRA
  VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS=1
  generic global expert IDs

rejects:
  internode
  non-Triton expert backends
  quantized expert kernels
  LoRA
  VLLM_DEEPEP_HT_LOCAL_EXPERT_IDS=1
  VLLM_DEEPEP_HT_DIRECT_ASSIGNMENT=1
```

Correctness:

```text
benchmarks/results/verify_deepep_ht_block_m_correctness.py
benchmarks/results/deepep_ht_block_m_correctness_tokens320_20260621.json
benchmarks/results/deepep_ht_block_m_correctness_tokens448_20260621.json
```

`fixed_both_64`도 `block_default` reference와 비교해 rank0/rank1 모두
`max_abs=0`, `mean_abs=0`, `relative_l2=0`, `assert_close=true`였다.

Safety follow-up correctness:

```text
benchmarks/results/deepep_ht_fixed_capacity_safety_correctness_tokens320_20260621.json
benchmarks/results/deepep_ht_fixed_capacity_safety_correctness_rank128_512_20260621.json
benchmarks/results/deepep_ht_fixed_capacity_safety_correctness_rank512_128_20260621.json
benchmarks/results/deepep_ht_fixed_capacity_safety_correctness_rank128_512_target0_20260621.json
benchmarks/results/deepep_ht_fixed_capacity_safety_correctness_rank128_512_target1_20260621.json
```

Balanced, asymmetric `128/512`, asymmetric `512/128`, and route-target
concentrated smoke cases all matched `block_default` with zero error. An
intentional under-capacity run with rank tokens `128/512`, target rank 0, and
capacity `256` failed before dispatch:

```text
VLLM_DEEPEP_HT_FIXED_CAPACITY_NUM_WORST_TOKENS must be at least the DP token
upper bound (640), got 256.
```

Paired performance:

```text
benchmarks/results/deepep_ht_fixed_capacity_vs_final_20260621_3seed_raw.csv
benchmarks/results/deepep_ht_fixed_capacity_vs_final_20260621_3seed_commands.log
benchmarks/results/deepep_ht_fixed_capacity_vs_final_20260621_3seed_summary.md
```

Run shape:

```text
tokens:       320, 448
settings:     block_both_64, block_fixed_both_64
input seeds:  1007, 2007, 3007
cycles:       4 balanced cycles
warmup/iters: 20 / 100
rows:         48 measurements, 24 paired comparisons
```

Paired critical-path delta, `block_fixed_both_64 - block_both_64`:

```text
tokens  recv-M median  median delta  pair pct  wins
320     640            -57.9 us      -3.98%    10/12
448     896            -67.0 us      -4.45%    12/12
```

320에는 one-off positive outlier가 있었지만 seed-level median은 세 seed 모두
음수였다. 448은 12/12 wins다.

Fixed-capacity section profile:

```text
benchmarks/results/deepep_ht_fixed_capacity_profile_20260621_raw.csv
benchmarks/results/deepep_ht_fixed_capacity_profile_20260621/
```

Representative section deltas:

```text
tokens  setting              prepare  receiver  topk_remap  metadata  experts  combine_copy
320     block_both_64        ~590us   ~321us    ~144us      ~70us     ~938us   ~48us
320     block_fixed_both_64  ~361us   ~160us    ~89us       ~6us      ~761us   ~30us
448     block_both_64        ~424us   ~204us    ~91us       ~44us     ~816us   ~30us
448     block_fixed_both_64  ~370us   ~161us    ~87us       ~6us      ~794us   ~32us
```

Section profiling perturbs absolute timings, but the direction is clear:
metadata creation almost disappears, receiver time drops, and full forward
improves by about 4% on the paired run. This validates fixed-capacity as the
next communication-side project.

Next steps:

```text
1. Replace the remaining -1 -> global invalid remap with raw -1 skip in generic
   alignment, so fixed-capacity can remove topk_remap as well as metadata.
2. Add an async/DBO sweep:
   settings = final_both_64, fixed_both_64, fixed_both_64 + DBO
   comm SMs = 8,12,16,20,24
3. Track route stats for fixed-capacity mode without CPU count list, likely by
   counting valid top-k rows on GPU or deriving from assignment stats.
```

## 25. DeepEP HT fixed-capacity raw `-1` alignment

Section 24 left one visible receiver cost: fixed-capacity still remapped DeepEP
HT receiver-local top-k ids and padded `-1` rows into global/sentinel expert
ids before generic alignment. This change removes that remap kernel for the
fixed-capacity path.

Implementation:

```text
DeepEP HT fixed-capacity receiver
  keeps raw local expert ids and raw -1 padded/non-local slots

expert assignment params
  use local_num_experts plus a local identity expert_map

CUDA moe_align_block_size
  skips expert_id < 0 and expert_id >= num_experts before expert_map lookup
```

The local identity map is intentionally kept. It lets the existing
ignore-invalid W2 reduction skip raw `-1` top-k slots without reintroducing a
top-k remap kernel.

Additional fixed-capacity guards:

```text
requires top_k=8
requires VLLM_MOE_TRITON_W2_REDUCE_FUSION=0
requires ignore-invalid min-tokens <= fixed capacity
rejects apply_router_weight_on_input=True
```

Correctness:

```text
benchmarks/results/deepep_ht_raw_negative_correctness_tokens320_20260621.json
benchmarks/results/deepep_ht_raw_negative_correctness_rank128_512_target1_20260621.json
```

Both balanced 320-token and asymmetric `rank0=128/rank1=512`,
`route_target_rank=1` smokes matched `block_default` with `max_abs_error=0`,
`mean_abs_error=0`, `relative_l2_error=0` for `fixed_both_64` on both ranks.

Raw align CUDA smoke:

```text
topk = [[0, -1, 1, 64], [2, -2, 3, -1]]
num_experts = 4
post_pad = 16
expert_ids = [0, 1, 2, 3]
```

Profile inputs:

```text
benchmarks/results/deepep_ht_raw_negative_profile_20260621_raw.csv
benchmarks/results/deepep_ht_raw_negative_profile_20260621_summary.md
benchmarks/results/deepep_ht_raw_negative_profile_20260621_direct_both64.rank*.json
benchmarks/results/deepep_ht_raw_negative_profile_20260621_direct_fixed.rank*.json
```

Single-seed paired run, tokens=320, warmup/iters=10/50:

```text
block_both_64        critical path 1609.2 us
block_fixed_both_64  critical path 1446.5 us
delta                -162.6 us / -10.11%
```

The 1-cycle result is only a smoke, not a final performance claim, but it
confirms the code path is live.

Representative direct section profile, tokens=320:

```text
setting              rank  topk_remap  metadata  dispatch_receiver
block_both_64        0     93.1 us     42.9 us   204.1 us
block_both_64        1     100.9 us    47.1 us   221.0 us
block_fixed_both_64  0     8.5 us      5.4 us    80.1 us
block_fixed_both_64  1     8.8 us      5.6 us    81.0 us
```

The fixed remap section now measures only the Python section wrapper and
assert/pass overhead. The remaining fixed-capacity work is mostly dispatch
wait/unpack/post-quant and expert compute; the explicit top-k remap kernel is
no longer on the receiver path.
