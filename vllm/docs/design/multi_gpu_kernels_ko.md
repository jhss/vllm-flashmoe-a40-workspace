# vLLM Multi-GPU Kernel 동작 방식

작성일: 2026-06-19

이 문서는 vLLM에서 multi-GPU 실행이 실제로 어떤 구조로 동작하는지,
특히 CUDA kernel, NCCL/custom collective, tensor parallelism, expert
parallelism, MoE all-to-all 경로가 어떻게 연결되는지 정리한다.

핵심 요약은 다음과 같다.

- vLLM은 하나의 거대한 "multi-GPU CUDA kernel"을 실행하는 방식이 아니다.
- GPU마다 별도 worker process가 있고, 각 process는 자기 GPU에서 rank-local
  CUDA/Triton/CUTLASS/FlashAttention/DeepGEMM kernel을 실행한다.
- GPU 간 데이터 이동은 `GroupCoordinator`가 감싼 process group을 통해
  `all_reduce`, `all_gather`, `reduce_scatter`, `send/recv`, MoE `dispatch/combine`
  같은 collective로 발생한다.
- dense 모델의 multi-GPU 핵심은 tensor parallel linear layer의
  all-reduce/all-gather이다.
- MoE 모델의 multi-GPU 핵심은 라우팅 결과에 따라 token을 expert 소유 rank로
  보내고 다시 합치는 expert-parallel dispatch/combine이다.

## 코드 지도

| 영역 | 주요 코드 |
| --- | --- |
| worker 생성과 rank 실행 | `vllm/v1/executor/multiproc_executor.py`, `vllm/v1/executor/ray_executor*.py` |
| GPU worker 초기화 | `vllm/v1/worker/gpu_worker.py` |
| distributed group 생성 | `vllm/distributed/parallel_state.py` |
| TP collective wrapper | `vllm/distributed/communication_op.py` |
| CUDA communicator | `vllm/distributed/device_communicators/cuda_communicator.py` |
| custom all-reduce | `vllm/distributed/device_communicators/custom_all_reduce.py` |
| MoE all-to-all manager | `vllm/distributed/device_communicators/all2all.py` |
| TP linear layer | `vllm/model_executor/layers/linear.py` |
| vocab embedding/logits gather | `vllm/model_executor/layers/vocab_parallel_embedding.py`, `vllm/model_executor/layers/logits_processor.py` |
| MoE runner | `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` |
| MoE kernel abstraction | `vllm/model_executor/layers/fused_moe/modular_kernel.py` |
| MoE EP backend 선택 | `vllm/model_executor/layers/fused_moe/all2all_utils.py` |
| MoE EP prepare/finalize | `vllm/model_executor/layers/fused_moe/prepare_finalize/*.py` |
| MoE expert compute | `vllm/model_executor/layers/fused_moe/experts/*.py` |
| context parallel attention | `vllm/v1/attention/backends/flash_attn.py`, `vllm/v1/attention/backends/flashinfer.py` |

## 큰 그림

vLLM V1의 실행 구조는 보통 다음 계층으로 나뉜다.

```text
API server
  -> engine core / scheduler
    -> executor
      -> GPU worker process per GPU
        -> model runner
          -> torch module forward
            -> rank-local CUDA kernels
            -> distributed collectives
```

`MultiprocExecutor`는 single-node에서 GPU worker를 process 단위로 띄운다.
각 worker는 `rank`, `local_rank`, `distributed_init_method`를 받고, 자기 GPU를
선택한 뒤 distributed environment를 초기화한다.

`gpu_worker.py`의 초기화 순서는 중요하다.

1. `torch.accelerator.set_device_index(self.device)`로 local GPU를 고정한다.
2. `init_worker_distributed_environment(...)`를 호출한다.
3. 내부에서 `init_distributed_environment(...)`가 torch distributed world를 만든다.
4. `ensure_model_parallel_initialized(...)`가 TP, PP, DP, EP 등 model-parallel
   subgroup을 만든다.
5. NCCL buffer가 잡힌 뒤 memory snapshot을 떠서 KV cache 용량을 계산한다.

따라서 vLLM에서 multi-GPU kernel을 볼 때는 먼저 "하나의 forward가 모든 rank에서
동시에 실행되고, rank-local kernel 사이에 collective가 들어간다"는 모델로 보는
것이 좋다.

## Distributed Group 모델

`parallel_state.py`의 `GroupCoordinator`가 핵심 추상화다.

`GroupCoordinator`는 한 parallel dimension에 대해 두 종류의 process group을
관리한다.

- `device_group`: GPU tensor collective용 group. CUDA에서는 보통 NCCL 계열이다.
- `cpu_group`: metadata, object broadcast, setup, 일부 coordination용 Gloo group.

그리고 `device_communicator`가 실제 device별 구현을 담당한다. CUDA에서는
`CudaCommunicator`가 선택된다.

vLLM이 만드는 주요 group은 다음과 같다.

| group | 의미 | 주 사용처 |
| --- | --- | --- |
| `WORLD` | 전체 distributed ranks | 초기화, global coordination |
| `TP` | tensor parallel group | dense linear all-reduce/all-gather, vocab gather |
| `PP` | pipeline parallel group | stage 간 activation send/recv |
| `DP` | data parallel group | DP worker 간 synchronized forward, MoE AG/RS fallback |
| `EP` | expert parallel group | MoE expert ownership, all-to-all dispatch/combine |
| `EPLB` | EP와 같은 rank set의 별도 group | expert load balancer weight 이동, deadlock 격리 |
| `PCP` | prefill context parallel group | prefill/context 관련 token 분산 |
| `DCP` | decode context parallel group | decode attention query all-gather/combine |

`initialize_model_parallel()`은 rank 배열을 다음 logical layout으로 해석한다.

```text
ExternalDP x DP x PP x PCP x TP
```

각 group은 이 tensor를 다른 축 기준으로 reshape/transpose해서 만든다. 예를 들어
TP group은 마지막 TP 축으로 묶고, PP group은 pipeline 축으로 묶는다. EP group은
MoE 모델일 때 `DP * PCP * TP`를 묶어 expert ownership을 만든다.

## Collective 호출 경로

레이어 코드에서는 직접 `torch.distributed`를 만지는 대신 wrapper를 사용한다.

```text
tensor_model_parallel_all_reduce(x)
  -> get_tp_group().all_reduce(x)
    -> GroupCoordinator.all_reduce(x)
      -> CudaCommunicator.all_reduce(x)
        -> chosen backend
```

TP wrapper는 `communication_op.py`에 있다.

- `tensor_model_parallel_all_reduce`
- `tensor_model_parallel_all_gather`
- `tensor_model_parallel_reduce_scatter`
- `tensor_model_parallel_gather`

`GroupCoordinator`는 torch custom op 경유와 직접 Python 호출 경유를 모두 지원한다.
custom op 경유는 `torch.compile`/CUDA graph와 함께 쓰기 쉽게 하기 위한 것이다.
collective는 out-of-place로 동작하도록 감싸져 있다.

## CUDA All-Reduce Backend 선택

CUDA에서 TP all-reduce는 `CudaCommunicator.all_reduce()`가 런타임에 backend를
고른다. 선택 순서는 대략 다음과 같다.

1. NCCL symmetric memory all-reduce custom op
2. ROCm quick reduce
3. FlashInfer all-reduce
4. vLLM custom all-reduce
5. torch symmetric memory communicator
6. PyNcclCommunicator
7. `torch.distributed.all_reduce` fallback

각 backend는 dtype, tensor size, world size, topology, env flag에 따라 거절될 수
있다. 그러면 다음 backend로 내려간다.

`CustomAllreduce`는 single-node에서 빠른 TP all-reduce를 노리는 경로다. 주요
제약은 다음과 같다.

- world size는 `[2, 4, 6, 8]`만 지원한다.
- multi-node group에서는 꺼진다.
- GPU P2P access가 필요하다.
- 2 GPU는 PCIe-only라도 가능성이 있지만, 4 GPU 이상 PCIe-only이면 보통 꺼진다.
- input byte size는 16의 배수여야 하고 weak-contiguous여야 한다.
- CUDA graph capture 중에는 graph buffer IPC metadata를 모아 등록한다.

이 때문에 A40 PCIe 2장에서는 custom all-reduce가 켜질 수 있지만, 4장 이상
PCIe-only topology에서는 NCCL/PyNCCL fallback으로 보는 경우가 많다.

## Dense Tensor Parallelism

dense transformer layer에서 multi-GPU 통신은 대부분 TP linear layer에서 발생한다.

### ColumnParallelLinear

`ColumnParallelLinear`는 weight의 output dimension을 TP rank별로 나눈다.

```text
Y = X A
A = [A_0, A_1, ..., A_{tp-1}]
rank i computes Y_i = X A_i
```

forward는 다음 흐름이다.

```text
local GEMM
  -> output_parallel
  -> gather_output=True이면 tensor_model_parallel_all_gather
  -> 아니면 shard 상태 유지
```

QKV projection, gate/up projection처럼 다음 연산이 shard 상태를 자연스럽게 받을 수
있는 경우에는 보통 gather하지 않는다.

### RowParallelLinear

`RowParallelLinear`는 weight의 input dimension을 TP rank별로 나눈다.

```text
X = [X_0, X_1, ..., X_{tp-1}]
A = [A_0; A_1; ...; A_{tp-1}]
Y = sum_i X_i A_i
```

forward는 다음 흐름이다.

```text
input이 이미 shard이면 그대로 사용
input이 full이면 rank별 slice 선택
  -> local GEMM으로 partial output 계산
  -> reduce_results=True이면 tensor_model_parallel_all_reduce
```

attention output projection, MLP down projection에서 이 all-reduce가 자주 보인다.

### VocabParallelEmbedding

`VocabParallelEmbedding`은 vocab dimension을 TP rank별로 나눈다.

1. 각 rank가 자기 vocab shard에 해당하지 않는 token id를 mask한다.
2. local embedding lookup을 한다.
3. mask된 위치는 0으로 만든다.
4. TP all-reduce로 rank별 partial embedding을 합친다.

각 token은 정확히 한 rank의 vocab shard에만 매칭되므로 all-reduce는 사실상
"sum으로 correct shard 결과만 남기는" 역할을 한다.

### LogitsProcessor와 vocab gather

LM head도 vocab-parallel이면 각 rank가 local vocab logits만 계산한다.
sampling에 full vocab logits가 필요하면 `LogitsProcessor`가 TP group에서 gather 또는
all-gather를 수행한다.

일부 경로는 full logits gather 대신 local argmax `(value, index)`만 모아 global
argmax를 만드는 최적화도 있다. 이 경우 통신량은 `batch * vocab_size`가 아니라
`batch * tp_size` 규모가 된다.

### Dense block의 통신 패턴

일반적인 TP transformer block은 다음처럼 볼 수 있다.

```text
hidden states replicated on TP ranks
  -> qkv_proj: ColumnParallelLinear, communication 없음 또는 적음
  -> attention kernel: rank-local heads와 rank-local KV cache 사용
  -> o_proj: RowParallelLinear, TP all-reduce
  -> gate/up proj: MergedColumnParallelLinear, communication 없음
  -> activation/local MLP
  -> down_proj: RowParallelLinear, TP all-reduce
  -> residual/norm
```

즉 dense TP의 주요 multi-GPU 비용은 attention kernel 내부라기보다 attention/MLP 주변
linear의 all-reduce인 경우가 많다. attention kernel은 보통 자기 rank의 head shard와
KV cache shard를 사용해 rank-local로 돈다.

## Pipeline Parallelism

PP는 layer stage를 rank group에 나누는 방식이다. 각 stage 내부에서는 위 TP 패턴이
그대로 적용되고, stage 경계에서는 activation을 다음 PP rank로 보낸다.

vLLM의 `GroupCoordinator`는 `send`, `recv`, `broadcast`도 제공한다. Ray backend에서는
Ray compiled graph용 PP communicator가 별도로 쓰일 수 있다. PP는 kernel 하나의
모양을 바꾸기보다는, layer 실행 순서와 rank 간 activation 이동을 바꾸는 병렬화다.

## Context Parallelism

context parallelism은 attention context를 나누는 경로다.

DCP(decode context parallel)의 FlashAttention/FlashInfer 경로에서는 query를 DCP
group에서 all-gather한 뒤 attention을 계산하고, partial attention output과 LSE를
combine한다.

예를 들어 FlashAttention backend는 대략 다음 흐름을 가진다.

```text
rank-local query
  -> get_dcp_group().all_gather(query, dim=1)
  -> flash_attn_varlen_func(...)
  -> DCP combine(output, lse)
```

FlashInfer decode 경로도 DCP가 켜져 있으면 decode query를 DCP group에서 all-gather한
뒤 wrapper를 실행하고 `dcp_combine`으로 결과를 합친다.

PCP(prefill context parallel)는 MoE runner에도 영향을 준다. `moe_runner.py`에는
PCP size가 1보다 클 때 hidden states와 router logits를 PCP group에서 all-gather하고,
출력은 reduce-scatter하는 fallback 흐름이 있다.

## MoE 병렬화의 핵심

MoE는 dense TP보다 복잡하다. vLLM의 MoE layer는 다음 구성요소로 만들어진다.

```text
FusedMoE(...)
  -> FusedMoEParallelConfig
  -> ExpertMapManager
  -> Router
  -> RoutedExperts
  -> MoERunner
```

`MoERunner`는 forward를 `torch.ops.vllm.moe_forward` custom op로 감싼다. 이 custom
op는 구현을 숨기려는 것이 아니라, torch.compile/CUDA graph가 MoE 내부의 동적 dispatch
로직을 안정적으로 다루도록 하는 경계 역할을 한다.

MoE forward의 큰 흐름은 다음과 같다.

```text
hidden_states, router_logits
  -> router.select_experts()
       topk_ids, topk_weights
  -> prepare
       quantize, pack, dispatch, metadata 생성
  -> experts
       grouped GEMM: w13 -> activation -> w2
  -> finalize
       top-k weight 적용, reduce, combine
  -> shared experts와 합산
  -> 필요하면 TP/EP all-reduce
```

## MoE Parallel Config

`FusedMoEParallelConfig.make()`가 MoE layer 관점의 병렬 전략을 결정한다.

중요한 점은 `enable_expert_parallel=True`일 때 MoE의 의미가 dense TP와 달라진다는
것이다.

### EP가 꺼진 경우

EP가 꺼져 있으면 MoE expert weight도 dense MLP처럼 tensor-sharded된다. 코드에서는
DP/PCP/TP를 flatten한 TP size로 보고 expert weight shard를 만든다.

```text
TP/DP/PCP ranks collectively shard expert tensors
expert GEMM도 tensor-parallel partial compute
필요한 곳에서 all-reduce
```

### EP가 켜진 경우

EP가 켜지면 각 rank가 expert를 통째로 소유한다.

```text
original flattened TP size = DP * PCP * TP
MoE 내부 tp_size = 1
MoE ep_size = flattened TP size
rank i owns a subset of experts
```

즉 EP에서는 하나의 expert GEMM을 여러 rank가 쪼개서 계산하는 것이 아니라, rank마다
서로 다른 expert를 완전히 들고 있다. 따라서 병목은 "GEMM partial sum all-reduce"보다
"token을 올바른 expert 소유 rank로 보내고 다시 합치는 dispatch/combine" 쪽으로
이동한다.

### EP라고 항상 all-to-all인 것은 아님

이 부분이 중요하다.

`use_all2all_kernels`는 코드상 `dp_size > 1 and use_ep`일 때 참이다. 즉 EP가 켜져도
DP가 1이면 DeepEP 같은 all-to-all backend를 쓰지 않을 수 있다.

예를 들어 `TP=2, DP=1, EP=True`라면 attention/MLP 주변의 hidden states는 두 rank에
replicated되어 있을 수 있다. 각 rank는 자기 local expert만 계산하고, 마지막에
rank별 contribution을 all-reduce해서 전체 token output을 만든다. 이 경우 token을
다른 DP rank에서 가져올 필요가 없으므로 all-to-all dispatch가 필수는 아니다.

반대로 `TP=1, DP=2, EP=True` 또는 `TP=2, DP=2, EP=True`라면 각 DP rank가 서로 다른
token batch를 들고 있다. 그런데 expert ownership은 DP/TP를 가로질러 나뉘므로 token을
expert 소유 rank로 보내야 한다. 이때 all-to-all 또는 AG/RS dispatch/combine이 핵심
비용이 된다.

## MoE Modular Kernel 구조

`modular_kernel.py`의 설계는 MoE를 다음 컴포넌트로 나눈다.

```text
[Router] -> [Prepare/Dispatch] -> [Experts] -> [Finalize/Combine]
```

주요 abstraction은 다음과 같다.

- `FusedMoEPrepareAndFinalizeModular`
  - input quantization
  - token dispatch
  - scale/topk metadata 이동
  - output combine
  - top-k weight 적용 또는 reduce 위치 조정
- `FusedMoEExperts`
  - expert GEMM 본체
  - `w13`, activation, `w2`
  - Triton, DeepGEMM, CUTLASS, Marlin, FlashInfer 등 구현 선택
- `FusedMoEKernel`
  - prepare/finalize와 experts를 조립한 실행 객체
- `TopKWeightAndReduce`
  - top-k weight 적용과 top-k 축 reduction을 누가 할지 명시

activation format은 두 종류가 있다.

| format | shape | 주 사용처 |
| --- | --- | --- |
| `Standard` | `[num_tokens, hidden]` 또는 top-k expanded layout | naive, DeepEP HT, 일반 Triton experts |
| `BatchedExperts` | `[num_experts, max_tokens_per_expert, hidden]` | DeepEP LL, batched experts |

backend와 expert kernel은 activation format, quantization format, top-k 처리 위치가
맞아야 한다.

## Naive AG/RS MoE Backend

`allgather_reducescatter` backend는 `AgRsAll2AllManager`가 담당한다.

dispatch는 실제 sparse all-to-all이라기보다 다음에 가깝다.

```text
각 DP rank의 hidden_states/topk_ids/topk_weights
  -> all_gatherv로 모든 DP rank token을 모음
  -> 각 rank가 자기 local expert에 해당하는 token만 유효하게 계산
```

combine은 다음과 같다.

```text
local expert output
  -> top-k weight/reduce
  -> reduce_scatterv로 원래 DP rank별 token chunk로 돌려줌
```

장점은 단순하고 일반적이라는 점이다. 단점은 모든 token을 모든 rank가 보게 되는
구조라서 EP size가 커질수록 통신량과 불필요한 metadata 처리 비용이 커질 수 있다.

## DeepEP High Throughput Backend

`deepep_high_throughput`은 `DeepEPHTAll2AllManager`와
`DeepEPHTPrepareAndFinalize`가 담당한다.

prepare 흐름은 다음과 같다.

```text
optional input quantization
  -> DeepEP get_dispatch_layout(topk_ids)
  -> DeepEP dispatch(tokens, scales, topk_ids, topk_weights)
  -> expert_x, expert_topk_ids, expert_topk_weights, token counts, handle
  -> local expert id/global expert id remap
  -> experts kernel
```

finalize 흐름은 다음과 같다.

```text
fused_expert_output
  -> top-k weight/reduce를 local에서 적용
  -> DeepEP combine(handle)
  -> original token layout output
```

특징은 다음과 같다.

- `Standard` activation format을 사용한다.
- dispatch/combine handle을 pair로 유지한다.
- DBO microbatching 때문에 handle은 ubatch별로 따로 저장한다.
- BF16 combine 제약이 있다.
- block-quantized FP8이면 dispatch 전에 quantize할 수 있고, 그 외에는 dispatch 뒤
  quantize할 수 있다.
- hidden size를 DeepEP transfer atom에 맞게 round up할 수 있다.
- `VLLM_DEEPEP_HT_NUM_SMS`로 통신 kernel이 사용할 SM 수를 조정한다.

HT backend는 이름 그대로 prefill처럼 token 수가 많고 throughput이 중요한 상황에
맞는 편이다.

## DeepEP Low Latency Backend

`deepep_low_latency`는 `DeepEPLLAll2AllManager`와
`DeepEPLLPrepareAndFinalize`가 담당한다.

prepare 흐름은 다음과 같다.

```text
global topk id -> physical expert id remap
  -> low_latency_dispatch(...)
       expert_x: [num_experts, max_tokens, hidden]
       expert_num_tokens
       handle
       recv hook
  -> 필요하면 quant/dequant/scale reshape
  -> batched experts kernel
```

finalize 흐름은 다음과 같다.

```text
fused_expert_output
  -> low_latency_combine(fused_expert_output, topk_ids, topk_weights, handle)
  -> output에 직접 write 가능
```

특징은 다음과 같다.

- `BatchedExperts` activation format을 사용한다.
- hidden size가 지원 목록에 맞아야 하며 필요하면 round up한다.
- combine kernel 안에서 top-k weight 적용과 reduction이 일어나는 경로가 있다.
- FP8 dispatch와 packed scale dispatch를 지원하는 경로가 있다.
- recv hook을 반환하므로 DBO나 async finalize와 엮기 좋다.
- RDMA 기반 low-latency mode에서는 communication이 SM을 거의 쓰지 않는 것으로
  모델링된다.

LL backend는 decode처럼 token 수가 작고 latency가 중요한 상황에 맞는 편이다.

## FlashInfer NVLink Backend

FlashInfer backend는 NVLink/MNNVL 시스템을 위한 one-sided/two-sided all-to-all
경로를 제공한다.

두 backend 모두 `all2all.py`에서 manager를 만들고,
`prepare_finalize/flashinfer_nvlink_*.py`에서 dispatch/combine을 호출한다.

일반적인 특징은 다음과 같다.

- MNNVL workspace를 EP group 크기에 맞게 잡는다.
- workspace는 max token, top-k, expert count, dispatch payload 크기에 의존한다.
- one-sided backend는 `MoeAlltoAll` 기반이며 dispatch payload에 hidden activation,
  scale, top-k id, top-k weight를 포함한다.
- two-sided backend는 별도 prepare workspace와 main workspace를 사용한다.

이 backend는 NVLink/NVSwitch/MNNVL topology가 맞을 때 의미가 크다. PCIe A40에서
관찰한 병목을 그대로 NVLink backend 가정으로 일반화하면 안 된다.

## NIXL, Mori, DeepEP V2

이 snapshot에는 추가 EP backend 경로도 있다.

- `nixl_ep`
  - elastic EP와 rank connect/disconnect를 염두에 둔 persistent buffer 구조가 있다.
  - staged EP size 변경과 commit 흐름이 있다.
- `mori_high_throughput`, `mori_low_latency`
  - Mori backend가 설치되어 있을 때 사용한다.
- `deepep_v2`
  - DeepEP V2 buffer와 CUDA graph 사용 여부를 반영한다.

이들은 모두 `maybe_make_prepare_finalize()`에서 `--all2all-backend` 값과 quant config에
따라 prepare/finalize 구현이 선택된다.

## Expert Compute Kernel

MoE expert compute는 dispatch 이후 rank-local이다. 즉 token이 local expert별로
정렬되거나 batched layout으로 들어오면, 해당 rank는 자기 GPU에서 GEMM kernel을 실행한다.

전형적인 expert MLP는 다음 구조다.

```text
x
  -> w13 GEMM
  -> activation, gate/up combine
  -> w2 GEMM
  -> top-k output layout
```

구현은 quantization과 GPU architecture에 따라 달라진다.

- Triton experts
- Batched Triton experts
- DeepGEMM experts
- CUTLASS FP8/FP4 experts
- Marlin experts
- FlashInfer experts
- GPT-OSS 특화 Triton experts
- ROCm AITER experts

expert compute의 성능은 token-per-expert 분포에 크게 좌우된다. top-k 라우팅이
skew되면 어떤 expert는 큰 GEMM을 수행하고 어떤 expert는 tiny GEMM만 수행한다. 이때
GPU occupancy, grouped GEMM scheduling, kernel launch overhead, ragged batch 처리 비용이
병목이 된다.

## EPLB

EPLB(Expert Parallel Load Balancer)는 expert routing skew를 줄이기 위한 경로다.

기본 흐름은 다음과 같다.

1. forward마다 expert load 통계를 모은다.
2. window/interval 기준으로 imbalance를 판단한다.
3. expert mapping 또는 redundant expert 배치를 조정한다.
4. 필요한 expert weight를 EP/EPLB communicator로 이동한다.

EPLB group은 EP와 같은 rank set이지만 별도 process group이다. 이유는 forward pass의
MoE collective와 load-balancing weight transfer가 같은 group에서 섞이면 deadlock
위험이 있기 때문이다.

EPLB는 memory overhead가 있다. redundant expert를 추가하면 각 rank가 더 많은 expert
weight를 들고 있어야 하므로 KV cache 예산을 줄일 수 있다.

## DBO와 Overlap

DBO(Dual Batch Overlap)가 켜지면 MoE prepare/finalize가 async hook 형태로 실행될 수
있다.

DeepEP HT의 prepare는 예를 들어 다음 식으로 overlap을 시도한다.

```text
compute stream work 기록
  -> comm stream으로 yield
  -> dispatch kernel launch
  -> receiver callback 반환
  -> 다른 ubatch compute와 겹침
  -> receiver가 event wait 후 expert_x 사용
```

DeepEP LL도 dispatch/combine에서 recv hook을 반환한다.

overlap이 잘 되려면 다음 조건이 중요하다.

- dispatch/combine이 CPU를 오래 block하지 않아야 한다.
- comm stream과 compute stream의 event dependency가 과도하지 않아야 한다.
- ubatch별 handle이 섞이지 않아야 한다.
- CUDA graph capture가 dynamic allocation이나 hidden sync로 깨지지 않아야 한다.

## CUDA Graph와 torch.compile 관점

vLLM은 CUDA graph와 torch.compile을 적극적으로 사용한다. multi-GPU collective와 MoE는
graph capture를 어렵게 만드는 요소가 많기 때문에 여러 장치가 들어간다.

주요 장치는 다음과 같다.

- `GroupCoordinator.graph_capture()`
  - custom all-reduce capture context를 열고 graph buffer를 등록한다.
- collective custom op
  - `torch.ops.vllm.all_reduce`, `all_gather`, `reduce_scatter` 등으로 감싸
    Dynamo가 group object를 직접 다루지 않게 한다.
- MoE custom op
  - `torch.ops.vllm.moe_forward`가 runner 내부 구현을 호출한다.
  - Python object lookup은 forward context의 layer name registry를 통해 한다.
- DeepEP/FlashInfer/NIXL workspace
  - model shape와 max token 기준으로 buffer를 미리 준비한다.

CUDA graph에서 multi-GPU 경로를 안정적으로 잡으려면 buffer 주소, tensor shape,
collective sequence, stream dependency가 반복 실행마다 같아야 한다. MoE는 token routing
때문에 shape와 token-per-expert가 흔들리므로, backend마다 max token, batched layout,
handle cache, recv hook 같은 장치를 둔다.

## 성능을 볼 때의 Mental Model

### Dense TP

병목 후보는 대개 다음 순서로 본다.

1. row-parallel projection 뒤 TP all-reduce
2. logits gather/all-gather
3. attention backend 자체의 rank-local kernel
4. KV cache read/write bandwidth
5. scheduler와 CUDA graph replay overhead

TP all-reduce payload는 대략 다음과 같이 잡을 수 있다.

```text
num_tokens * hidden_size * dtype_bytes
```

row-parallel output마다 이 규모의 tensor가 TP group에서 all-reduce된다.

### MoE EP

MoE EP 병목 후보는 더 많다.

1. router/gate GEMM과 top-k
2. top-k id/weight metadata 처리
3. token permute/packing/quantization
4. dispatch all-to-all 또는 AG/RS
5. local expert grouped GEMM
6. combine all-to-all 또는 reduce-scatter
7. unpermute/top-k weighted reduce
8. shared experts와 final all-reduce
9. DBO overlap 실패와 stream wait

특히 A40 PCIe 환경에서는 dispatch/combine 통신이 크게 보일 가능성이 높다. 하지만
A100 SXM/NVLink에서는 병목이 grouped GEMM, packing, scheduler overhead로 이동할 수
있다. 따라서 A40에서 관찰한 최적화를 NVLink 환경에 그대로 고정하면 위험하다.

## Profiling 체크리스트

multi-GPU kernel을 분석할 때는 결과마다 다음을 같이 기록한다.

- GPU 모델, GPU 수, PCIe/NVLink topology: `nvidia-smi topo -m`
- CUDA driver/toolkit, PyTorch, vLLM commit
- `tensor_parallel_size`, `pipeline_parallel_size`, `data_parallel_size`
- `--enable-expert-parallel`, `--all2all-backend`
- `--enable-dbo`, CUDA graph mode
- NCCL env: `NCCL_DEBUG`, `NCCL_ALGO`, `NCCL_PROTO`, IB/RDMA 관련 env
- MoE env: DeepEP/FlashInfer/NIXL backend 관련 env
- batch shape: prefill/decode, token count, top-k, hidden/intermediate size
- expert distribution: tokens per expert, per-rank imbalance
- benchmark command와 warmup/measurement 반복 수

Nsight Systems에서는 다음을 먼저 확인한다.

- NCCL all-reduce가 row-parallel projection 뒤에 serialize되는지
- MoE dispatch와 expert GEMM 사이에 stream wait가 긴지
- combine 뒤 unpermute/reduce가 CPU sync를 유발하는지
- DBO가 실제로 comm/compute를 겹치는지
- CUDA graph replay가 되는지, eager fallback이 섞이는지

Nsight Compute는 Nsight Systems에서 hotspot으로 확인된 kernel에만 적용하는 편이
좋다. MoE에서는 grouped GEMM, permute/unpermute, top-k, quantization kernel이 후보가
된다.

## 예시 1: TP=2 Dense Layer

```text
rank 0 owns qkv/o/mlp shard 0
rank 1 owns qkv/o/mlp shard 1

hidden replicated
  -> qkv column GEMM on each rank
  -> attention local heads
  -> o_proj row GEMM partial output
  -> TP all-reduce
  -> gate/up column GEMM
  -> local activation
  -> down row GEMM partial output
  -> TP all-reduce
```

이 경우 GPU 간 kernel은 mostly all-reduce다. attention kernel 자체는 local shard에서
돈다.

## 예시 2: TP=1, DP=2, EP=True MoE

```text
rank 0 has DP token chunk A and owns experts subset 0
rank 1 has DP token chunk B and owns experts subset 1

router chooses global experts for A/B
  -> dispatch sends tokens to expert-owner ranks
  -> each rank computes local experts
  -> combine sends weighted output back to original token owners
```

이 경우 MoE layer의 핵심 통신은 dispatch/combine이다. `all2all_backend` 선택이 성능에
직접적이다.

## 예시 3: TP=2, DP=1, EP=True MoE

```text
rank 0 and rank 1 both see the same token batch
rank 0 owns experts subset 0
rank 1 owns experts subset 1

router chooses global experts
  -> 각 rank는 local expert contribution만 계산
  -> final output all-reduce로 contribution 합산
```

이 경우 EP이지만 DP가 1이므로 DeepEP all-to-all 경로가 아닐 수 있다. 병목은 local
expert compute와 final all-reduce 쪽에 더 가깝다.

## 흔한 오해

### "multi-GPU kernel"은 CUDA kernel 하나가 여러 GPU에서 도는 것인가?

대부분 아니다. 각 GPU에서 같은 Python forward가 rank별로 실행되고, 각 rank-local CUDA
kernel 사이를 collective가 연결한다. NCCL/DeepEP/FlashInfer 같은 communication kernel은
여러 GPU가 참여하지만, dense attention/GEMM kernel은 rank-local이다.

### EP는 항상 all-to-all인가?

아니다. vLLM 코드에서는 all-to-all MoE kernel 사용 조건이 `dp_size > 1 and use_ep`에
가깝다. DP가 1이면 모든 rank가 같은 token batch를 볼 수 있으므로 expert contribution을
계산한 뒤 all-reduce하는 경로가 가능하다.

### FlashMoE를 그대로 port하면 해결되는가?

그럴 수도 있지만 보장되지 않는다. vLLM 병목이 all-to-all인지, ragged grouped GEMM인지,
packing인지, scheduler/CUDA graph인지 먼저 분리해야 한다. FlashMoE식 scheduling이나
fused combine이 도움이 되는지는 vLLM shape와 serving workload에서 검증해야 한다.

### A40 PCIe 최적화가 A100 SXM에도 맞는가?

항상 그렇지 않다. PCIe에서는 communication이 두드러져도, NVLink에서는 compute나
packing이 병목이 될 수 있다. 이 workspace의 목표처럼 A40은 재현 가능한 개발/측정
장비로 쓰고, A100 SXM에서는 같은 sweep을 짧게 재실행해 bottleneck 이동을 확인해야 한다.

## 빠른 코드 추적 경로

dense TP all-reduce를 보고 싶으면 다음 순서로 따라가면 된다.

```text
RowParallelLinear.forward
  -> tensor_model_parallel_all_reduce
  -> get_tp_group().all_reduce
  -> CudaCommunicator.all_reduce
  -> custom/PyNCCL/torch backend
```

MoE EP dispatch를 보고 싶으면 다음 순서가 좋다.

```text
MoERunner.forward
  -> torch.ops.vllm.moe_forward
  -> MoERunner._forward_impl
  -> router.select_experts
  -> RoutedExperts.forward_modular
  -> FusedMoEKernel.apply
  -> PrepareAndFinalize.prepare
  -> all2all manager dispatch
  -> experts kernel
  -> PrepareAndFinalize.finalize
  -> all2all manager combine
```

backend 선택을 보고 싶으면 다음 순서다.

```text
FusedMoE(...)
  -> FusedMoEParallelConfig.make
  -> RoutedExperts._get_quant_method
  -> maybe_make_prepare_finalize
  -> all2all backend-specific PrepareAndFinalize
  -> quant method selects experts kernel
```

## 이 workspace의 MoE EP 최적화 관점

현재 workspace의 목표는 vLLM multi-GPU MoE/EP 성능 개선이다. 따라서 다음 순서로
작업하는 것이 안전하다.

1. 먼저 timing/NVTX를 추가해 routing, prepare, dispatch, expert GEMM, combine,
   finalize를 분리한다.
2. token-per-expert histogram과 rank imbalance를 항상 같이 기록한다.
3. A40 PCIe에서는 communication과 hidden sync를 먼저 의심하되, A100 SXM에서는
   bottleneck이 이동할 수 있음을 전제로 한다.
4. FlashMoE-style 변경은 standalone benchmark와 vLLM shape parity를 거친 뒤,
   명시적 backend/env flag 뒤에 둔다.
5. kernel microbenchmark win보다 end-to-end serving throughput/latency 개선을 우선한다.
