# vLLM Multi-GPU Kernel 동작 방식: Top-Down 설명

작성일: 2026-06-19

이 문서는 vLLM에서 multi-GPU 실행이 어떤 흐름으로 내려가며 동작하는지
top-down 관점에서 설명한다. 핵심은 다음 한 문장이다.

> vLLM은 하나의 거대한 multi-GPU CUDA kernel을 실행하는 것이 아니라, GPU마다
> 별도 rank-local kernel을 실행하고 그 사이를 collective communication kernel이
> 연결한다.

따라서 multi-GPU 동작을 볼 때는 "어떤 CUDA kernel이 여러 GPU를 동시에 쓰는가"보다
"각 rank가 어떤 local 계산을 하고, 어느 지점에서 어떤 collective로 데이터를
교환하는가"를 따라가는 것이 훨씬 정확하다.

## 0. 전체 흐름 한눈에 보기

vLLM V1에서 하나의 요청이 GPU kernel 실행까지 내려가는 큰 흐름은 다음과 같다.

```text
API request
  -> engine core / scheduler
  -> executor
  -> GPU worker process per GPU
  -> model runner
  -> PyTorch module forward
  -> layer별 rank-local CUDA/Triton kernel
  -> 필요한 지점에서 NCCL/custom/DeepEP/FlashInfer collective
  -> rank별 output 조합
  -> sampling / response
```

이 흐름은 크게 여섯 단계로 나누어 볼 수 있다.

| 단계 | 질문 | 핵심 구성요소 |
| --- | --- | --- |
| 1. 실행 단위 생성 | GPU마다 누가 forward를 실행하는가? | executor, GPU worker, rank |
| 2. 통신 그룹 구성 | 어떤 rank끼리 통신하는가? | WORLD, TP, PP, DP, EP, DCP, PCP |
| 3. dense layer 실행 | 일반 transformer layer는 어떻게 나뉘는가? | ColumnParallelLinear, RowParallelLinear, TP collective |
| 4. attention/context 실행 | attention은 local인가, context를 나누는가? | FlashAttention/FlashInfer, KV cache, DCP/PCP |
| 5. MoE layer 실행 | token을 expert 소유 rank로 어떻게 보내는가? | router, prepare, dispatch, expert GEMM, combine |
| 6. 성능/graph 관리 | overlap, CUDA graph, dynamic shape는 어떻게 다루는가? | DBO, custom op, workspace, communicator |

아래는 같은 내용을 더 구체적인 forward 관점으로 펼친 그림이다.

```text
rank-local model forward
  -> dense block이면
       local GEMM / attention
       TP all-reduce 또는 all-gather
  -> MoE block이면
       router top-k
       prepare/dispatch
       local expert GEMM
       finalize/combine
       필요 시 TP/EP/DP reduce
  -> logits 처리이면
       vocab-parallel logits gather 또는 local top-k/argmax gather
```

## 1. 실행 단위: 요청은 어떻게 rank-local forward가 되는가

사용자 요청은 먼저 engine과 scheduler를 거쳐 GPU worker들에게 전달된다. multi-GPU
환경에서는 보통 GPU마다 하나의 worker process가 있고, 각 process는 자기 rank와
local GPU를 가진다.

```text
executor
  -> worker process rank 0 -> cuda:0
  -> worker process rank 1 -> cuda:1
  -> ...
```

이 단계의 주요 구성요소는 다음과 같다.

| 구성요소 | 역할 | 주요 코드 |
| --- | --- | --- |
| executor | worker process 생성과 RPC/작업 분배 | `vllm/v1/executor/multiproc_executor.py`, `ray_executor*.py` |
| GPU worker | device 선택, distributed 초기화, model runner 보유 | `vllm/v1/worker/gpu_worker.py` |
| model runner | 실제 model forward 실행 | `vllm/v1/worker/gpu_model_runner.py` 계열 |
| distributed init | torch distributed와 model-parallel group 초기화 | `vllm/distributed/parallel_state.py` |

GPU worker 초기화 흐름은 다음과 같이 내려간다.

```text
GPUWorker.init_device
  -> torch.accelerator.set_device_index(local_rank)
  -> init_worker_distributed_environment(...)
  -> init_distributed_environment(...)
  -> ensure_model_parallel_initialized(...)
  -> memory snapshot / KV cache budget 계산
```

중요한 점은 모든 rank가 같은 Python model forward 구조를 실행하지만, rank마다 가진
weight shard, expert shard, KV cache shard, process group membership이 다르다는 것이다.
이 차이가 multi-GPU 실행을 만든다.

## 2. 통신 그룹: 어떤 rank끼리 데이터를 주고받는가

rank-local kernel 사이의 데이터 이동은 `GroupCoordinator`가 감싼 process group을 통해
일어난다.

```text
layer code
  -> tensor_model_parallel_all_reduce(...)
  -> get_tp_group().all_reduce(...)
  -> GroupCoordinator
  -> CudaCommunicator
  -> NCCL / custom all-reduce / torch distributed fallback
```

`GroupCoordinator`는 한 parallel dimension에 대해 두 종류의 group을 관리한다.

| group 종류 | 역할 |
| --- | --- |
| `device_group` | GPU tensor collective용 group. CUDA에서는 NCCL 계열이 주로 사용된다. |
| `cpu_group` | metadata, object broadcast, setup, coordination용 Gloo group. |

vLLM이 만드는 주요 parallel group은 다음과 같다.

| group | 의미 | 주 사용처 |
| --- | --- | --- |
| `WORLD` | 전체 ranks | 초기화와 global coordination |
| `TP` | tensor parallel group | dense linear all-reduce/all-gather, vocab gather |
| `PP` | pipeline parallel group | stage 간 activation send/recv |
| `DP` | data parallel group | DP worker 간 token/output 분배, MoE AG/RS fallback |
| `EP` | expert parallel group | MoE expert ownership, dispatch/combine |
| `EPLB` | EP와 같은 rank set의 별도 group | expert load balancer weight 이동, deadlock 격리 |
| `PCP` | prefill context parallel group | prefill/context token 분산 |
| `DCP` | decode context parallel group | decode attention query all-gather/combine |

`initialize_model_parallel()`은 rank 배열을 보통 다음 logical layout으로 해석한다.

```text
ExternalDP x DP x PP x PCP x TP
```

각 group은 이 layout에서 특정 축을 기준으로 rank들을 묶어 만든다. 예를 들어 TP group은
마지막 TP 축을 따라 묶고, PP group은 pipeline 축을 따라 묶는다. MoE EP group은 MoE
설정에 따라 DP/PCP/TP 축을 가로질러 expert ownership을 만들 수 있다.

## 3. Dense Transformer 흐름: TP linear가 multi-GPU 비용의 중심이다

dense transformer block의 multi-GPU 흐름은 대략 다음과 같다.

```text
hidden states
  -> qkv projection
  -> attention
  -> output projection
  -> MLP gate/up projection
  -> activation
  -> MLP down projection
  -> residual / norm
```

TP가 켜져 있으면 각 projection은 column-parallel 또는 row-parallel 방식으로 나뉜다.
여기서 multi-GPU 통신이 생긴다.

### 3.1 ColumnParallelLinear: output dimension을 나눈다

`ColumnParallelLinear`는 weight의 output dimension을 TP rank별로 나눈다.

```text
Y = X A
A = [A_0, A_1, ..., A_{tp-1}]
rank i computes Y_i = X A_i
```

forward 흐름은 다음과 같다.

```text
input X
  -> rank-local GEMM
  -> output_parallel = Y_i
  -> gather_output=True이면 TP all-gather
  -> 아니면 shard 상태 유지
```

구성요소는 다음과 같다.

| 요소 | 설명 |
| --- | --- |
| local GEMM | 각 rank가 자기 weight shard만 계산한다. |
| output shard | output feature 일부만 rank에 존재한다. |
| optional all-gather | 다음 연산이 full hidden을 요구할 때만 shard를 모은다. |

QKV projection, gate/up projection처럼 다음 연산이 shard 상태를 자연스럽게 받을 수 있는
경우에는 gather하지 않는 편이 많다.

### 3.2 RowParallelLinear: input dimension을 나누고 결과를 합친다

`RowParallelLinear`는 weight의 input dimension을 TP rank별로 나눈다.

```text
X = [X_0, X_1, ..., X_{tp-1}]
A = [A_0; A_1; ...; A_{tp-1}]
Y = sum_i X_i A_i
```

forward 흐름은 다음과 같다.

```text
input이 이미 shard이면 그대로 사용
input이 full이면 rank별 slice 선택
  -> rank-local GEMM으로 partial output 계산
  -> reduce_results=True이면 TP all-reduce
  -> full output 복원
```

구성요소는 다음과 같다.

| 요소 | 설명 |
| --- | --- |
| input shard | 각 rank가 input hidden dimension 일부를 가진다. |
| partial output | 각 rank의 GEMM 결과는 최종 output의 partial sum이다. |
| TP all-reduce | partial output을 sum해서 모든 rank가 같은 output을 갖는다. |

attention output projection과 MLP down projection에서 이 all-reduce가 자주 보인다.

### 3.3 Dense block을 통째로 보면

일반적인 TP transformer block은 다음처럼 읽을 수 있다.

```text
hidden states replicated on TP ranks
  -> qkv_proj: ColumnParallelLinear
       rank-local GEMM, 보통 gather 없음
  -> attention
       rank-local heads / rank-local KV cache
  -> o_proj: RowParallelLinear
       rank-local GEMM
       TP all-reduce
  -> gate/up proj: MergedColumnParallelLinear
       rank-local GEMM, 보통 gather 없음
  -> activation/local MLP
  -> down_proj: RowParallelLinear
       rank-local GEMM
       TP all-reduce
```

따라서 dense TP의 주요 multi-GPU 비용은 attention kernel 자체라기보다 row-parallel
projection 뒤의 all-reduce인 경우가 많다.

## 4. Attention과 Context Parallel 흐름

일반적인 TP attention에서는 각 rank가 자기 head shard와 KV cache shard를 사용해
rank-local attention kernel을 실행한다.

```text
rank-local Q/K/V shard
  -> FlashAttention 또는 FlashInfer kernel
  -> rank-local attention output
```

context parallelism이 켜지면 attention 앞뒤에 추가 collective가 붙는다.

### 4.1 Decode Context Parallel(DCP)

DCP는 decode query/context를 여러 rank에 나누는 경로다.

```text
rank-local query
  -> get_dcp_group().all_gather(query)
  -> FlashAttention/FlashInfer decode wrapper
  -> partial output/LSE combine
```

구성요소는 다음과 같다.

| 요소 | 설명 |
| --- | --- |
| DCP all-gather | 각 rank의 query/context 조각을 attention 계산에 필요한 형태로 모은다. |
| attention backend | FlashAttention 또는 FlashInfer wrapper가 실행된다. |
| DCP combine | partial attention output과 LSE를 합친다. |

### 4.2 Prefill Context Parallel(PCP)

PCP는 prefill context를 나누는 경로다. MoE runner에도 영향을 준다.

```text
rank-local hidden/router logits
  -> PCP all-gather
  -> layer 또는 MoE 계산
  -> 필요하면 reduce-scatter
```

PCP가 켜져 있으면 token dimension의 layout이 바뀌므로, MoE routing과 dispatch에서도
token ownership을 더 조심해서 다뤄야 한다.

## 5. MoE 흐름: router에서 dispatch/combine까지

MoE layer는 dense layer보다 한 단계 더 복잡하다. dense TP가 weight shard의 partial
sum을 합치는 문제라면, MoE EP는 token을 expert 소유 rank로 보내는 문제다.

MoE forward의 top-down 흐름은 다음과 같다.

```text
hidden_states
  -> router
       router_logits -> topk_ids, topk_weights
  -> prepare
       quantization / metadata / dispatch
  -> expert assignment
       sorted_token_ids / expert_ids / num_tokens_post_padded
  -> local expert compute
       w13 GEMM -> activation -> w2 GEMM
  -> finalize
       top-k weight / reduce / combine
  -> shared experts and final reductions
```

이 흐름은 `modular_kernel.py`에서 다음 컴포넌트로 분리된다.

```text
[Router] -> [Prepare/Dispatch] -> [Experts] -> [Finalize/Combine]
```

### 5.1 MoE 병렬 전략 결정

먼저 `FusedMoEParallelConfig.make()`가 MoE layer의 병렬 전략을 정한다.

```text
model config / parallel config
  -> enable_expert_parallel 여부 확인
  -> local/global expert 수 계산
  -> expert_map 구성
  -> all2all backend 사용 여부 결정
```

EP가 꺼져 있으면 expert weight도 dense MLP처럼 tensor-sharded될 수 있다.

```text
expert tensor shard
  -> rank-local partial expert GEMM
  -> 필요한 곳에서 TP all-reduce
```

EP가 켜지면 각 rank가 expert를 통째로 소유한다.

```text
rank 0 owns experts 0..63
rank 1 owns experts 64..127
...
```

이 경우 하나의 expert GEMM을 여러 rank가 나누어 계산하는 것이 아니라, token을 그
expert를 가진 rank로 보내고 local expert GEMM을 실행한다.

주의할 점은 EP라고 항상 all-to-all인 것은 아니라는 것이다. vLLM 코드에서는
`dp_size > 1 and use_ep`일 때 all-to-all MoE backend가 핵심 경로가 된다. DP가 1이면
모든 rank가 같은 token batch를 볼 수 있으므로, 각 rank가 local expert contribution만
계산하고 마지막에 all-reduce로 합치는 경로가 가능하다.

### 5.2 Router: token마다 어떤 expert로 갈지 정한다

router 단계는 token hidden state에서 expert 선택 정보를 만든다.

```text
hidden_states
  -> gate/router projection
  -> router_logits
  -> top-k selection
  -> topk_ids, topk_weights
```

구성요소는 다음과 같다.

| 요소 | 설명 |
| --- | --- |
| `topk_ids` | 각 token이 선택한 expert id. shape은 보통 `[num_tokens, top_k]`. |
| `topk_weights` | 선택된 expert output을 섞을 때 사용할 routing weight. |
| expert map | global expert id를 local expert id 또는 invalid로 바꾸는 mapping. |

라우팅 분포는 이후 성능을 크게 좌우한다. 특정 expert에 token이 몰리면 grouped GEMM의
batch가 커지고, 반대로 많은 expert가 tiny batch만 받으면 launch/occupancy 손실이 커진다.

### 5.3 Prepare/Dispatch: token을 expert 소유 rank로 보낸다

prepare 단계는 backend에 따라 모양이 다르지만, 목표는 같다.

```text
hidden_states, topk_ids, topk_weights
  -> backend가 요구하는 layout으로 변환
  -> 필요하면 activation quantization
  -> token과 routing metadata를 expert 소유 rank로 dispatch
  -> expert_tokens_meta 생성
```

주요 구성요소는 다음과 같다.

| 요소 | 설명 |
| --- | --- |
| activation layout | `Standard` 또는 `BatchedExperts`. experts kernel과 맞아야 한다. |
| quantization | FP8/block quant 등 backend와 expert kernel 제약에 따라 prepare 전후로 수행된다. |
| dispatch collective | AG/RS, DeepEP, FlashInfer, NIXL, Mori 등이 token 이동을 담당한다. |
| `ExpertTokensMetadata` | local expert별 token count 같은 schedule metadata를 담는다. |

### 5.4 Expert assignment: GEMM이 읽을 schedule을 만든다

expert compute kernel은 token을 expert별 block으로 정렬한 schedule을 기대한다. 일반
Triton MoE 경로는 다음 ABI를 사용한다.

```text
sorted_token_ids
expert_ids
num_tokens_post_padded
```

각 값의 의미는 다음과 같다.

| 값 | 의미 |
| --- | --- |
| `sorted_token_ids` | top-k expanded token pair id를 expert별로 정렬한 배열. padding은 sentinel. |
| `expert_ids` | block마다 어떤 local expert weight를 읽을지 나타내는 배열. invalid block은 `-1`. |
| `num_tokens_post_padded` | `BLOCK_SIZE_M`에 맞게 padding된 총 token-pair 수. |

기존 generic 경로는 `moe_align_block_size()`를 통해 `topk_ids`를 스캔하고 expert별로
정렬한다. DeepEP HT direct assignment 실험 경로는 DeepEP가 이미 준 local expert별
token count를 사용해 이 schedule을 직접 만든다.

```text
DeepEP HT expert_topk_ids + expert_num_tokens
  -> expert별 padded count와 offset 계산
  -> sorted_token_ids / expert_ids 작성
  -> Triton/A100 MoE GEMM ABI로 전달
```

이 workspace의 direct assignment 경로는 feature flag 뒤에 있고, BF16 top-k=8,
local experts=64, no quant, no LoRA 같은 보수 조건에서만 켜진다. 조건이 맞지 않으면
기존 `_prepare_expert_assignment()`로 fallback한다.

### 5.5 Local expert compute: 이제 각 rank의 GPU kernel이다

dispatch와 assignment가 끝나면 expert compute는 다시 rank-local kernel이다.

```text
expert input layout
  -> w13 GEMM
  -> activation / gate-up combine
  -> w2 GEMM
  -> top-k expanded output
```

구현은 GPU, dtype, quantization에 따라 달라진다.

| experts 구현 | 특징 |
| --- | --- |
| Triton experts | 일반 fused MoE 경로. `sorted_token_ids` ABI를 사용한다. |
| A100 BF16 specialized kernel | 특정 A100 BF16 shape에 맞춘 실험/특화 경로. |
| Batched Triton experts | `[expert, max_tokens, hidden]` batched layout을 사용한다. |
| DeepGEMM/CUTLASS/Marlin/FlashInfer experts | FP8/FP4/int quant 또는 특정 architecture에 맞춘 경로. |
| ROCm AITER experts | ROCm 환경용 expert kernel. |

성능은 token-per-expert 분포에 민감하다. 같은 총 token 수라도 expert별 token batch가
ragged하면 grouped GEMM occupancy와 memory locality가 나빠질 수 있다.

### 5.6 Finalize/Combine: expert output을 원래 token owner로 돌린다

finalize 단계는 expert output을 원래 token layout으로 되돌린다.

```text
local expert output
  -> top-k weight 적용
  -> top-k 축 reduce
  -> combine collective
  -> original token owner rank의 output
```

어떤 backend는 top-k weight/reduce를 experts kernel 안에서 일부 처리하고, 어떤 backend는
finalize에서 처리한다. 이 책임 분리는 `TopKWeightAndReduce` abstraction으로 표현된다.

## 6. MoE all-to-all backend별 흐름

MoE prepare/finalize의 내부 흐름은 `--all2all-backend`에 따라 달라진다.

### 6.1 allgather_reducescatter: 단순하지만 넓게 모은다

`allgather_reducescatter` backend는 `AgRsAll2AllManager`가 담당한다.

dispatch 관점:

```text
각 DP rank의 hidden_states/topk_ids/topk_weights
  -> all_gatherv로 모든 DP rank token을 모음
  -> 각 rank가 local expert에 해당하는 token만 계산
```

combine 관점:

```text
local expert output
  -> top-k weight/reduce
  -> reduce_scatterv로 원래 DP rank별 token chunk로 돌려줌
```

장점은 일반적이고 단순하다는 점이다. 단점은 모든 token을 넓게 모으기 때문에 EP size가
커질수록 불필요한 데이터 이동과 metadata 처리가 커질 수 있다는 점이다.

### 6.2 DeepEP High Throughput: token dispatch/combine 전용 backend

`deepep_high_throughput`은 `DeepEPHTAll2AllManager`와
`DeepEPHTPrepareAndFinalize`가 담당한다.

prepare 흐름:

```text
optional input quantization
  -> DeepEP get_dispatch_layout(topk_ids)
  -> DeepEP dispatch(tokens, scales, topk_ids, topk_weights)
  -> expert_x, expert_topk_ids, expert_topk_weights, token counts, handle
  -> local/global expert id 처리
  -> expert assignment
  -> experts kernel
```

finalize 흐름:

```text
fused_expert_output
  -> top-k weight/reduce를 local에서 적용
  -> DeepEP combine(handle)
  -> original token layout output
```

구성요소는 다음과 같다.

| 요소 | 설명 |
| --- | --- |
| dispatch handle | combine이 원래 token owner로 되돌릴 때 필요한 DeepEP handle. |
| ubatch별 handle | DBO microbatching에서 두 ubatch가 섞이지 않도록 따로 저장한다. |
| local top-k id | DeepEP dispatch 뒤에는 expert id가 receiver-local id일 수 있다. |
| token counts | local expert별 token 수. direct assignment 최적화의 입력이 된다. |
| `VLLM_DEEPEP_HT_NUM_SMS` | DeepEP communication kernel이 사용할 SM 수를 조정한다. |

HT backend는 prefill처럼 token 수가 많고 throughput이 중요한 상황에 맞는 편이다.

### 6.3 DeepEP Low Latency: batched expert layout 중심

`deepep_low_latency`는 `DeepEPLLAll2AllManager`와
`DeepEPLLPrepareAndFinalize`가 담당한다.

prepare 흐름:

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

finalize 흐름:

```text
fused_expert_output
  -> low_latency_combine(fused_expert_output, topk_ids, topk_weights, handle)
  -> output에 직접 write 가능
```

LL backend는 decode처럼 token 수가 작고 latency가 중요한 상황에 맞는 편이다.

### 6.4 FlashInfer NVLink, NIXL, Mori, DeepEP V2

다른 backend들도 같은 prepare/experts/finalize 틀 안에 들어온다.

| backend | top-down 역할 |
| --- | --- |
| FlashInfer NVLink one-sided/two-sided | NVLink/MNNVL topology에서 one-sided 또는 two-sided all-to-all 수행 |
| NIXL EP | elastic EP와 persistent buffer, rank connect/disconnect를 고려한 경로 |
| Mori HT/LL | Mori backend가 설치된 환경에서 high-throughput/low-latency dispatch 제공 |
| DeepEP V2 | DeepEP V2 buffer와 CUDA graph 사용 여부를 반영한 경로 |

이들은 모두 `maybe_make_prepare_finalize()`에서 backend 이름, quant config, parallel config에
따라 선택된다.

## 7. Pipeline Parallel과 stage 간 흐름

PP는 layer stage를 rank group에 나누는 방식이다.

```text
stage 0 ranks
  -> local layers forward
  -> activation send
stage 1 ranks
  -> activation recv
  -> local layers forward
```

stage 내부에서는 앞서 설명한 TP/MoE/attention 흐름이 그대로 적용된다. PP는 kernel 하나의
모양을 바꾼다기보다 layer 실행 순서와 stage 간 activation 이동을 바꾸는 병렬화다.

## 8. CUDA communication backend는 어떻게 선택되는가

TP all-reduce 같은 dense collective는 `CudaCommunicator`가 backend를 고른다.

```text
GroupCoordinator.all_reduce
  -> CudaCommunicator.all_reduce
  -> backend capability check
  -> selected communication kernel
```

대략적인 후보 순서는 다음과 같다.

1. NCCL symmetric memory all-reduce custom op
2. ROCm quick reduce
3. FlashInfer all-reduce
4. vLLM custom all-reduce
5. torch symmetric memory communicator
6. PyNcclCommunicator
7. `torch.distributed.all_reduce` fallback

각 backend는 dtype, tensor size, world size, topology, env flag에 따라 거절될 수 있다.
예를 들어 vLLM custom all-reduce는 single-node에서 빠른 TP all-reduce를 노리지만,
world size, P2P access, multi-node 여부, tensor alignment 같은 제약이 있다.

## 9. CUDA graph, torch.compile, DBO는 왜 중요한가

multi-GPU 경로는 Python object, dynamic routing, collective sequence, stream dependency가
많아서 CUDA graph와 torch.compile에 부담을 준다. vLLM은 이 부담을 줄이기 위해 여러
경계를 둔다.

```text
torch module forward
  -> custom op boundary
  -> preallocated workspace
  -> fixed collective sequence
  -> graph replay
```

주요 구성요소는 다음과 같다.

| 요소 | 역할 |
| --- | --- |
| collective custom op | `torch.ops.vllm.all_reduce`, `all_gather` 등으로 Dynamo가 group object를 직접 다루지 않게 한다. |
| MoE custom op | `torch.ops.vllm.moe_forward`가 dynamic MoE 구현을 하나의 op 경계로 감싼다. |
| graph capture context | custom all-reduce buffer 등록과 capture-time metadata를 관리한다. |
| workspace | DeepEP/FlashInfer/NIXL 등이 max token과 shape 기준으로 buffer를 미리 준비한다. |
| DBO | 두 microbatch의 communication과 compute를 겹치려는 overlap 장치다. |

DBO가 켜진 DeepEP HT prepare는 다음 흐름을 노린다.

```text
compute stream work 기록
  -> comm stream으로 yield
  -> dispatch kernel launch
  -> receiver callback 반환
  -> 다른 ubatch compute와 overlap
  -> receiver가 event wait 후 expert_x 사용
```

overlap이 잘 되려면 dispatch/combine이 CPU를 오래 block하지 않아야 하고, ubatch별 handle,
stream event dependency, CUDA graph capture 조건이 안정적이어야 한다.

## 10. 성능을 볼 때의 top-down mental model

성능 분석도 top-down으로 내려가면 덜 헷갈린다.

### 10.1 먼저 어떤 병렬화 흐름인지 분류한다

```text
dense TP인가?
  -> row-parallel all-reduce가 주요 후보

MoE EP인가?
  -> routing, dispatch, expert GEMM, combine을 분리

context parallel인가?
  -> query all-gather와 attention combine 확인

pipeline parallel인가?
  -> stage boundary send/recv와 bubble 확인
```

### 10.2 Dense TP 병목 후보

1. row-parallel projection 뒤 TP all-reduce
2. logits gather/all-gather
3. attention backend의 rank-local kernel
4. KV cache read/write bandwidth
5. scheduler와 CUDA graph replay overhead

TP all-reduce payload는 대략 다음 크기다.

```text
num_tokens * hidden_size * dtype_bytes
```

### 10.3 MoE EP 병목 후보

1. router/gate GEMM과 top-k
2. top-k id/weight metadata 처리
3. token permute/packing/quantization
4. dispatch all-to-all 또는 AG/RS
5. expert assignment schedule 생성
6. local expert grouped GEMM
7. combine all-to-all 또는 reduce-scatter
8. unpermute/top-k weighted reduce
9. shared experts와 final all-reduce
10. DBO overlap 실패와 stream wait

A40 PCIe에서는 dispatch/combine communication이 크게 보일 가능성이 높다. 하지만
A100 SXM/NVLink에서는 병목이 grouped GEMM, packing, metadata setup, scheduler overhead로
이동할 수 있다. 따라서 A40에서 찾은 최적화를 A100 SXM에 그대로 고정하면 위험하다.

## 11. Profiling 체크리스트

실험 결과마다 다음을 같이 기록한다.

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

Nsight Systems에서는 먼저 다음을 본다.

- row-parallel projection 뒤 NCCL all-reduce가 serialize되는지
- MoE dispatch와 expert GEMM 사이에 stream wait가 긴지
- combine 뒤 unpermute/reduce가 CPU sync를 유발하는지
- DBO가 실제로 communication과 compute를 겹치는지
- CUDA graph replay가 되는지, eager fallback이 섞이는지

Nsight Compute는 Nsight Systems에서 hotspot으로 확인된 kernel에만 적용하는 편이 좋다.
MoE에서는 grouped GEMM, permute/unpermute, top-k, quantization kernel이 후보가 된다.

## 12. 코드 추적 지도: 흐름별로 어디를 보면 되는가

### 12.1 실행과 group 초기화

```text
vllm/v1/executor/multiproc_executor.py
  -> vllm/v1/worker/gpu_worker.py
  -> vllm/distributed/parallel_state.py
```

### 12.2 dense TP all-reduce

```text
RowParallelLinear.forward
  -> tensor_model_parallel_all_reduce
  -> get_tp_group().all_reduce
  -> GroupCoordinator.all_reduce
  -> CudaCommunicator.all_reduce
  -> custom/PyNCCL/torch backend
```

관련 파일:

- `vllm/model_executor/layers/linear.py`
- `vllm/distributed/communication_op.py`
- `vllm/distributed/device_communicators/cuda_communicator.py`
- `vllm/distributed/device_communicators/custom_all_reduce.py`

### 12.3 vocab embedding/logits gather

```text
VocabParallelEmbedding
  -> local vocab lookup
  -> TP all-reduce

LogitsProcessor
  -> local vocab logits
  -> gather/all-gather 또는 local top-k/argmax gather
```

관련 파일:

- `vllm/model_executor/layers/vocab_parallel_embedding.py`
- `vllm/model_executor/layers/logits_processor.py`

### 12.4 MoE EP dispatch

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

관련 파일:

- `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`
- `vllm/model_executor/layers/fused_moe/modular_kernel.py`
- `vllm/model_executor/layers/fused_moe/all2all_utils.py`
- `vllm/model_executor/layers/fused_moe/prepare_finalize/*.py`
- `vllm/model_executor/layers/fused_moe/experts/*.py`

### 12.5 MoE backend 선택

```text
FusedMoE(...)
  -> FusedMoEParallelConfig.make
  -> RoutedExperts._get_quant_method
  -> maybe_make_prepare_finalize
  -> backend-specific PrepareAndFinalize
  -> quant method selects experts kernel
```

### 12.6 context parallel attention

```text
FlashAttention/FlashInfer backend
  -> DCP query all-gather
  -> attention wrapper
  -> DCP combine
```

관련 파일:

- `vllm/v1/attention/backends/flash_attn.py`
- `vllm/v1/attention/backends/flashinfer.py`

## 13. 예시로 보는 전체 흐름

### 예시 A: TP=2 dense layer

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

이 경우 GPU 간 communication은 대부분 row-parallel projection 뒤 all-reduce다.

### 예시 B: TP=1, DP=2, EP=True MoE

```text
rank 0 has DP token chunk A and owns experts subset 0
rank 1 has DP token chunk B and owns experts subset 1

router chooses global experts for A/B
  -> dispatch sends tokens to expert-owner ranks
  -> each rank computes local experts
  -> combine sends weighted output back to original token owners
```

이 경우 MoE layer의 핵심 통신은 dispatch/combine이다. `all2all_backend` 선택이 성능에
직접적인 영향을 준다.

### 예시 C: TP=2, DP=1, EP=True MoE

```text
rank 0 and rank 1 both see the same token batch
rank 0 owns experts subset 0
rank 1 owns experts subset 1

router chooses global experts
  -> each rank computes only local expert contribution
  -> final output all-reduce combines contributions
```

이 경우 EP이지만 DP가 1이므로 DeepEP all-to-all 경로가 아닐 수 있다. 병목은 local
expert compute와 final all-reduce 쪽에 더 가깝다.

## 14. 흔한 오해 정리

### "multi-GPU kernel"은 CUDA kernel 하나가 여러 GPU에서 도는 것인가?

대부분 아니다. 각 GPU에서 rank-local CUDA/Triton/GEMM kernel이 돌고, NCCL/DeepEP/
FlashInfer 같은 communication kernel이 rank 사이를 연결한다.

### EP는 항상 all-to-all인가?

아니다. DP가 1이면 모든 rank가 같은 token batch를 볼 수 있으므로 local expert
contribution을 계산한 뒤 all-reduce하는 경로가 가능하다. DP가 2 이상이고 expert
ownership이 rank를 가로지르면 dispatch/combine이 중요해진다.

### attention kernel이 multi-GPU 병목의 전부인가?

보통 아니다. dense TP에서는 attention 주변의 row-parallel all-reduce가 더 눈에 띌 수
있고, MoE EP에서는 dispatch/combine과 expert assignment, grouped GEMM이 더 중요할 수
있다.

### FlashMoE를 그대로 port하면 해결되는가?

보장되지 않는다. vLLM 병목이 all-to-all인지, ragged grouped GEMM인지, packing인지,
scheduler/CUDA graph인지 먼저 분리해야 한다. FlashMoE식 scheduling이나 fused combine이
도움이 되는지는 vLLM shape와 serving workload에서 검증해야 한다.

## 15. 이 workspace의 MoE EP 최적화 관점

현재 workspace의 목표는 vLLM multi-GPU MoE/EP 성능 개선이다. 따라서 다음 순서가
안전하다.

1. timing/NVTX를 추가해 routing, prepare, dispatch, expert assignment, expert GEMM,
   combine, finalize를 분리한다.
2. token-per-expert histogram과 rank imbalance를 항상 같이 기록한다.
3. A40 PCIe에서는 communication과 hidden sync를 먼저 의심하되, A100 SXM에서는 병목이
   grouped GEMM, packing, metadata setup, scheduler overhead로 이동할 수 있음을 전제로
   한다.
4. FlashMoE-style 변경은 standalone benchmark와 vLLM shape parity를 거친 뒤, 명시적
   backend/env flag 뒤에 둔다.
5. kernel microbenchmark win보다 end-to-end serving throughput/latency 개선을 우선한다.
