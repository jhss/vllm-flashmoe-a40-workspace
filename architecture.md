# vLLM Multi-GPU MoE EP — 최적화 경로 다이어그램

작성일: 2026-06-22

이 문서는 `architecture.html`(인터랙티브 다이어그램)의 텍스트 설명이다. 이번
A100 SXM MoE Expert Parallel 실험의 커밋들이 **multi-GPU MoE 추론의 어느
단계를, 어떻게 최적화했는지**를 forward 경로 위에 표시한다.

- 입력: `architecture.html` (브라우저에서 열기)
- 모드 토글: **Baseline ↔ Optimized** — 각 단계의 최적화 전/후 코드와 latency가 함께 바뀐다
- 실험 조건: synthetic Qwen3-like BF16, 2×A100-SXM4-80GB, hidden=2048,
  intermediate=768, experts=128, top-k=8, tokens=128~1024

## 여는 법

```bash
# Windows
start architecture.html
# macOS
open architecture.html
# Linux
xdg-open architecture.html
```

단축키: `Space` 재생/정지 · `←`/`→` step · `1`–`3` flow · `O` 모드 · `T` 테마 ·
`F` 전체화면 · `R` 레이아웃 리셋 · 노드를 드래그하면 위치 이동.

---

## 노드 (컴포넌트)

| 노드 | 역할(색) | 의미 |
|---|---|---|
| Client | 요청(mint) | API request, 생성 결과 수신 |
| GPU Worker | Scheduler/Worker(sky) | rank-local model runner forward |
| TP Attention | 연산 커널(magenta) | ColumnParallel qkv → head attention → RowParallel o_proj → all-reduce |
| Router | Router/Assign(amber) | gate GEMM → top-k=8 → topk_ids / topk_weights |
| Dispatch | 통신(violet) | DeepEP HT / AG-RS, token을 expert 소유 rank로 all-to-all |
| **Receiver Remap ★** | remap(orange) | local→global expert id 변환 — **최적화 ①** |
| Expert Assign | Router/Assign(amber) | `moe_align_block_size` → sorted_token_ids / expert_ids |
| **Expert GEMM ★** | 연산 커널(magenta) | W13 → SiLU·gate → W2 fused_moe_kernel — **최적화 ②** |
| **Combine ★** | 통신(violet) | top-k reduce + reduce-scatter combine — **최적화 ③** |
| Sampler | Scheduler/Worker(sky) | lm_head → logits → sampling |

★ = 이번 커밋이 손댄 최적화 지점.

---

## Flow 1 — 전체 forward 경로

request가 어떻게 rank-local forward가 되고, MoE를 거쳐 token으로 나오는지 top-down.

1. **요청 → rank-local forward** (`client → worker`): scheduler가 batch를 GPU worker로.
   GPU마다 worker process 1개, 각자 자기 rank의 forward 실행.
2. **TP Attention block** (`worker → attn`): ColumnParallel qkv → rank-local head
   attention → RowParallel o_proj → TP all-reduce.
3. **MoE layer 진입 · Router** (`attn → router`): gate → router_logits → top-k=8.
4. **Dispatch (EP all-to-all)** (`router → dispatch`): token을 expert 소유 rank로.
   EP 시 rank0=expert 0~63, rank1=64~127.
5. **Expert compute (W13 → W2)** (`dispatch → experts`): rank-local expert GEMM.
   - Baseline: Triton base config, 2GPU EP full forward(t=1024) ~1763 us
   - Optimized: A100 tuned config, ~1613 us (**~8% ↓**)
6. **Finalize / Combine** (`experts → combine`): top-k weight 적용 후 reduce-scatter.
   - Baseline: `output.copy_(tmp)`, finalize 0.120 ms
   - Optimized: `combine(out=output)` in-place, finalize 0.101 ms
7. **Sampling** (`combine → sampler`): lm_head → logits → next token.
8. **Response** (`sampler → client`): token 반환 (decode면 반복).

## Flow 2 — MoE EP block (DeepEP HT 상세)

MoE receiver 경로를 5단계로 확대. 이번 실험의 무대.

1. **Router top-k 선택** (`router → dispatch`): topk_ids [128,8], topk_weights [128,8].
2. **DeepEP HT dispatch (all-to-all)** (`dispatch → remap`): `recv_x`(token 목록)와
   `recv_topk_idx`(local id / -1 혼재)를 받음. recv_x는 expert-contiguous가 아님.
3. **① Receiver top-k remap ★** (`remap → assign`): local→global id 변환.
   - Baseline: `torch.where`(새 tensor) — 0.089 ms
   - Optimized: Triton in-place (`VLLM_DEEPEP_HT_TRITON_TOPK_REMAP=1`) — 0.049 ms (~40us ↓)
4. **② Expert assignment** (`assign → experts`): `moe_align_block_size`로
   sorted_token_ids / expert_ids / num_tokens_post_padded 생성. recv_x가
   expert-contiguous가 아니라 align 비용이 핵심 — `VLLM_DEEPEP_HT_LOCAL_EXPERT_IDS`
   실험이 성능 중립이었던 이유 (다음 단계는 DeepEP HT 전용 assignment kernel).
5. **③ Expert GEMM + Combine ★** (`experts → combine`): W13 → SiLU·gate → W2 후 combine.
   - Baseline: base config + copy combine — finalize 0.120 ms
   - Optimized: A100 tuned config + in-place combine — expert ~8% ↓, finalize 0.101 ms

## Flow 3 — 최적화 포인트

이번 커밋이 손댄 3곳만 모은 tour. 모드 토글로 전/후 비교.

1. **① DeepEP HT top-k remap** (`dispatch → remap`):
   `VLLM_DEEPEP_HT_TRITON_TOPK_REMAP` — receiver remap 0.089→0.049 ms,
   full forward 1.414→1.386 ms.
2. **② W1/W2 A100 tuned config** (`remap → experts`):
   `VLLM_MOE_TRITON_W1/W2_A100_TUNED_CONFIG` — SM80 BF16 t≥1024 전용, W2 단독 -25%,
   2GPU EP ~8%. correctness max_abs=0.
3. **③ AG/RS in-place combine** (`experts → combine`):
   `reduce_scatterv(out=output)` — finalize 0.120→0.101 ms (full forward는
   NCCL collective latency가 지배라 노이즈 수준).

---

## Baseline ↔ Optimized 모드 차이

| 단계 | Baseline | Optimized | flag |
|---|---|---|---|
| Receiver remap | `torch.where` 0.089 ms | Triton in-place 0.049 ms | `VLLM_DEEPEP_HT_TRITON_TOPK_REMAP` |
| Expert GEMM | base cfg, 2GPU EP ~1763 us | A100 tuned, ~1613 us (~8%↓) | `VLLM_MOE_TRITON_W1/W2_A100_TUNED_CONFIG` |
| Combine | alloc + copy, 0.120 ms | in-place reduce-scatter, 0.101 ms | `combine(out=output)` |

## 핵심 메시지

- **decode**: communication / launch floor가 병목 (작고 sparse한 expert batch).
- **prefill**: expert GEMM(`fused_moe_kernel`)과 kernel boundary가 병목.
- receiver remap은 작은 승리, in-place combine은 노이즈 수준, **W1/W2 config tuning이
  지금까지 가장 큰 단일 승리(~8%)**.
- 다음 방향: DeepEP HT 전용 expert-assignment kernel, prefill 전용 fusion
  (GEMM2 epilogue + top-k reduce), AG/RS communication–compute overlap.

> 주의: latency 수치는 매 section마다 synchronize를 넣는 section profiler 기준으로,
> 상대적 병목 위치 파악용이다. 절대 end-to-end serving latency와는 다르다.
