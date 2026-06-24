# DeepEP HT MoE — 실행 과정 & 최적화 (다이어그램 설명)

`architecture.html`(인터랙티브 다이어그램)의 텍스트 설명. vLLM **DeepEP High-Throughput
MoE Expert-Parallel** forward의 전체 실행 경로와, 이번 프로젝트가 재구성한 fast path를
보여준다. **Baseline ↔ Optimized** 모드 토글로 각 단계의 전/후를 비교한다.

- 환경: 2×A100-SXM, BF16, top-k=8, 128 experts, hidden=2048, intermediate=768, tokens 320/448
- 결과(paired same-session): **−15.51% (320t) / −14.22% (448t)**, 12/12 paired win

여는 법: 브라우저에서 `architecture.html` 열기. 단축키: `Space` 재생, `←/→` step,
`1`/`2` flow, `O` 모드(Baseline/Optimized), `T` 테마, `F` 전체화면.

---

## 노드 (실행 단계)

| 노드 | 역할(색) | 의미 |
|---|---|---|
| Router | embed | gate → top-k=8 → topk_ids/topk_weights |
| Dispatch | vector(통신) | DeepEP HT all-to-all, token을 expert 소유 rank로 |
| **Receiver ★** | seed(주황) | recv_x / recv_topk_idx(local id·−1). **최적화 핵심** |
| **Assignment ★** | embed | moe_align → sorted_token_ids 등 GEMM schedule |
| **Expert GEMM ★** | compute | W13 → SiLU·gate → W2 (fused_moe_kernel ×2) |
| Combine | vector(통신) | top-k weighted reduce-scatter |

★ = mode(Baseline/Optimized)에 따라 동작이 바뀌는 최적화 지점.

---

## Flow 1 — 전체 실행 과정 (DeepEP HT MoE forward)

router → dispatch → receiver → assignment → expert GEMM → combine. 모드 토글로 전/후 비교.

1. **Dispatch (all-to-all)** (`router → dispatch`): top-k=8 선택 후 token을 expert 소유
   rank로 보냄. 128 experts를 2 rank가 64개씩 소유.
2. **Receiver ★** (`dispatch → receive`): recv_x + recv_topk_idx(local id / −1) 수신.
   - Baseline: dynamic receive + local→global **remap**
   - Optimized: **fixed-capacity**(capacity = tokens×world_size = 640/896) + **raw local id**(remap 없음). 실제 recv ≈ 638/892 → 활용률 ~99%
3. **Assignment ★** (`receive → align`): GEMM schedule 생성.
   - Baseline: generic align, invalid(non-local) block 포함, W13/W2 schedule 2번
   - Optimized: `IGNORE_INVALID_EXPERTS=1`(+`MIN_TOKENS=0`), CUDA align이 raw local id/−1 직접 skip, BLOCK_M=64로 W13/W2 schedule 공유
4. **Expert GEMM ★** (`align → experts`): W13 → silu_and_mul → W2.
   - Baseline: `BLOCK_M=128`
   - Optimized: `BLOCK_M=64` (profile-guided, shape-specific) — W2 단독 ~−25%, compute 전체 −10~12%
5. **Finalize / Combine** (`experts → combine`): top-k weighted reduce-scatter → 다음 layer로.

## Flow 2 — 최적화한 부분 (기여 + 결과)

각 최적화가 어디 살고 얼마나 기여했는지 + 최종 결과.

1. **③ Receiver fast path** (`receive`): fixed-capacity + raw-local.
   - 단독 fixed-capacity(remap 유지)는 **+2.12% / +2.30% 회귀**, raw-local까지 더해야 −6.17%/−6.47% → net receiver 기여 약 **−3.6~4.3%**. ("절반만 적용하면 안 되는" 의존성)
2. **① Invalid-route filtering** (`align`): non-local route(top-k=8에서 pair의 ~50%)가 만들던
   invalid GEMM block 제거. filtering만 **−2.65% / −3.34%** (commit `001b3cb`).
3. **② BLOCK_M=64 tile — 가장 큰 레버** (`experts`): filtering 위에 BLOCK_M=64를 더하면
   compute 이득의 대부분. `−2.65% → −12.87%`(320t), `−3.34% → −10.52%`(448t).
   runtime selector가 아니라 이 shape에서 찾은 값을 고정 적용.
4. **= 최종 결과** (`combine`): 누적 **−15.51% / −14.22%**, 12/12 paired,
   correctness `assert_close` 통과(rtol=1.6e-2, max_abs 0.00195312), alignment 477 passed.

---

## Baseline ↔ Optimized 요약

| 단계 | Baseline | Optimized |
|---|---|---|
| Receiver | dynamic recv + local→global remap | fixed-capacity + raw local id (remap 없음) |
| Assignment | generic align, invalid 포함, schedule ×2 | raw-local align, invalid skip, BLOCK_M=64 schedule 공유 |
| Expert GEMM | BLOCK_M=128 | BLOCK_M=64 (가장 큰 레버) |

## 주의 / 한계

- synthetic MoE forward, 단일 shape(tokens 320/448), EP=2, top-k=8. 실모델/실서빙/높은 EP는 미검증.
- fixed-capacity 용량은 routing oracle이 아니라 `tokens×world_size` worst-case 상한이며,
  본 워크로드는 활용률 ~99%로 유리했다 (높은 EP/낮은 top-k에선 활용률 하락 가능).
- compute 이득의 대부분은 BLOCK_M=64(tile)이고, invalid filtering은 ~2.6~3.3%로 작다.

전체 측정 근거: `vllm/benchmarks/results/`. 자세한 서사는 저장소 루트 `README.md`.
