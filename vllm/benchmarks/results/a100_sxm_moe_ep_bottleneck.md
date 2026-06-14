# A100 SXM MoE EP Bottleneck Summary

Environment:

- GPUs: 2x NVIDIA A100-SXM4-80GB, sm_80, 80 GB
- Topology: GPU0-GPU1 is `NV12`
- Driver: 580.126.16
- Torch: 2.11.0+cu128 / CUDA 12.8
- vLLM: 0.22.1rc1.dev0, editable with precompiled extensions
- Backends: `allgather_reducescatter`, `deepep_high_throughput`
- Key env: `NCCL_P2P_DISABLE=0`, `PYTORCH_NVML_BASED_CUDA_CHECK=1`
- Benchmark:
  `benchmarks/kernels/sweep_moe_ep_a40.py --preset a40_quick --world-sizes 1,2 --warmup 3 --iters 10 --output benchmarks/results/a100_sxm_quick_moe_ep.csv --fail-fast`
- Raw data: `benchmarks/results/a100_sxm_quick_moe_ep.csv`
  and `benchmarks/results/a100_sxm_qwen3_deepep_ht_moe_ep.csv`
- Section profiles:
  `benchmarks/results/a100_sxm_qwen3_agrs_sections.rank*.json`,
  `benchmarks/results/a100_sxm_qwen3_deepep_ht_sections.rank*.json`,
  `benchmarks/results/a100_sxm_qwen3_deepep_ht_sms24_sections.rank*.json`,
  `benchmarks/results/a100_sxm_qwen3_deepep_ht_sms24_receiver_sections.rank*.json`,
  `benchmarks/results/a100_sxm_qwen3_deepep_ht_sms24_triton_remap_sections.rank*.json`,
  and `benchmarks/results/a100_sxm_qwen3_agrs_inplace_combine_sections.rank*.json`
- Repro metadata: `benchmarks/results/a100_sxm_quick_moe_ep.topology.txt`
  / `benchmarks/results/a100_sxm_qwen3_deepep_ht_moe_ep.topology.txt`
  and matching `.meta.json` files

## Main Finding

NVLink is active and helps, but the current AG/RS expert-parallel path is still
latency dominated on this 2x A100 SXM node. 2GPU EP is slower than 1GPU for all
measured synthetic BF16 shapes because dispatch plus combine costs about
0.64-0.66 ms per MoE layer.

That fixed communication cost is now the primary optimization target. A100
local expert compute is fast enough that halving local experts does not repay
the AG/RS communication overhead.

## Breakdown

### AG/RS Baseline

| shape | tokens | 1GPU full | 2GPU full | speedup | top-k | comm | comm share | route CV | zero experts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B-like | 32 | 0.711 ms | 1.087 ms | 0.65x | 0.060 ms | 0.643 ms | 59.2% | 0.631 | 14 |
| Qwen3-30B-A3B-like | 128 | 0.830 ms | 1.182 ms | 0.70x | 0.059 ms | 0.640 ms | 54.1% | 0.362 | 0 |
| Qwen3-30B-A3B-like | 512 | 0.917 ms | 1.464 ms | 0.63x | 0.060 ms | 0.655 ms | 44.7% | 0.184 | 0 |
| Qwen2-MoE-57B-like | 32 | 0.610 ms | 1.033 ms | 0.59x | 0.066 ms | 0.641 ms | 62.1% | 0.723 | 8 |
| Qwen2-MoE-57B-like | 128 | 0.734 ms | 1.169 ms | 0.63x | 0.067 ms | 0.653 ms | 55.9% | 0.388 | 0 |

### DeepEP HT

Installed DeepEP V1 for A100 intranode testing:

- Package: `deep-ep==1.1.0+be8053d`
- API availability: `deep_ep.Buffer` yes, `deep_ep.ElasticBuffer` no
- vLLM detection: `has_deep_ep=True`, `has_deep_ep_v2=False`
- Build mode: SM80, `DISABLE_SM90_FEATURES=1`, `DISABLE_NVSHMEM=1`
- Scope: high-throughput intranode/NVLink only; low-latency/RDMA is disabled

Standalone DeepEP intranode smoke test passed on 2x A100 with
`num_tokens=128`, `hidden=2048`, `top_k=8`, `num_experts=128`. Best measured
DeepEP kernel timings in that smoke test were dispatch `45.87 us` and combine
`38.83 us`.

vLLM FusedMoE integration with `deepep_high_throughput` did not improve the
synthetic Qwen3-like BF16 full-forward result:

| shape | tokens | 1GPU full | 2GPU DeepEP HT full | speedup | AG/RS 2GPU full | route CV |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B-like | 32 | 0.711 ms | 1.300 ms | 0.55x | 1.087 ms | 0.631 |
| Qwen3-30B-A3B-like | 128 | 0.828 ms | 1.417 ms | 0.58x | 1.182 ms | 0.362 |
| Qwen3-30B-A3B-like | 512 | 0.916 ms | 1.683 ms | 0.54x | 1.464 ms | 0.184 |

The benchmark reports `dispatch_us`/`combine_us` as null for DeepEP HT because
vLLM drives DeepEP through the FusedMoE prepare/finalize path; the raw
`get_ep_group().dispatch/combine()` methods are not implemented for that
manager.

Synchronized section profiling for Qwen3-like tokens=128 shows that the
DeepEP regression is not in the local expert kernel:

| backend | setting | prepare | experts | finalize | 2GPU full |
|---|---|---:|---:|---:|---:|
| AG/RS | default | 0.147 ms | 0.707 ms | 0.120 ms | 1.195 ms |
| DeepEP HT | 20 SMs | 0.323 ms | 0.708 ms | 0.165 ms | 1.431 ms |
| DeepEP HT | 24 SMs | 0.303 ms | 0.701 ms | 0.160 ms | 1.404 ms |

The DeepEP prepare path at 24 SMs breaks down to dispatch submit `0.136 ms`
and dispatch receiver/wait `0.127 ms`; combine submit is `0.099 ms` and the
combine receiver/output copy is `0.027 ms`. The `24` SM setting is now exposed
through `VLLM_DEEPEP_HT_NUM_SMS` and is a small win on this A100 node, but it
does not close the gap to AG/RS.

Deeper receiver profiling shows that the extra DeepEP HT prepare cost is mostly
metadata and top-k id adaptation for vLLM's existing expert-kernel interface:

| path | prepare | dispatch submit | dispatch receiver | receiver wait | top-k remap | metadata | post quant |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepEP HT 24SM, `torch.where` remap | 0.385 ms | 0.141 ms | 0.207 ms | 0.009 ms | 0.089 ms | 0.043 ms | 0.009 ms |
| DeepEP HT 24SM, Triton remap | 0.364 ms | 0.141 ms | 0.187 ms | 0.009 ms | 0.049 ms | 0.065 ms | 0.009 ms |

The Triton remap result uses
`VLLM_DEEPEP_HT_TRITON_TOPK_REMAP=1`, which replaces
`torch.where(expert_topk_ids == -1, invalid, expert_topk_ids + offset)` with a
feature-gated in-place Triton kernel. On this Qwen3-like tokens=128 shape, the
full DeepEP HT 24SM forward improved from `1.414 ms` to `1.386 ms` in a matched
warmup/iteration run. The direct receiver remap section dropped by about
`40 us`, while the end-to-end gain was about `27 us`.

The AG/RS fallback finalize path was also tested with an in-place
`reduce_scatterv(out=output)` path so the modular MoE finalize no longer
allocates a reduce-scatter result and then copies it into the caller-provided
output buffer. The full forward result stayed within run-to-run noise
(`1.178 ms` before, `1.179 ms` after), but synchronized section profiling shows
the intended local effect:

| path | prepare | experts | finalize | AG/RS finalize combine |
|---|---:|---:|---:|---:|
| AG/RS baseline | 0.147 ms | 0.707 ms | 0.120 ms | 0.101 ms |
| AG/RS in-place combine | 0.145 ms | 0.702 ms | 0.101 ms | 0.082 ms |

Correctness smoke: a 2GPU `reduce_scatterv(..., out=...)` check matched the
existing out-of-place result and verified that the returned tensor aliases the
provided output buffer.

## Interpretation

- NVLink is working: `nvidia-smi topo -m` reports `NV12`, `nvidia-smi nvlink -s`
  reports 12 active 25 GB/s links per GPU, and P2P read/write/native atomics are
  `OK`.
- The AG/RS communication floor is roughly constant at 0.64-0.66 ms across
  these token counts and shapes.
- Disabling NCCL P2P for Qwen3-like 512-token 2GPU increased dispatch+combine
  from 0.655 ms to 0.919 ms, so NVLink/P2P is reducing communication time.
- Top-k is stable at about 0.06-0.07 ms and is not the first bottleneck.
- Routing imbalance is visible for small token counts, but it is not enough to
  explain the regression. The communication floor dominates before expert
  compute tuning can pay off.
- FlashInfer NVLink one-sided is installed, but it does not support this
  unquantized BF16 synthetic configuration.
- FlashInfer NVLink two-sided was also tested on the same BF16 unquantized
  shape and failed with `No Unquantized MoE backend supports the deployment
  configuration`.
- A100 does not report FP8 support in vLLM (`current_platform.supports_fp8()`
  is `False`), so FP8/NVFP4 MoE communication experiments are not conclusive on
  this node. Those should be run on Hopper/Blackwell.
- Current DeepEP V2 from the vLLM install script is Hopper-oriented. On A100
  SM80 it fails to build because the SM80 branch is not implemented in the
  pinned setup path and the V2 code requires SM90/NCCL Gin features. The
  installed A100 workaround is DeepEP V1 intranode HT only.
- DeepEP V1 standalone communication is much faster than the AG/RS measured
  communication floor, but the current vLLM BF16 full-forward integration is
  slower than AG/RS on these small synthetic shapes. The next question is where
  prepare/finalize, padding/layout, synchronization, or expert execution is
  erasing the raw communication win.
- The AG/RS equal-size path already falls back from `all_gatherv` to regular
  `all_gather`, and grouped hidden/top-k gathers are already wrapped in an NCCL
  group. The remaining AG/RS improvements are structural rather than a simple
  backend selection fix.

## Optimization Direction

Prioritize communication path work before local GEMM tuning:

1. Keep `VLLM_DEEPEP_HT_NUM_SMS=24` as the tested best DeepEP HT setting so
   far on this A100 node, but do not expect it to beat AG/RS by itself.
2. Keep `VLLM_DEEPEP_HT_TRITON_TOPK_REMAP=0` by default until it is tested on
   more shapes. The flag is useful as a concrete receiver optimization
   experiment, and it reduced the current DeepEP HT full-forward latency by
   about `27 us`.
3. Continue DeepEP receiver work at the interface boundary: avoid global-id
   remap entirely by letting the expert kernel consume local expert ids, and
   eliminate or cache `ExpertTokensMetadata.make_from_list` when the expert
   backend can use GPU-side counts directly.
   - A feature-gated prototype,
     `VLLM_DEEPEP_HT_LOCAL_EXPERT_IDS=1`, keeps DeepEP HT received top-k ids in
     local expert-id space and maps invalid entries through a local sentinel
     expert. On the current A100 Qwen3-like BF16 shape it did not improve
     latency: `1.355 ms` global-id path vs `1.383 ms` local-id path in the
     matched fixed-profile run. This is a useful negative result: the remaining
     cost is not just global/local id conversion, because DeepEP HT `recv_x` is
     still a rank-local token list with local ids embedded in `recv_topk_idx`,
     so the generic `moe_align_block_size` token/expert assignment work remains.
4. For AG/RS fallback, the in-place combine path removes a measurable
   `~19 us` finalize sub-step but does not change the overall conclusion. The
   remaining AG/RS improvements must attack collective latency itself:
   persistent gather buffers, fewer metadata payloads, fused combine/reduce, or
   overlap between dispatch, local expert GEMM, and combine.
5. Treat expert-kernel tuning as second priority for these shapes. The measured
   2GPU non-communication savings are only about 0.10-0.28 ms, while AG/RS
   communication costs about 0.64 ms.
6. If continuing kernel-level work on A100, the next real target is a DeepEP HT
   specific expert-assignment kernel that consumes local `recv_topk_idx`,
   `topk_weights`, and per-expert counts directly instead of routing through
   the generic `moe_align_block_size` contract.

## Prefill vs Decode Profile

Nsight Compute/Systems are not installed in the current PATH (`ncu`/`nsys` were
not found), so a lightweight `torch.profiler` CUDA summary was added to
`benchmarks/kernels/benchmark_moe_ep_a40.py` via:

```bash
--phase-name decode|prefill
--torch-profile-iters N
--torch-profile-output <path>
```

Synthetic phase split:

- decode-like: `tokens=16`
- prefill-like: `tokens=512`

AG/RS:

| phase | full | prepare | experts | finalize | raw dispatch | raw combine |
|---|---:|---:|---:|---:|---:|---:|
| decode-like | 1.005 ms | 0.126 ms | 0.516-0.536 ms | 0.086-0.107 ms | 0.403 ms | 0.336 ms |
| prefill-like | 1.493 ms | 0.166-0.172 ms | 0.936-0.959 ms | 0.121-0.145 ms | 0.407 ms | 0.348 ms |

DeepEP HT 24SM with Triton top-k remap:

| phase | full | prepare | experts | finalize |
|---|---:|---:|---:|---:|
| decode-like | 1.238 ms | 0.364-0.374 ms | 0.521-0.545 ms | 0.140-0.162 ms |
| prefill-like | 1.682 ms | 0.403-0.409 ms | 0.934-0.947 ms | 0.202-0.216 ms |

Kernel-level interpretation:

- Decode-like routing is extremely sparse: mean `1.0` token per expert and
  `46/128` experts receive zero tokens. The bottleneck is collective/launch
  floor plus ragged tiny expert batches, not a large GEMM microkernel alone.
- Prefill-like routing is dense: mean `32.0` tokens per expert and zero empty
  experts. The two `fused_moe_kernel` launches dominate the compute side
  (`~0.61 ms/fwd` in the torch profiler), with smaller but visible
  `moe_sum` (`~0.03 ms/fwd`) and `silu_and_mul` (`~0.026 ms/fwd`) boundaries.
- FlashMoE's direct vLLM adapter is not a drop-in for this Qwen3-like shape
  because it currently supports only top-1/top-2 BF16 EP, while the target
  shape is top-k=8. The useful idea to carry over is not the adapter itself but
  the architecture: a persistent/tile scheduler that fuses dispatch, expert
  compute, and combine, or at least fuses GEMM2 epilogue with top-k weighted
  reduction for the prefill path.

Recommended next kernel project:

1. Use Nsight Systems to confirm the prefill/decode timeline and stream
   dependencies once `nsys` is available.
2. Use Nsight Compute on `fused_moe_kernel` for prefill to measure tensor-core
   utilization, memory throughput, occupancy, and stall reasons.
3. Prototype a prefill-first fusion:
   - GEMM2 epilogue applies top-k weight and writes directly to reduced output,
     replacing the separate `moe_sum` boundary.
   - Then evaluate a deeper DeepEP HT assignment-to-GEMM scheduler path.
4. Treat decode separately: overlap communication/compute and reduce launch
   count before investing in a larger GEMM kernel rewrite.

## H100/H200 Validation Plan

This A100 node is useful for SM80/NVLink diagnosis, but it cannot validate the
main DeepEP V2 and FP8/NVFP4 serving target. The Hopper validation run should
answer three questions:

1. Does current DeepEP V2 build and load cleanly?
   - Install the vLLM-pinned DeepEP V2 commit with Hopper CUDA/NCCL versions.
   - Verify `has_deep_ep_v2=True`, `deep_ep.ElasticBuffer` exists, and NCCL Gin
     requirements are met.
   - Record `nvidia-smi topo -m`, `nvidia-smi nvlink -s`, driver, CUDA, NCCL,
     torch, and vLLM commit.

2. Does DeepEP V2 beat AG/RS on the same BF16 synthetic shapes?
   - Re-run the Qwen3-like sweep for tokens `32,128,512`.
   - Compare `allgather_reducescatter`, `deepep_high_throughput`, and
     `deepep_v2`.
   - Repeat section profiling so prepare/finalize can be compared directly with
     the A100 table.

3. Do quantized MoE paths change the conclusion?
   - Run FP8/NVFP4 candidates on Hopper/H200 where vLLM reports FP8 support.
   - Include FlashInfer NVLink one-sided/two-sided where supported.
   - Measure full forward, prepare/finalize, expert time, and output parity
     against BF16 for a fixed seed.

Success criteria for the Hopper phase: DeepEP V2 or a quantized all-to-all path
must show an end-to-end MoE layer win, not only a raw dispatch/combine win. If
DeepEP still loses, the profile must identify whether the remaining gap is
global/local expert id conversion, metadata transfer, stream synchronization,
padding/layout conversion, or expert-kernel incompatibility.
