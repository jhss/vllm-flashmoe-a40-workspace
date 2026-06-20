# Workspace Operating Plan

This file is the handoff plan for future Codex sessions working in this
workspace.

## Objective

Improve vLLM multi-GPU MoE and expert-parallel performance. FlashMoE is a
comparison point and possible source of implementation ideas; it does not need
to be ported as-is if profiling shows another path is better.

## Ground Rules

- Treat the current 2x A40 server as the low-cost development box for builds,
  correctness checks, instrumentation, synthetic sweeps, and PCIe bottleneck
  characterization.
- Do not overfit changes to the A40 PCIe topology if they are likely to hurt
  the target A100 SXM/NVLink setup.
- Keep every experimental optimization feature-gated or benchmark-only until
  it has correctness and performance evidence.
- Never commit credentials, model weights, Hugging Face caches, large profiler
  traces, or unsanitized conversation logs.
- Record hardware topology, CUDA/driver/PyTorch/vLLM versions, benchmark
  commands, and relevant environment variables with each performance result.

## EP Bottleneck Map

Use this map when profiling vLLM MoE expert parallelism:

1. Routing and load balance: top-k routing skew, expert imbalance, and tiny
   per-expert token batches.
2. Prepare path: top-k metadata handling, token sorting, permute/gather,
   dtype/layout packing, scratch allocation, and hidden host synchronizations.
3. Dispatch communication: all-to-all payload over PCIe or NVLink, NCCL stream
   waits, rank imbalance, and backend selection.
4. Expert compute: grouped GEMM occupancy, ragged expert batches, quantized
   layout compatibility, kernel launch overhead, and activation fusion.
5. Combine path: return all-to-all, unpermute/scatter, top-k weighted reduce,
   extra copies, and final layout conversion.
6. Overlap: DBO microbatching, communication/compute overlap, CUDA stream
   dependencies, and CUDA graph capture breaks.
7. End-to-end effects: scheduler stalls, allocator churn, warmup behavior,
   model-runner overhead, and serving throughput vs kernel-only wins.

## Current Assets

- `vllm/benchmarks/kernels/benchmark_moe_ep_a40.py`
- `vllm/benchmarks/kernels/sweep_moe_ep_a40.py`
- `vllm/benchmarks/kernels/benchmark_flashmoe.py`
- `vllm/benchmarks/results/a40_moe_ep_bottleneck.md`
- `vllm/benchmarks/results/a40_quick_moe_ep.csv`
- `vllm/vllm/model_executor/layers/fused_moe/experts/flashmoe.py`
- `FlashMoE/` standalone CUDA 13/A40 compatibility branch

## Execution Plan

### Phase 1: A40 Baseline

- Capture `nvidia-smi topo -m`, `nvidia-smi -q`, NCCL topology/debug output,
  CUDA driver/toolkit versions, PyTorch version, and vLLM commit.
- Run synthetic MoE EP sweeps across token count, hidden size, intermediate
  size, expert count, top-k, dtype, and all-to-all backend.
- Run one small end-to-end vLLM serving benchmark for a reproducible MoE model
  if the model fits comfortably on 2x A40.
- Use Nsight Systems first to locate serialization and communication waits;
  use Nsight Compute only for kernels that are proven hotspots.

### Phase 2: Bottleneck Attribution

- Split timing into gate/top-k, prepare/permute, all-to-all dispatch, expert
  GEMM, all-to-all combine, unpermute/reduce, and post-MoE overhead.
- Log tokens-per-expert histograms and per-rank imbalance for each sweep.
- Compare DBO on/off, CUDA graph behavior, NCCL backend choices, and stream
  overlap.
- Decide whether the dominant A40 bottleneck is communication, ragged GEMM,
  packing/copies, synchronization, or end-to-end scheduler overhead.

### Phase 3: Low-Risk Improvements

- Add missing timing counters or NVTX ranges before changing algorithms.
- Reuse or preallocate scratch buffers where allocator churn is visible.
- Remove avoidable synchronizations and CPU waits on the MoE path.
- Reduce copies in permute/unpermute/reduce if profiling shows material cost.
- Improve grouped GEMM behavior for small or ragged expert batches.
- Keep FlashMoE-style experiments behind an explicit backend selector or env
  flag until they win on correctness and measured performance.

### Phase 4: FlashMoE Evaluation

- Keep FlashMoE as a standalone benchmark first.
- Test vLLM-relevant shapes, not only paper/demo shapes.
- Validate numerical parity against the existing vLLM fused MoE path.
- Port only the pieces that solve a measured vLLM bottleneck: scheduling,
  topology-aware dispatch, fused combine, or expert compute.

### Phase 5: A100 SXM Migration

- Rent A100 SXM only after the A40 benchmark scripts are reproducible and short.
- Verify NVLink with `nvidia-smi topo -m`; A100 PCIe rentals may not include the
  same NVLink topology as A100 SXM nodes.
- Re-run the exact A40 sweeps and compare communication/compute ratios.
- Reprioritize after seeing whether the bottleneck moves from all-to-all to
  expert GEMM, packing, or scheduler overhead.
- Only then do NVLink-specific overlap or topology tuning.

## Immediate Experiment Queue

The current high-priority experiment is the DeepEP HT direct expert-assignment
path, controlled by these opt-in flags:

- `VLLM_DEEPEP_HT_DIRECT_ASSIGNMENT=1`: preserve DeepEP HT raw local expert IDs
  and build the Triton MoE assignment schedule from DeepEP's per-local-expert
  token counts instead of rerunning the generic global-ID alignment path.
- `VLLM_DEEPEP_HT_DIRECT_ASSIGNMENT_DEBUG=1`: enable extra validation for the
  direct assignment path. Use this only for correctness/debug runs, not timing.
- `VLLM_DEEPEP_HT_LOCAL_EXPERT_IDS=1`: older local-ID remap path. Keep it as an
  ablation against baseline and direct assignment.

Run the following experiments in order:

1. Correctness smoke tests:
   - Run `vllm/tests/kernels/moe/test_deepep_ht_expert_assignment.py` on CUDA.
   - Cover raw `-1` local IDs, out-of-range IDs with `expert_map`, empty
     experts, skewed experts, and `BLOCK_SIZE_M` values used by W1 and W2.
   - Compare direct assignment output against the generic assignment path for
     balanced and skewed top-k distributions.

2. Assignment-only microbenchmarks:
   - Compare generic `moe_align_block_size` scheduling against direct DeepEP HT
     scheduling.
   - Sweep tokens `128,256,512,1024,2048,4096`, top-k `2,4,8`, local experts
     `8,16,32`, and `BLOCK_SIZE_M` `16,32,64,128`.
   - Include balanced routing, Zipf/skewed routing, empty experts, and high
     invalid-slot ratios.
   - Record assignment time, padded token count, padding overhead, invalid-slot
     ratio, tokens-per-expert histogram, and whether W1/W2 reused a cached
     schedule.

3. Full MoE kernel ablation on 2x A40:
   - Use `vllm/benchmarks/kernels/benchmark_moe_ep_a40.py` for targeted shapes.
   - Compare baseline, `VLLM_DEEPEP_HT_LOCAL_EXPERT_IDS=1`, and
     `VLLM_DEEPEP_HT_DIRECT_ASSIGNMENT=1`.
   - Keep debug validation off for timing.
   - Measure prepare/permute, dispatch communication, expert GEMM, combine,
     unpermute/reduce, and total forward time.
   - Run both world size 1 and 2 when possible so local scheduling wins are not
     confused with all-to-all effects.

4. A40 sweep:
   - Use `vllm/benchmarks/kernels/sweep_moe_ep_a40.py` once the targeted
     ablation identifies promising shapes.
   - Store compact CSV or Markdown summaries under `vllm/benchmarks/results/`.
   - Do not commit raw profiler traces or large intermediate artifacts.

5. Nsight Systems profile:
   - Profile only the smallest set of shapes where the direct path changes
     timing by more than noise.
   - Check whether direct assignment removes scheduling work, exposes a new
     synchronization, changes stream overlap, or simply moves the bottleneck to
     dispatch/combine communication.
   - Use Nsight Compute only for kernels proven hot by Nsight Systems.

6. End-to-end serving sanity:
   - Run one short reproducible serving benchmark with a MoE model that fits on
     2x A40.
   - Compare baseline vs direct assignment with the same prompts, batching,
     DBO/CUDA graph settings, and all-to-all backend.
   - Treat kernel-only wins as insufficient unless throughput or latency also
     improves beyond run-to-run noise.

7. A100 SXM verification:
   - After the A40 scripts are short and reproducible, rerun the same baseline,
     local-ID, and direct-assignment comparisons on A100 SXM/NVLink.
   - Keep the feature gated unless it improves A100 end-to-end behavior or has a
     clearly documented negative result.

Decision rules:

- Keep the direct assignment path only if correctness passes and full forward or
  serving improves beyond noise on more than one relevant shape.
- If the win appears only in assignment microbenchmarks, document the negative
  end-to-end result and stop expanding the optimization.
- If the win is A40 PCIe-specific and disappears on A100 SXM, keep the path
  benchmark-only or disable it by default.
- Record commit hash, hardware topology, CUDA/driver/PyTorch/vLLM versions,
  exact commands, environment variables, warmup/iteration counts, and median
  timing for every reported result.

## Success Criteria

- Reproducible baseline tables for A40 and A100 SXM.
- A clear EP bottleneck summary tied to traces, timings, and topology.
- At least one feature-gated optimization with correctness coverage, or a
  documented negative result explaining why it does not help.
- End-to-end throughput or latency improvement, not only a kernel microbenchmark
  win.
- Clean branches with no credentials, large binaries, caches, or unsanitized
  traces committed.

## Cost Control

- Use the A40 server for iteration, correctness, CI-like checks, and profiler
  script development.
- Keep A100 SXM runs short: topology check, baseline sweep, targeted profile,
  optimization verification.
- Prefer synthetic sweeps and short serving tests over long workloads while the
  bottleneck is still being identified.
