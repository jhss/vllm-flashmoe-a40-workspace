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

The current high-priority experiment is DeepEP HT generic
`ignore-invalid` follow-up work. The A100 SXM paired runs in
`vllm/benchmarks/results/a100_sxm_moe_ep_code_changes.md` show that forced
generic `ignore-invalid` already wins at the smallest measured received-M
ranges. The 3-seed balanced pilot found median wins from received-M 16 through
254/255, so the defensible conclusion is that the break-even is below that
range for this workload and the current 1024 threshold is too conservative. Do
not change the global default to 0 without more backend and shape coverage.

Relevant flags:

- `VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS=1`: opt into the generic EP
  invalid-expert filtering path.
- `VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS_MIN_TOKENS=0`: force the path on
  for DeepEP HT benchmark confirmation runs only.
- `VLLM_DEEPEP_HT_LOCAL_EXPERT_IDS=1` and
  `VLLM_DEEPEP_HT_DIRECT_ASSIGNMENT=1`: keep these as ablations. They are not
  the current default candidates.

Run the following experiments in order:

1. Small-M break-even confirmation:
   - Use DeepEP HT, top-k 8, the A100 BF16 synthetic shape from the report, and
     input tokens `8,16,32,48,64,96,128`.
   - Completed a 3-seed pilot with seeds `1007,2007,3007` and 4 balanced
     cycles. All measured tokens had negative paired median deltas for forced
     ON; received-M 16 was still faster.
   - If more confidence is needed, extend to 5 seeds and keep the same 4
     balanced cycles:
     `OFF->ON`, `ON->OFF`, `ON->OFF`, `OFF->ON`.
   - Runner:
     `python benchmarks/results/run_deepep_ht_paired_matrix.py --mode break-even`.
   - Record received tokens per rank, valid/invalid route pairs, activation
     state per rank, critical-path latency, paired deltas, seed-level medians,
     and positive transient outliers.

2. Post-ignore `BLOCK_SIZE_M` matrix:
   - Completed the 54-run screening after making assignment creation lazy.
     Tokens `320,448`, settings `default,both_64,both_32`, seeds
     `1007,2007,3007`, and 3 Latin-square cycles all showed strong wins for
     smaller `BLOCK_M`.
   - `both_64` was best in the screening: median paired deltas were about
     `-8.9%` at both tokens with `9/9` wins. `both_32` also won, but was
     slower than `both_64`, so padding alone is not sufficient.
   - Next split W1/W2 contribution before running a broad sweep:
     `tokens=320,448`, settings `default,w1_64,w2_64,both_64`, seeds
     `1007,2007,3007`, cycles `4`.
   - Runner examples:
     `python benchmarks/results/run_deepep_ht_paired_matrix.py --mode block-m-screening`
     and
     `python benchmarks/results/run_deepep_ht_paired_matrix.py --mode block-m-sweep --tokens 320 448 --block-m-settings default w1_64 w2_64 both_64 --input-seed-bases 1007 2007 3007`.
   - Record W1 latency, activation latency, W2 latency, reduce latency, full
     critical path, W1/W2 padded rows, W1/W2 block size, valid blocks, and
     padding ratio.

3. Routing-aware config selector:
   - If smaller `BLOCK_SIZE_M` wins in the matrix, prototype selection based on
     post-filter expert histograms:
     `sum(ceil_div(expert_count[e], block_m) * block_m)`.
   - Use candidates `32,64,128`, then apply a small GEMM-efficiency LUT instead
     of choosing only by padded-row count.
   - Keep the selector DeepEP HT/workload gated until skewed routing and other
     EP backend measurements are available.

4. Better ignore activation policy:
   - After the small-M run, compare a simple DeepEP HT threshold against a
     saved-work estimate based on invalid route pairs and the number of GEMM
     programs removed.
   - Prefer a policy that distinguishes low-M/high-invalid workloads from
     high-M/low-invalid workloads.

5. Skew and correctness coverage:
   - Test uniform, moderately skewed, hot-expert, empty-expert, and real router
     trace distributions.
   - Add a 2-rank full-output correctness comparison for OFF vs ON before any
     default change.

Decision rules:

- Keep `ignore-invalid` enabled by env override for DeepEP HT experiments until
  small-M, skewed-routing, and correctness confirmation are complete.
- Promote a DeepEP HT-specific default only if full forward or serving improves
  beyond noise across relevant shapes and routing distributions.
- Consider a generic EP default only after other all-to-all backends and model
  shapes have measured coverage.
- Record commit hash, hardware topology, CUDA/driver/PyTorch/vLLM versions,
  exact commands, environment variables, warmup/iteration counts, paired
  medians, and win counts for every reported result.

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
