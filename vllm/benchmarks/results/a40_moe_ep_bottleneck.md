# A40 MoE EP Bottleneck Summary

Environment:

- GPUs: 2x NVIDIA A40, sm_86, 46 GB
- Topology: GPU0-GPU1 is `PXB`, no NVLink
- Backend: `allgather_reducescatter`
- Benchmark: `benchmarks/kernels/sweep_moe_ep_a40.py --preset a40_quick --world-sizes 1,2 --warmup 3 --iters 10`
- Raw data: `benchmarks/results/a40_quick_moe_ep.csv`
- Repro metadata: `benchmarks/results/a40_quick_moe_ep.topology.txt`
  and `benchmarks/results/a40_quick_moe_ep.meta.json`

## Main Finding

On this A40 PCIe/PXB server, expert parallelism is dominated by communication.
The EP path is:

1. top-k routing
2. all-gather dispatch of activations and routing metadata
3. local expert computation on fewer local experts
4. reduce-scatter combine

For the measured shapes, dispatch plus combine takes 36-57% of the 2-GPU
forward time. This erases most or all of the benefit from halving the local
expert count.

## Breakdown

| shape | tokens | 1GPU full | 2GPU full | speedup | top-k | dispatch | combine | comm share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B-like | 32 | 1.930 ms | 1.867 ms | 1.03x | 0.066 ms | 0.386 ms | 0.320 ms | 37.8% |
| Qwen3-30B-A3B-like | 128 | 2.208 ms | 2.349 ms | 0.94x | 0.283 ms | 0.549 ms | 0.400 ms | 40.4% |
| Qwen3-30B-A3B-like | 512 | 2.429 ms | 3.563 ms | 0.68x | 0.076 ms | 1.052 ms | 0.962 ms | 56.5% |
| Qwen2-MoE-57B-like | 32 | 1.625 ms | 1.700 ms | 0.96x | 0.083 ms | 0.560 ms | 0.376 ms | 55.0% |
| Qwen2-MoE-57B-like | 128 | 2.024 ms | 2.222 ms | 0.91x | 0.075 ms | 0.435 ms | 0.362 ms | 35.9% |

## Interpretation

- Top-k routing is not the primary bottleneck. It is usually under 0.12 ms;
  the Qwen3 128-token value is a measurement outlier or transient.
- Dispatch cost grows with token count and hidden size because it gathers
  token activations across ranks.
- Combine cost is also large because the AG/RS backend reduces and scatters
  full hidden outputs across PCIe.
- The expert compute reduction from EP is visible, but not large enough to
  overcome PCIe/PXB communication.
- Qwen3 512 tokens is the clearest failure case: 2GPU EP is 0.68x of 1GPU,
  with communication alone costing 2.013 ms.

## Current Optimization Direction

Keep doing these on A40:

- Local expert-kernel tuning by shape and device.
- Benchmark harness cleanup and shape sweep expansion.
- Router/top-k and local expert profiling.
- FlashMoE compile/porting smoke tests.

Do not draw final EP scaling conclusions from this machine:

- The topology is `PXB`, not NVLink.
- AG/RS communication is the dominant cost.
- FlashMoE/NVSHMEM-style approaches are designed for much stronger P2P fabrics.

## A100 SXM Validation Plan

Run the same benchmark on A100 SXM/HGX:

```bash
cd /workspace/vllm
python benchmarks/kernels/sweep_moe_ep_a40.py \
  --preset a40_quick \
  --world-sizes 1,2 \
  --warmup 3 \
  --iters 10 \
  --output benchmarks/results/a100_sxm_quick_moe_ep.csv
```

Before interpreting results, confirm:

```bash
nvidia-smi topo -m
```

Expected for useful EP testing: GPU links should show `NV#` or NVSwitch-style
connectivity, not `PXB`, `PHB`, or `SYS`.

Compare these fields against A40:

- `dispatch_us`
- `combine_us`
- `comm_ms`
- `comm_share`
- `speedup_vs_world1`
- `expert_tokens_cv`
- `expert_tokens_zero`

If A100 SXM reduces `comm_ms` substantially and 2GPU speedup becomes positive,
then the next useful work is overlap or fused communication/expert execution.
If communication remains high even on NVLink, then vLLM's AG/RS EP backend is
the first optimization target before FlashMoE-style kernel integration.
