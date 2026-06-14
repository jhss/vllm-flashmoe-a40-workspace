# Sanitized Conversation Log: A40 MoE EP / FlashMoE Work

This file summarizes the working conversation and actions from the A40
multi-GPU MoE profiling session. Sensitive information, including GitHub
tokens, has been intentionally redacted and must not be reconstructed here.

## Goal

Profile vLLM multi-GPU MoE execution, identify EP bottlenecks, make low-cost
progress on the local A40 2-GPU machine, and prepare the same workflow for a
future A100 SXM/NVLink validation run. FlashMoE was treated as a possible
reference/backend, but not as a hard dependency.

## Hardware Context

- Local machine: 2x NVIDIA A40, 46 GB, sm_86
- `nvidia-smi topo -m`: GPU0-GPU1 is `PXB`
- No NVLink on the local A40 machine
- Conclusion: this machine is useful for development, local kernel profiling,
  and benchmark harness work, but not for final EP scaling conclusions.

## Installation And Environment

- vLLM was installed editable from `/workspace/vllm`.
- Final vLLM environment used Torch `2.11.0+cu130` / CUDA 13.
- FlashMoE was installed editable from `/workspace/FlashMoE`.
- `pip check` reported no broken requirements after dependency cleanup.
- FlashMoE CUDA 13 JIT compile still fails due CUDA 13 CCCL/libcu++ API
  incompatibilities, including missing `cuda::fast_mod_div`,
  `cuda::round_up`, and `cuda::ptx::*` APIs.

## FlashMoE Integration Work

Implemented an experimental vLLM `flashmoe` backend path:

- Added `flashmoe` to `MoEBackend`.
- Added an experimental FlashMoE adapter under
  `vllm/model_executor/layers/fused_moe/experts/flashmoe.py`.
- Added runner/oracle/routed-experts hooks for FlashMoE execution.
- Added `benchmarks/kernels/benchmark_flashmoe.py`.

FlashMoE was not validated end-to-end on 2GPU A40 because:

- Local A40 topology is PCIe/PXB with no NVLink.
- FlashMoE/NVSHMEM multi-GPU path is sensitive to P2P/NVSHMEM topology.
- CUDA 13 porting work remains.

## vLLM MoE EP Profiling Work

Added reusable benchmark tooling:

- `benchmarks/kernels/benchmark_moe_ep_a40.py`
- `benchmarks/kernels/sweep_moe_ep_a40.py`

The benchmark separates:

- full FusedMoE forward latency
- top-k routing latency
- all-gather dispatch latency
- reduce-scatter combine latency

The sweep saves raw data and topology:

- `benchmarks/results/a40_quick_moe_ep.csv`
- `benchmarks/results/a40_quick_moe_ep.topology.txt`
- `benchmarks/results/a40_moe_ep_bottleneck.md`

## A40 MoE Tuning

Added A40-specific MoE Triton configs for Qwen3-like shapes:

- `vllm/model_executor/layers/fused_moe/configs/E=128,N=768,device_name=NVIDIA_A40.json`
- `vllm/model_executor/layers/fused_moe/configs/E=64,N=768,device_name=NVIDIA_A40.json`

Manual tuning showed local expert kernel improvement potential around 2-7%
depending on token count and local expert count.

## EP Bottleneck Finding

The primary A40 EP bottleneck is communication, not top-k routing.

For the measured shapes, `dispatch + combine` took 36-57% of 2GPU forward
time. The clearest case:

- Qwen3-like, 512 tokens
- 1GPU: 2.429 ms
- 2GPU EP: 3.563 ms
- speedup: 0.68x
- dispatch + combine: 2.013 ms
- communication share: 56.5%

Interpretation:

- The local expert compute reduction from EP is visible.
- The PCIe/PXB communication cost erases the benefit on A40.
- A40 is therefore the wrong machine for final EP scaling conclusions.

## A100 SXM Plan

Use the same sweep on A100 SXM/HGX:

```bash
cd /workspace/vllm
python benchmarks/kernels/sweep_moe_ep_a40.py \
  --preset a40_quick \
  --world-sizes 1,2 \
  --warmup 3 \
  --iters 10 \
  --output benchmarks/results/a100_sxm_quick_moe_ep.csv
```

Before interpreting results:

```bash
nvidia-smi topo -m
```

Expected useful topology:

- GPU links should show `NV#` or NVSwitch-style connectivity.
- `PXB`, `PHB`, or `SYS` means the result is not representative of NVLink EP.

Compare against A40:

- `dispatch_us`
- `combine_us`
- `comm_ms`
- `speedup_vs_world1`

If A100 SXM reduces `comm_ms` enough for 2GPU speedup to become positive, the
next target is overlap/fused communication and expert execution. If not, vLLM's
AG/RS EP backend is the first optimization target before deeper FlashMoE-style
integration.

## GitHub Push

Forks were created under the authenticated GitHub account:

- `jhss/vllm`
- `jhss/FlashMoE`

Branches pushed:

- vLLM: `codex/flashmoe-a40-moe-profiling`
- FlashMoE: `codex/cuda13-a40-compat`

Sensitive token handling:

- A GitHub token was provided during the conversation.
- The token is not included in this file.
- Temporary local token/askpass files were removed after push.
- The token should be revoked/rotated because it was exposed in the chat.
