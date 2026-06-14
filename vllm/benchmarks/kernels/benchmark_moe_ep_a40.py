#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small MoE EP profiler for PCIe-only 2-GPU systems.

This benchmark intentionally uses the real FusedMoE layer path instead of
calling only the raw Triton expert kernel. It is meant to separate:

* local routing/expert compute cost,
* allgather dispatch cost, and
* reduce-scatter combine cost

for the allgather_reducescatter EP backend.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Callable

import torch
from torch.multiprocessing import spawn

# On the local A40 PXB machine, NCCL P2P can hang. Let callers override this,
# but default to the stable path for the profiler.
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

from vllm.config import (  # noqa: E402
    CompilationConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import (  # noqa: E402
    cleanup_dist_env_and_memory,
    get_ep_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import get_forward_context, set_forward_context  # noqa: E402
from vllm.model_executor.layers.fused_moe import FusedMoE  # noqa: E402
from vllm.utils.math_utils import next_power_of_2  # noqa: E402
from vllm.utils.network_utils import get_distributed_init_method  # noqa: E402
from vllm.utils.torch_utils import set_random_seed  # noqa: E402
from vllm.v1.worker.workspace import (  # noqa: E402
    init_workspace_manager,
    is_workspace_manager_initialized,
)


@dataclass
class BenchResult:
    rank: int
    world_size: int
    backend: str
    tokens: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    local_experts: int
    top_k: int
    dtype: str
    warmup: int
    iters: int
    full_forward_us: float
    topk_us: float | None
    dispatch_us: float | None
    combine_us: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=1, choices=[1, 2])
    parser.add_argument("--backend", default="allgather_reducescatter")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=768)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--csv", action="store_true")
    return parser.parse_args()


def make_vllm_config(args: argparse.Namespace) -> VllmConfig:
    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        data_parallel_size=args.world_size,
        enable_expert_parallel=args.world_size > 1,
        all2all_backend=args.backend,
    )
    compilation_config = CompilationConfig()
    compilation_config.pass_config.fuse_allreduce_rms = False
    max_tokens = max(1, next_power_of_2(args.tokens))
    return VllmConfig(
        parallel_config=parallel_config,
        compilation_config=compilation_config,
        scheduler_config=SchedulerConfig.default_factory(
            max_num_batched_tokens=max_tokens,
            max_num_seqs=min(max_tokens, max(1, args.tokens)),
        ),
    )


def init_rank(vllm_config: VllmConfig, rank: int, local_rank: int) -> None:
    torch.accelerator.set_device_index(local_rank)
    torch.set_default_device(torch.device("cuda", local_rank))

    pc = vllm_config.parallel_config
    pc.data_parallel_rank = rank
    pc.rank = 0

    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=pc.world_size,
            rank=0,
            distributed_init_method=get_distributed_init_method(
                pc.master_addr, pc.master_port
            ),
            local_rank=local_rank,
            backend="nccl",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=pc.tensor_parallel_size,
            pipeline_model_parallel_size=pc.pipeline_parallel_size,
        )


def expert_ids_for_rank(layer: torch.nn.Module, num_experts: int) -> torch.Tensor:
    expert_map = layer.routed_experts.expert_map
    if expert_map is None:
        return torch.arange(num_experts, device="cuda")
    return torch.nonzero(expert_map >= 0, as_tuple=False).flatten().to("cuda")


def make_layer_and_inputs(
    args: argparse.Namespace,
    rank: int,
    vllm_config: VllmConfig,
) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    set_random_seed(args.seed)
    dtype = torch.bfloat16
    device = torch.device("cuda", rank)

    full_w13 = torch.randn(
        args.num_experts,
        2 * args.intermediate_size,
        args.hidden_size,
        device=device,
        dtype=dtype,
    ) / 10
    full_w2 = torch.randn(
        args.num_experts,
        args.hidden_size,
        args.intermediate_size,
        device=device,
        dtype=dtype,
    ) / 10

    with set_current_vllm_config(vllm_config):
        layer = FusedMoE(
            num_experts=args.num_experts,
            top_k=args.top_k,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            params_dtype=dtype,
            renormalize=False,
            quant_config=None,
            tp_size=1,
            dp_size=args.world_size,
            pcp_size=1,
            prefix=f"a40_moe_ep_profile_{rank}",
            activation="silu",
        )

    local_ids = expert_ids_for_rank(layer, args.num_experts)
    with torch.no_grad():
        layer.routed_experts.w13_weight.copy_(full_w13.index_select(0, local_ids))
        layer.routed_experts.w2_weight.copy_(full_w2.index_select(0, local_ids))
    layer.routed_experts.quant_method.process_weights_after_loading(
        layer.routed_experts
    )

    hidden_states = (
        torch.randn(args.tokens, args.hidden_size, device=device, dtype=dtype) / 10
    )
    router_logits = torch.randn(
        args.tokens, args.num_experts, device=device, dtype=dtype
    )
    return layer, hidden_states, router_logits


@contextmanager
def make_forward_context(
    args: argparse.Namespace,
    vllm_config: VllmConfig,
    device: torch.device,
):
    num_tokens_across_dp = torch.full(
        (args.world_size,), args.tokens, dtype=torch.int, device=device
    )
    with set_forward_context(
        None,
        vllm_config,
        num_tokens=args.tokens,
        num_tokens_across_dp=num_tokens_across_dp,
    ):
        ctx = get_forward_context()
        if ctx.dp_metadata is None:
            yield
        else:
            with ctx.dp_metadata.sp_local_sizes(1):
                yield


def time_cuda(fn: Callable[[], object], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def run_worker(
    local_rank: int,
    args: argparse.Namespace,
    vllm_config: VllmConfig,
) -> None:
    rank = local_rank
    init_rank(vllm_config, rank=rank, local_rank=local_rank)
    device = torch.device("cuda", local_rank)

    try:
        if not is_workspace_manager_initialized():
            init_workspace_manager(device)

        layer, hidden_states, router_logits = make_layer_and_inputs(
            args, rank, vllm_config
        )
        local_experts = layer.routed_experts.local_num_experts

        def forward_once():
            with make_forward_context(args, vllm_config, device):
                return layer(hidden_states, router_logits)

        full_forward_us = time_cuda(forward_once, args.warmup, args.iters)

        topk_us: float | None = None
        dispatch_us: float | None = None
        combine_us: float | None = None

        def topk_once():
            return layer.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_indices_dtype=layer._quant_method.topk_indices_dtype,
            )

        topk_us = time_cuda(topk_once, args.warmup, args.iters)
        topk_weights, topk_ids = topk_once()
        torch.accelerator.synchronize()

        if args.world_size > 1:

            def dispatch_once():
                with make_forward_context(args, vllm_config, device):
                    return get_ep_group().dispatch(
                        hidden_states,
                        topk_weights,
                        topk_ids,
                        is_sequence_parallel=False,
                    )

            dispatch_us = time_cuda(dispatch_once, args.warmup, args.iters)
            with make_forward_context(args, vllm_config, device):
                dispatched_hidden, _, _ = get_ep_group().dispatch(
                    hidden_states,
                    topk_weights,
                    topk_ids,
                    is_sequence_parallel=False,
                )
            combine_input = torch.randn_like(dispatched_hidden)
            torch.accelerator.synchronize()

            def combine_once():
                with make_forward_context(args, vllm_config, device):
                    return get_ep_group().combine(
                        combine_input,
                        is_sequence_parallel=False,
                    )

            combine_us = time_cuda(combine_once, args.warmup, args.iters)

        result = BenchResult(
            rank=rank,
            world_size=args.world_size,
            backend=args.backend,
            tokens=args.tokens,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_experts=args.num_experts,
            local_experts=local_experts,
            top_k=args.top_k,
            dtype=str(hidden_states.dtype),
            warmup=args.warmup,
            iters=args.iters,
            full_forward_us=full_forward_us,
            topk_us=topk_us,
            dispatch_us=dispatch_us,
            combine_us=combine_us,
        )

        torch.distributed.barrier()
        if rank == 0:
            row = asdict(result)
            if args.csv:
                print(",".join(row.keys()))
                print(",".join("" if v is None else str(v) for v in row.values()))
            else:
                print(json.dumps(row, indent=2, sort_keys=True))
    finally:
        torch.accelerator.synchronize()
        cleanup_dist_env_and_memory()


def main() -> None:
    args = parse_args()
    if args.num_experts % args.world_size != 0:
        raise ValueError("--num-experts must be divisible by --world-size")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.device_count() < args.world_size:
        raise RuntimeError(
            f"Need {args.world_size} CUDA devices, found {torch.cuda.device_count()}"
        )

    vllm_config = make_vllm_config(args)
    if args.world_size == 1:
        run_worker(0, args, vllm_config)
    else:
        spawn(run_worker, args=(args, vllm_config), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
