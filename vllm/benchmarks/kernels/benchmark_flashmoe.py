#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone FlashMoE layer benchmark for experimental vLLM EP work.

This measures FlashMoE's router plus distributed MoE forward path for one
BF16 gated-SiLU MoE layer. Use it next to vLLM serving benchmarks for existing
EP backends such as deepep_low_latency, deepep_high_throughput, and
flashinfer_nvlink_one_sided.

Example:
    PYTHONPATH=/workspace/FlashMoE torchrun --standalone --nproc-per-node=8 \
        benchmarks/kernels/benchmark_flashmoe.py --torch-init
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark FlashMoE router + distributed MoE forward.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--intermediate-size", type=int, default=8192)
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=1, choices=(1, 2))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device-id", type=int, default=None)
    parser.add_argument(
        "--expert-peer-capacity",
        type=int,
        default=None,
        help="FlashMoE capacity per peer. Defaults to tokens * top_k to avoid drops.",
    )
    parser.add_argument(
        "--random-bias",
        action="store_true",
        help="Use random expert biases instead of vLLM-like zero biases.",
    )
    parser.add_argument(
        "--torch-init",
        action="store_true",
        help="Initialize torch.distributed before FlashMoE/NVSHMEM.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Print rank-level rows as CSV instead of a compact summary.",
    )
    parser.add_argument("--label", default="flashmoe")
    return parser.parse_args()


def resolve_device_id(flashmoe: Any, requested: int | None) -> int:
    if requested is not None:
        return requested
    if os.environ.get("LOCAL_RANK") is not None:
        return int(os.environ["LOCAL_RANK"])
    return int(flashmoe.get_local_rank())


def maybe_init_torch_dist(use_torch_init: bool, device_id: int) -> bool:
    if not use_torch_init:
        return False

    import torch.distributed as dist

    torch.cuda.set_device(device_id)
    if dist.is_initialized():
        return False

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", device_id)
    try:
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            rank=rank,
            world_size=world_size,
            device_id=device,
        )
    except (TypeError, ValueError):
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )
    return True


def maybe_destroy_torch_dist(started: bool) -> None:
    if not started:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def shared_seed(rank: int, device_id: int, requested_seed: int) -> int:
    seed = requested_seed
    if seed < 0 and rank == 0:
        seed = random.randint(1, 2**31 - 1)

    torch_device = torch.device("cuda", device_id)
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            seed_tensor = torch.tensor([seed], dtype=torch.int64, device=torch_device)
            dist.broadcast(seed_tensor, src=0)
            return int(seed_tensor.item())
    except ImportError:
        pass

    try:
        from mpi4py import MPI

        return int(MPI.COMM_WORLD.bcast(seed, root=0))
    except ImportError:
        return seed


def gather_times(rank: int, time_ms: float) -> list[float] | None:
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            gathered: list[float] | None = [0.0] * dist.get_world_size()
            if rank != 0:
                gathered = None
            dist.gather_object(time_ms, object_gather_list=gathered, dst=0)
            return gathered
    except ImportError:
        pass

    try:
        from mpi4py import MPI

        gathered = MPI.COMM_WORLD.gather(time_ms, root=0)
        return gathered if rank == 0 else None
    except ImportError:
        return [time_ms] if rank == 0 else None


def make_random(
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return (
        torch.empty(shape, device=device, dtype=dtype)
        .uniform_(-1.0, 1.0)
        .contiguous()
    )


def print_results(args: argparse.Namespace, world: int, times: list[float]) -> None:
    assert times
    max_ms = max(times)
    if args.csv:
        print("label,tokens,hidden,intermediate,experts,top_k,world,rank,ms")
        for rank, rank_ms in enumerate(times):
            print(
                f"{args.label},{args.tokens},{args.hidden_size},"
                f"{args.intermediate_size},{args.num_experts},{args.top_k},"
                f"{world},{rank},{rank_ms:.6f}"
            )
        print(
            "summary,"
            f"{args.tokens},{args.hidden_size},{args.intermediate_size},"
            f"{args.num_experts},{args.top_k},{world},max,{max_ms:.6f}"
        )
        return

    print(
        "FlashMoE router+forward: "
        f"S={args.tokens} H={args.hidden_size} I={args.intermediate_size} "
        f"E={args.num_experts} top_k={args.top_k} world={world}"
    )
    print(
        "per-rank ms: "
        + ", ".join(f"r{rank}={rank_ms:.5f}" for rank, rank_ms in enumerate(times))
    )
    print(
        f"summary ms: avg={statistics.mean(times):.5f} "
        f"p50={statistics.median(times):.5f} max={max_ms:.5f}"
    )


def main() -> None:
    args = parse_args()

    try:
        import flashmoe
    except ImportError as e:
        raise SystemExit(
            "flashmoe is not importable. Install flashmoe-py or set "
            "PYTHONPATH to the FlashMoE checkout."
        ) from e

    if args.iters <= 0:
        raise SystemExit("--iters must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")

    try:
        import cuda.core as cuda
    except ImportError as e:
        raise SystemExit(
            "cuda.core is not importable. Install cuda-python in the FlashMoE "
            "benchmark environment."
        ) from e

    device_id = resolve_device_id(flashmoe, args.device_id)
    started_dist = maybe_init_torch_dist(args.torch_init, device_id)
    dev = cuda.Device(device_id)
    dev.set_current()
    stream = dev.create_stream()
    stream_ptr = int(stream.handle)

    flash_handle = None
    router_handle = None
    try:
        flashmoe.cb.initialize()
        rank = flashmoe.cb.get_rank()
        world = flashmoe.cb.get_world_size()
        if args.num_experts % world != 0:
            raise SystemExit("--num-experts must be divisible by world size")
        local_experts = args.num_experts // world
        expert_map = [expert // local_experts for expert in range(args.num_experts)]

        init_args = flashmoe.InitArgs(
            data_type=flashmoe.DataType.BF16,
            mlp_type=flashmoe.MLPType.GATED,
            act_type=flashmoe.ActivationType.SILU,
            tokens_per_rank=args.tokens,
            token_dim=args.hidden_size,
            ffn_size=args.intermediate_size,
            num_experts=args.num_experts,
            top_k=args.top_k,
            gpu_arch=int(dev.arch) * 10,
            stream_ptr=stream_ptr,
            device_id=device_id,
            ep_world=world,
            num_local_experts=local_experts,
            ep_rank=rank,
            my_pe=rank,
            expert_map=expert_map,
            rank_map=list(range(world)),
            expert_peer_capacity=args.expert_peer_capacity
            if args.expert_peer_capacity is not None
            else args.tokens * args.top_k,
        )
        flash_handle = flashmoe.initialize(init_args)
        router_handle = flashmoe.router.initialize(init_args)

        seed = shared_seed(rank, device_id, args.seed)
        torch_device = torch.device("cuda", device_id)
        dtype = torch.bfloat16

        torch.manual_seed(seed + rank)
        tokens = make_random(
            (args.tokens, args.hidden_size),
            torch_device,
            dtype,
        )
        expert_counts = torch.zeros(
            args.num_experts,
            device=torch_device,
            dtype=torch.int32,
        ).contiguous()

        torch.manual_seed(seed)
        router_weights = make_random(
            (args.hidden_size, args.num_experts),
            torch_device,
            dtype,
        )

        torch.manual_seed(seed + 1009 + rank)
        local_expert_up = make_random(
            (local_experts, args.intermediate_size, args.hidden_size),
            torch_device,
            dtype,
        )
        local_expert_up_v = make_random(
            (local_experts, args.intermediate_size, args.hidden_size),
            torch_device,
            dtype,
        )
        local_expert_down = make_random(
            (local_experts, args.hidden_size, args.intermediate_size),
            torch_device,
            dtype,
        )

        bias_shape = (local_experts, args.intermediate_size)
        down_bias_shape = (local_experts, args.hidden_size)
        if args.random_bias:
            local_bias_up = make_random(bias_shape, torch_device, dtype)
            local_bias_up_v = make_random(bias_shape, torch_device, dtype)
            local_bias_down = make_random(down_bias_shape, torch_device, dtype)
        else:
            local_bias_up = torch.zeros(bias_shape, device=torch_device, dtype=dtype)
            local_bias_up_v = torch.zeros(bias_shape, device=torch_device, dtype=dtype)
            local_bias_down = torch.zeros(
                down_bias_shape,
                device=torch_device,
                dtype=dtype,
            )

        moe_out = torch.empty_like(tokens)
        forward_args = flashmoe.ForwardArgs(
            mt=flashmoe.MLPType.GATED,
            tokens=tokens.data_ptr(),
            expert_counts=expert_counts.data_ptr(),
            local_expert_up=local_expert_up.data_ptr(),
            local_expert_up_v=local_expert_up_v.data_ptr(),
            local_bias_up=local_bias_up.data_ptr(),
            local_bias_up_v=local_bias_up_v.data_ptr(),
            local_expert_down=local_expert_down.data_ptr(),
            local_bias_down=local_bias_down.data_ptr(),
            moe_out=moe_out.data_ptr(),
            stream_ptr=stream_ptr,
        )
        router_args = flashmoe.router.RouterForwardArgs(
            tokens=tokens.data_ptr(),
            weights=router_weights.data_ptr(),
            expert_counts=expert_counts.data_ptr(),
            stream_ptr=stream_ptr,
        )

        torch_stream = torch.cuda.ExternalStream(stream_ptr, device=torch_device)

        def run_one_iter() -> None:
            expert_counts.zero_()
            flashmoe.router.forward(router_handle, flash_handle, router_args)
            flashmoe.forward(flash_handle, forward_args)

        dev.sync()
        with torch.cuda.stream(torch_stream):
            for _ in range(args.warmup):
                run_one_iter()
        torch_stream.synchronize()
        flashmoe.cb.sync_all(stream_ptr)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(torch_stream):
            start.record()
            for _ in range(args.iters):
                run_one_iter()
            end.record()
        end.synchronize()
        flashmoe.cb.sync_all(stream_ptr)

        rank_ms = start.elapsed_time(end) / args.iters
        times = gather_times(rank, rank_ms)
        if rank == 0 and times is not None:
            print_results(args, world, times)
    finally:
        if flash_handle is not None:
            flashmoe.finalize(flash_handle, stream_ptr)
        if router_handle is not None:
            flashmoe.router.finalize(router_handle, stream_ptr)
        stream.close()
        maybe_destroy_torch_dist(started_dist)


if __name__ == "__main__":
    main()
