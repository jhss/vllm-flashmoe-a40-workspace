#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split DeepEP HT direct-assignment cost from prebuilt expert GEMM cost.

This benchmark is diagnostic: it uses a deterministic synthetic top-k routing
tensor and projects it into the same receiver-local id space used by DeepEP HT.
It then compares generic global assignment, local-id generic assignment, and
the DeepEP HT direct assignment helper with identical routing distributions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Callable

import torch

os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

from vllm import _custom_ops as ops  # noqa: E402
from vllm import envs  # noqa: E402
from vllm.model_executor.layers.fused_moe.config import (  # noqa: E402
    FUSED_MOE_UNQUANTIZED_CONFIG,
)
from vllm.model_executor.layers.fused_moe import (  # noqa: E402
    deepep_ht_expert_assignment as ht_assignment,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import (  # noqa: E402
    _a100_moe_tuned_config,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (  # noqa: E402
    invoke_fused_moe_triton_kernel,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import (  # noqa: E402
    moe_fused_mul_sum,
)
from vllm.triton_utils import tl, triton  # noqa: E402

_fill_expert_ids_kernel = ht_assignment._fill_expert_ids_kernel
_make_expert_counts = ht_assignment._make_expert_counts
_scatter_token_ids_kernel = ht_assignment._scatter_token_ids_kernel
deepep_ht_prepare_expert_assignment = (
    ht_assignment.deepep_ht_prepare_expert_assignment
)
deepep_ht_remap_to_local_sentinel = ht_assignment.deepep_ht_remap_to_local_sentinel


@dataclass(frozen=True)
class RoutingInputs:
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    raw_local_topk_ids: torch.Tensor
    local_sentinel_topk_ids: torch.Tensor
    expert_counts: torch.Tensor
    global_expert_map: torch.Tensor
    local_expert_map: torch.Tensor


@dataclass(frozen=True)
class GemmInputs:
    hidden_states: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    w1_out: torch.Tensor
    activation_out: torch.Tensor
    w2_out: torch.Tensor
    output: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="128,512,1024,2048")
    parser.add_argument("--block-ms", default="16,32,64,128")
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=768)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--local-experts", type=int, default=64)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--skip-assignment", action="store_true")
    parser.add_argument("--skip-components", action="store_true")
    parser.add_argument("--skip-gemm", action="store_true")
    parser.add_argument(
        "--nvtx-ranges",
        action="store_true",
        help="Wrap the detailed direct helper clone with NVTX ranges.",
    )
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def iqr(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    q1 = ordered[len(ordered) // 4]
    q3 = ordered[(len(ordered) * 3) // 4]
    return q3 - q1


def summarize_samples(samples: list[float]) -> dict[str, float]:
    return {
        "median_us": median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "iqr_us": iqr(samples),
    }


def time_cuda(
    fn: Callable[[], object],
    *,
    warmup: int,
    iters: int,
    repeats: int,
) -> list[float]:
    samples = []
    for _ in range(repeats):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iters)
    return samples


@contextmanager
def nvtx_range(name: str, enabled: bool):
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


def make_routing_inputs(args: argparse.Namespace, tokens: int) -> RoutingInputs:
    device = torch.device("cuda", args.device)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed * 100_003 + tokens)

    router_logits = torch.randn(
        tokens,
        args.num_experts,
        device=device,
        dtype=torch.float32,
        generator=gen,
    )
    topk_values, topk_ids = torch.topk(router_logits, args.top_k, dim=-1)
    topk_weights = torch.softmax(topk_values, dim=-1).to(torch.bfloat16).contiguous()
    topk_ids = topk_ids.to(torch.int64).contiguous()

    rank_offset = args.rank * args.local_experts
    if rank_offset + args.local_experts > args.num_experts:
        raise ValueError("rank/local-experts range exceeds num-experts")

    global_expert_map = torch.full(
        (args.num_experts,), -1, dtype=torch.int32, device=device
    )
    global_expert_map[rank_offset : rank_offset + args.local_experts] = torch.arange(
        args.local_experts, dtype=torch.int32, device=device
    )
    local_expert_map = torch.empty(
        args.local_experts + 1, dtype=torch.int32, device=device
    )
    local_expert_map[:-1] = torch.arange(
        args.local_experts, dtype=torch.int32, device=device
    )
    local_expert_map[-1] = -1

    raw_local_topk_ids = global_expert_map[topk_ids].contiguous()
    local_sentinel_topk_ids = deepep_ht_remap_to_local_sentinel(
        raw_local_topk_ids, args.local_experts
    ).contiguous()
    valid_ids = raw_local_topk_ids[raw_local_topk_ids >= 0].to(torch.int64)
    expert_counts = torch.bincount(
        valid_ids, minlength=args.local_experts
    ).to(torch.int32)

    return RoutingInputs(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        raw_local_topk_ids=raw_local_topk_ids,
        local_sentinel_topk_ids=local_sentinel_topk_ids,
        expert_counts=expert_counts,
        global_expert_map=global_expert_map,
        local_expert_map=local_expert_map,
    )


def make_gemm_inputs(args: argparse.Namespace, tokens: int) -> GemmInputs:
    device = torch.device("cuda", args.device)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed * 1_000_003 + tokens)
    dtype = torch.bfloat16

    hidden_states = (
        torch.randn(
            tokens,
            args.hidden_size,
            device=device,
            dtype=dtype,
            generator=gen,
        )
        / 10
    ).contiguous()
    w1 = (
        torch.randn(
            args.local_experts,
            2 * args.intermediate_size,
            args.hidden_size,
            device=device,
            dtype=dtype,
            generator=gen,
        )
        / 10
    ).contiguous()
    w2 = (
        torch.randn(
            args.local_experts,
            args.hidden_size,
            args.intermediate_size,
            device=device,
            dtype=dtype,
            generator=gen,
        )
        / 10
    ).contiguous()
    w1_out = torch.empty(
        tokens,
        args.top_k,
        2 * args.intermediate_size,
        dtype=dtype,
        device=device,
    )
    activation_out = torch.empty(
        tokens * args.top_k,
        args.intermediate_size,
        dtype=dtype,
        device=device,
    )
    w2_out = torch.empty(
        tokens,
        args.top_k,
        args.hidden_size,
        dtype=dtype,
        device=device,
    )
    output = torch.empty(tokens, args.hidden_size, dtype=dtype, device=device)
    return GemmInputs(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        w1_out=w1_out,
        activation_out=activation_out,
        w2_out=w2_out,
        output=output,
    )


def build_schedule(
    kind: str,
    routing: RoutingInputs,
    *,
    block_m: int,
    num_experts: int,
    local_experts: int,
    ignore_invalid_experts: bool,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    if kind == "generic_global":
        return moe_align_block_size(
            routing.topk_ids,
            block_m,
            num_experts,
            routing.global_expert_map,
            ignore_invalid_experts=ignore_invalid_experts,
        )
    if kind == "local_id_generic":
        return moe_align_block_size(
            routing.local_sentinel_topk_ids,
            block_m,
            local_experts + 1,
            routing.local_expert_map,
            ignore_invalid_experts=ignore_invalid_experts,
        )
    if kind == "direct":
        return deepep_ht_prepare_expert_assignment(
            routing.raw_local_topk_ids,
            routing.expert_counts,
            block_m,
            ignore_invalid_experts=ignore_invalid_experts,
        )
    raise ValueError(f"unknown schedule kind: {kind}")


def schedule_reduce_ids(
    kind: str, routing: RoutingInputs
) -> tuple[torch.Tensor, torch.Tensor]:
    if kind == "generic_global":
        return routing.topk_ids, routing.global_expert_map
    if kind == "local_id_generic":
        return routing.local_sentinel_topk_ids, routing.local_expert_map
    if kind == "direct":
        return routing.raw_local_topk_ids, routing.local_expert_map
    raise ValueError(f"unknown schedule kind: {kind}")


def schedule_stats(
    routing: RoutingInputs,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    block_m: int,
    ignore_invalid_experts: bool,
    local_experts: int,
) -> dict[str, int | float | str]:
    post_padded = int(num_tokens_post_padded.item())
    used_blocks = cdiv(post_padded, block_m)
    used_expert_ids = expert_ids[:used_blocks]
    valid_mask = (used_expert_ids >= 0) & (used_expert_ids < local_experts)
    valid_expert_ids = used_expert_ids[valid_mask].to(torch.int64)
    block_counts = torch.bincount(
        valid_expert_ids, minlength=local_experts
    ).to(torch.int64)
    padded_counts = block_counts * block_m
    real_counts = routing.expert_counts.to(torch.int64)

    valid_real = int(real_counts.sum().item())
    total_pairs = routing.raw_local_topk_ids.numel()
    scheduled_real = valid_real if ignore_invalid_experts else total_pairs
    total_padding = max(post_padded - scheduled_real, 0)
    valid_padding = max(int(padded_counts.sum().item()) - valid_real, 0)

    return {
        "num_tokens_post_padded": post_padded,
        "used_blocks": used_blocks,
        "allocated_blocks": int(expert_ids.numel()),
        "valid_blocks": int(valid_mask.sum().item()),
        "invalid_blocks": int((~valid_mask).sum().item()),
        "valid_real_tokens": valid_real,
        "total_pairs": total_pairs,
        "total_padding_tokens": total_padding,
        "valid_padding_tokens": valid_padding,
        "total_padding_ratio": total_padding / max(scheduled_real, 1),
        "valid_padding_ratio": valid_padding / max(valid_real, 1),
        "real_count_min": int(real_counts.min().item()),
        "real_count_max": int(real_counts.max().item()),
        "real_count_zero": int((real_counts == 0).sum().item()),
        "padded_count_min": int(padded_counts.min().item()),
        "padded_count_max": int(padded_counts.max().item()),
        "padded_count_zero": int((padded_counts == 0).sum().item()),
        "real_counts_json": json.dumps(real_counts.cpu().tolist()),
        "padded_counts_json": json.dumps(padded_counts.cpu().tolist()),
    }


def direct_detailed_schedule(
    routing: RoutingInputs,
    *,
    block_m: int,
    ignore_invalid_experts: bool,
    nvtx_ranges: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    include_invalid = not ignore_invalid_experts
    topk_ids = routing.raw_local_topk_ids
    num_local_experts = routing.expert_counts.numel()

    with nvtx_range("direct_counts", nvtx_ranges):
        expert_counts = _make_expert_counts(
            topk_ids, routing.expert_counts, include_invalid=include_invalid
        )
        num_schedule_experts = expert_counts.numel()

    with nvtx_range("direct_prefix_sum", nvtx_ranges):
        padded_counts = (
            torch.div(
                expert_counts + block_m - 1,
                block_m,
                rounding_mode="floor",
            )
            * block_m
        ).to(torch.int32)
        expert_offsets = torch.empty_like(padded_counts)
        expert_offsets[0] = 0
        if num_schedule_experts > 1:
            torch.cumsum(
                padded_counts[:-1],
                dim=0,
                dtype=torch.int32,
                out=expert_offsets[1:],
            )
        num_tokens_post_padded = padded_counts.sum(dtype=torch.int32).view(1)

    with nvtx_range("direct_alloc_init", nvtx_ranges):
        max_num_tokens_padded = topk_ids.numel() + num_schedule_experts * (
            block_m - 1
        )
        max_num_blocks = triton.cdiv(max_num_tokens_padded, block_m)
        sorted_token_ids = torch.full(
            (max_num_tokens_padded,),
            topk_ids.numel(),
            dtype=torch.int32,
            device=topk_ids.device,
        )
        expert_ids = torch.full(
            (max_num_blocks,),
            -1,
            dtype=torch.int32,
            device=topk_ids.device,
        )
        expert_write_offsets = torch.zeros_like(expert_counts)
        overflow_flag = expert_write_offsets

    with nvtx_range("direct_fill_expert_ids", nvtx_ranges):
        blocks_per_program = 64
        max_blocks_per_expert = triton.cdiv(topk_ids.numel(), block_m)
        if max_blocks_per_expert > 0:
            grid = (
                num_schedule_experts,
                triton.cdiv(max_blocks_per_expert, blocks_per_program),
            )
            _fill_expert_ids_kernel[grid](
                expert_counts,
                expert_offsets,
                expert_ids,
                num_schedule_experts,
                num_local_experts,
                block_m,
                blocks_per_program,
            )

    with nvtx_range("direct_scatter_token_ids", nvtx_ranges):
        block_size = 256
        _scatter_token_ids_kernel[(triton.cdiv(topk_ids.numel(), block_size),)](
            topk_ids,
            expert_counts,
            expert_offsets,
            expert_write_offsets,
            sorted_token_ids,
            overflow_flag,
            topk_ids.numel(),
            num_local_experts,
            include_invalid,
            False,
            block_size,
        )
    return sorted_token_ids, expert_ids, num_tokens_post_padded


def direct_component_inputs(
    routing: RoutingInputs,
    *,
    block_m: int,
    ignore_invalid_experts: bool,
):
    include_invalid = not ignore_invalid_experts
    topk_ids = routing.raw_local_topk_ids
    expert_counts = _make_expert_counts(
        topk_ids, routing.expert_counts, include_invalid=include_invalid
    )
    padded_counts = (
        torch.div(expert_counts + block_m - 1, block_m, rounding_mode="floor")
        * block_m
    ).to(torch.int32)
    expert_offsets = torch.empty_like(padded_counts)
    expert_offsets[0] = 0
    if expert_counts.numel() > 1:
        torch.cumsum(
            padded_counts[:-1],
            dim=0,
            dtype=torch.int32,
            out=expert_offsets[1:],
        )
    max_num_tokens_padded = topk_ids.numel() + expert_counts.numel() * (block_m - 1)
    max_num_blocks = triton.cdiv(max_num_tokens_padded, block_m)
    sorted_token_ids = torch.full(
        (max_num_tokens_padded,),
        topk_ids.numel(),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    expert_ids = torch.full(
        (max_num_blocks,), -1, dtype=torch.int32, device=topk_ids.device
    )
    write_offsets = torch.zeros_like(expert_counts)
    torch.cuda.synchronize()
    return (
        expert_counts,
        padded_counts,
        expert_offsets,
        sorted_token_ids,
        expert_ids,
        write_offsets,
    )


def direct_component_functions(
    routing: RoutingInputs,
    *,
    block_m: int,
    ignore_invalid_experts: bool,
) -> dict[str, Callable[[], object]]:
    include_invalid = not ignore_invalid_experts
    topk_ids = routing.raw_local_topk_ids
    (
        expert_counts,
        _,
        expert_offsets,
        sorted_token_ids,
        expert_ids,
        write_offsets,
    ) = direct_component_inputs(
        routing,
        block_m=block_m,
        ignore_invalid_experts=ignore_invalid_experts,
    )
    num_schedule_experts = expert_counts.numel()
    num_local_experts = routing.expert_counts.numel()
    max_num_tokens_padded = topk_ids.numel() + num_schedule_experts * (block_m - 1)
    max_num_blocks = triton.cdiv(max_num_tokens_padded, block_m)

    def counts():
        return _make_expert_counts(
            topk_ids, routing.expert_counts, include_invalid=include_invalid
        )

    def prefix_sum():
        padded_counts = (
            torch.div(expert_counts + block_m - 1, block_m, rounding_mode="floor")
            * block_m
        ).to(torch.int32)
        offsets = torch.empty_like(padded_counts)
        offsets[0] = 0
        if num_schedule_experts > 1:
            torch.cumsum(
                padded_counts[:-1], dim=0, dtype=torch.int32, out=offsets[1:]
            )
        return padded_counts.sum(dtype=torch.int32).view(1)

    def alloc_init():
        sorted_ids = torch.full(
            (max_num_tokens_padded,),
            topk_ids.numel(),
            dtype=torch.int32,
            device=topk_ids.device,
        )
        ids = torch.full(
            (max_num_blocks,), -1, dtype=torch.int32, device=topk_ids.device
        )
        offsets = torch.zeros_like(expert_counts)
        return sorted_ids, ids, offsets

    def fill_expert_ids():
        blocks_per_program = 64
        max_blocks_per_expert = triton.cdiv(topk_ids.numel(), block_m)
        if max_blocks_per_expert > 0:
            grid = (
                num_schedule_experts,
                triton.cdiv(max_blocks_per_expert, blocks_per_program),
            )
            _fill_expert_ids_kernel[grid](
                expert_counts,
                expert_offsets,
                expert_ids,
                num_schedule_experts,
                num_local_experts,
                block_m,
                blocks_per_program,
            )

    def scatter_token_ids():
        write_offsets.zero_()
        block_size = 256
        _scatter_token_ids_kernel[(triton.cdiv(topk_ids.numel(), block_size),)](
            topk_ids,
            expert_counts,
            expert_offsets,
            write_offsets,
            sorted_token_ids,
            write_offsets,
            topk_ids.numel(),
            num_local_experts,
            include_invalid,
            False,
            block_size,
        )

    return {
        "direct_counts": counts,
        "direct_prefix_sum": prefix_sum,
        "direct_alloc_init": alloc_init,
        "direct_fill_expert_ids": fill_expert_ids,
        "direct_scatter_token_ids": scatter_token_ids,
    }


def actual_gemm_config(
    gemm: GemmInputs,
    args: argparse.Namespace,
    tokens: int,
) -> dict[str, int]:
    config = try_get_optimal_moe_config(
        gemm.w1.size(),
        gemm.w2.size(),
        args.top_k,
        FUSED_MOE_UNQUANTIZED_CONFIG.config_name(torch.bfloat16),
        tokens,
        block_shape=None,
    )
    if (
        envs.VLLM_MOE_TRITON_W1_A100_TUNED_CONFIG
        or envs.VLLM_MOE_TRITON_W2_A100_TUNED_CONFIG
        or envs.VLLM_MOE_A100_BF16_SPECIALIZED_KERNELS
    ):
        config = _a100_moe_tuned_config(config)
    return config


def run_w1(
    gemm: GemmInputs,
    schedule: tuple[torch.Tensor | None, torch.Tensor, torch.Tensor],
    config: dict[str, int],
    args: argparse.Namespace,
) -> None:
    sorted_ids, expert_ids, num_post = schedule
    invoke_fused_moe_triton_kernel(
        gemm.hidden_states,
        gemm.w1,
        gemm.w1_out,
        None,
        None,
        None,
        sorted_ids,
        expert_ids,
        num_post,
        False,
        args.top_k,
        config,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
    )


def run_activation(gemm: GemmInputs, args: argparse.Namespace) -> None:
    torch.ops._C.silu_and_mul(
        gemm.activation_out,
        gemm.w1_out.view(-1, 2 * args.intermediate_size),
    )


def run_w2(
    gemm: GemmInputs,
    routing: RoutingInputs,
    schedule: tuple[torch.Tensor | None, torch.Tensor, torch.Tensor],
    config: dict[str, int],
    args: argparse.Namespace,
    *,
    ignore_invalid_experts: bool,
) -> None:
    sorted_ids, expert_ids, num_post = schedule
    invoke_fused_moe_triton_kernel(
        gemm.activation_out,
        gemm.w2,
        gemm.w2_out,
        None,
        None,
        routing.topk_weights,
        sorted_ids,
        expert_ids,
        num_post,
        not ignore_invalid_experts,
        1,
        config,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
    )


def run_reduce(
    gemm: GemmInputs,
    routing: RoutingInputs,
    kind: str,
    *,
    ignore_invalid_experts: bool,
) -> None:
    if ignore_invalid_experts:
        topk_ids, expert_map = schedule_reduce_ids(kind, routing)
        moe_fused_mul_sum(
            gemm.w2_out,
            routing.topk_weights,
            gemm.output,
            topk_ids,
            expert_map,
        )
    else:
        ops.moe_sum(gemm.w2_out, gemm.output)


def benchmark_assignment(
    args: argparse.Namespace,
    tokens_list: list[int],
    block_ms: list[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for tokens in tokens_list:
        routing = make_routing_inputs(args, tokens)
        for block_m in block_ms:
            for ignore in (False, True):
                for kind in ("generic_global", "local_id_generic", "direct"):
                    samples = time_cuda(
                        lambda: build_schedule(
                            kind,
                            routing,
                            block_m=block_m,
                            num_experts=args.num_experts,
                            local_experts=args.local_experts,
                            ignore_invalid_experts=ignore,
                        ),
                        warmup=args.warmup,
                        iters=args.iters,
                        repeats=args.repeats,
                    )
                    _, expert_ids, num_post = build_schedule(
                        kind,
                        routing,
                        block_m=block_m,
                        num_experts=args.num_experts,
                        local_experts=args.local_experts,
                        ignore_invalid_experts=ignore,
                    )
                    torch.cuda.synchronize()
                    row: dict[str, object] = {
                        "phase": "assignment",
                        "tokens": tokens,
                        "block_m": block_m,
                        "ignore_invalid": int(ignore),
                        "kind": kind,
                        "warmup": args.warmup,
                        "iters": args.iters,
                        "repeats": args.repeats,
                        **summarize_samples(samples),
                    }
                    row.update(
                        schedule_stats(
                            routing,
                            expert_ids,
                            num_post,
                            block_m=block_m,
                            ignore_invalid_experts=ignore,
                            local_experts=args.local_experts,
                        )
                    )
                    rows.append(row)
    return rows


def benchmark_components(
    args: argparse.Namespace,
    tokens_list: list[int],
    block_ms: list[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for tokens in tokens_list:
        routing = make_routing_inputs(args, tokens)
        for block_m in block_ms:
            for ignore in (False, True):
                # Also execute the detailed clone once to make optional NVTX
                # ranges available for Nsight runs.
                direct_detailed_schedule(
                    routing,
                    block_m=block_m,
                    ignore_invalid_experts=ignore,
                    nvtx_ranges=args.nvtx_ranges,
                )
                torch.cuda.synchronize()
                for component, fn in direct_component_functions(
                    routing,
                    block_m=block_m,
                    ignore_invalid_experts=ignore,
                ).items():
                    samples = time_cuda(
                        fn,
                        warmup=args.warmup,
                        iters=args.iters,
                        repeats=args.repeats,
                    )
                    rows.append(
                        {
                            "phase": "direct_component",
                            "tokens": tokens,
                            "block_m": block_m,
                            "ignore_invalid": int(ignore),
                            "kind": component,
                            "warmup": args.warmup,
                            "iters": args.iters,
                            "repeats": args.repeats,
                            **summarize_samples(samples),
                        }
                    )
    return rows


def benchmark_gemm(
    args: argparse.Namespace,
    tokens_list: list[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for tokens in tokens_list:
        routing = make_routing_inputs(args, tokens)
        gemm = make_gemm_inputs(args, tokens)
        config = actual_gemm_config(gemm, args, tokens)
        block_m = int(config["BLOCK_SIZE_M"])

        for ignore in (False, True):
            for kind in ("generic_global", "local_id_generic", "direct"):
                schedule = build_schedule(
                    kind,
                    routing,
                    block_m=block_m,
                    num_experts=args.num_experts,
                    local_experts=args.local_experts,
                    ignore_invalid_experts=ignore,
                )
                # Fill dependent buffers once before measuring isolated stages.
                run_w1(gemm, schedule, config, args)
                run_activation(gemm, args)
                run_w2(
                    gemm,
                    routing,
                    schedule,
                    config,
                    args,
                    ignore_invalid_experts=ignore,
                )
                run_reduce(gemm, routing, kind, ignore_invalid_experts=ignore)
                torch.cuda.synchronize()

                phase_fns: dict[str, Callable[[], object]] = {
                    "w1_only": lambda schedule=schedule, config=config: run_w1(
                        gemm, schedule, config, args
                    ),
                    "activation_only": lambda: run_activation(gemm, args),
                    "w2_only": (
                        lambda schedule=schedule, config=config, ignore=ignore: run_w2(
                            gemm,
                            routing,
                            schedule,
                            config,
                            args,
                            ignore_invalid_experts=ignore,
                        )
                    ),
                    "reduce_only": lambda kind=kind, ignore=ignore: run_reduce(
                        gemm, routing, kind, ignore_invalid_experts=ignore
                    ),
                    "experts_prebuilt": (
                        lambda schedule=schedule,
                        config=config,
                        kind=kind,
                        ignore=ignore: (
                            run_w1(gemm, schedule, config, args),
                            run_activation(gemm, args),
                            run_w2(
                                gemm,
                                routing,
                                schedule,
                                config,
                                args,
                                ignore_invalid_experts=ignore,
                            ),
                            run_reduce(
                                gemm,
                                routing,
                                kind,
                                ignore_invalid_experts=ignore,
                            ),
                        )
                    ),
                }
                for gemm_phase, fn in phase_fns.items():
                    samples = time_cuda(
                        fn,
                        warmup=args.warmup,
                        iters=args.iters,
                        repeats=args.repeats,
                    )
                    _, expert_ids, num_post = schedule
                    row: dict[str, object] = {
                        "phase": "gemm",
                        "gemm_phase": gemm_phase,
                        "tokens": tokens,
                        "block_m": block_m,
                        "ignore_invalid": int(ignore),
                        "kind": kind,
                        "warmup": args.warmup,
                        "iters": args.iters,
                        "repeats": args.repeats,
                        "config_json": json.dumps(config, sort_keys=True),
                        **summarize_samples(samples),
                    }
                    row.update(
                        schedule_stats(
                            routing,
                            expert_ids,
                            num_post,
                            block_m=block_m,
                            ignore_invalid_experts=ignore,
                            local_experts=args.local_experts,
                        )
                    )
                    rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc}"


def write_metadata(
    args: argparse.Namespace,
    tokens: list[int],
    block_ms: list[int],
) -> None:
    metadata = {
        "argv": sys.argv,
        "tokens": tokens,
        "block_ms": block_ms,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(args.device),
        "device_capability": torch.cuda.get_device_capability(args.device),
        "vllm_commit": command_output(["git", "rev-parse", "HEAD"]),
        "nvidia_smi_topo": command_output(["nvidia-smi", "topo", "-m"]),
        "env": {
            name: os.environ.get(name, "")
            for name in (
                "VLLM_DEEPEP_HT_DIRECT_ASSIGNMENT",
                "VLLM_DEEPEP_HT_DIRECT_ASSIGNMENT_DEBUG",
                "VLLM_DEEPEP_HT_LOCAL_EXPERT_IDS",
                "VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS",
                "VLLM_MOE_TRITON_EP_MASKED_ACTIVATION",
                "VLLM_MOE_TRITON_W1_A100_TUNED_CONFIG",
                "VLLM_MOE_TRITON_W2_A100_TUNED_CONFIG",
                "VLLM_MOE_A100_BF16_SPECIALIZED_KERNELS",
                "VLLM_MOE_TRITON_TOPK8_SUM",
                "CUDA_VISIBLE_DEVICES",
                "PYTHONPATH",
            )
        },
    }
    with args.output_prefix.with_suffix(".metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    with args.output_prefix.with_suffix(".commands.log").open("w") as f:
        f.write(" ".join(sys.argv))
        f.write("\n")


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(args.device)
    tokens = parse_int_list(args.tokens)
    block_ms = parse_int_list(args.block_ms)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    write_metadata(args, tokens, block_ms)

    if not args.skip_assignment:
        assignment_rows = benchmark_assignment(args, tokens, block_ms)
        write_csv(
            args.output_prefix.with_name(
                args.output_prefix.name + "_assignment.csv"
            ),
            assignment_rows,
        )

    if not args.skip_components:
        component_rows = benchmark_components(args, tokens, block_ms)
        write_csv(
            args.output_prefix.with_name(
                args.output_prefix.name + "_components.csv"
            ),
            component_rows,
        )

    if not args.skip_gemm:
        gemm_rows = benchmark_gemm(args, tokens)
        write_csv(
            args.output_prefix.with_name(args.output_prefix.name + "_gemm.csv"),
            gemm_rows,
        )


if __name__ == "__main__":
    main()
