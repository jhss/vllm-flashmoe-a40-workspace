# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A100-specialized BF16 MoE Triton kernels.

These kernels intentionally target a narrow experimental path:
SM80/A100, BF16, top-k=8 routing, no quantization, no bias, no LoRA.
The goal is to provide a kernel-body fork that can be optimized independently
from the generic fused_moe_kernel.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def a100_bf16_moe_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + offs_m
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_n[None, :]
    c_mask = token_mask[:, None] & (offs_n[None, :] < N)

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        zeros = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.bfloat16)
        tl.store(c_ptrs, zeros, mask=c_mask)
        return

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    offs_bn = offs_n % N
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=offs_k[:, None] < k_remaining,
            other=0.0,
        )
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(
            topk_weights_ptr + offs_token,
            mask=token_mask,
            other=0.0,
        )
        accumulator *= moe_weight[:, None]

    tl.store(c_ptrs, accumulator.to(tl.bfloat16), mask=c_mask)


def invoke_a100_bf16_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
) -> None:
    assert A.dtype == torch.bfloat16
    assert B.dtype == torch.bfloat16
    assert C.dtype == torch.bfloat16
    assert A.is_contiguous()
    assert B.stride(-1) == 1
    assert C.is_contiguous()
    assert sorted_token_ids.stride(0) == 1
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1

    M = A.size(0)
    num_tokens = M * top_k
    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        EM = min(sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"])

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),
    )

    launch_config = dict(config)
    launch_config["SPLIT_K"] = 1
    block_size_k = launch_config.pop("BLOCK_SIZE_K")
    launch_config.pop("SPLIT_K", None)

    a100_bf16_moe_kernel[grid](
        A,
        B,
        C,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.size(1),
        B.size(2),
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        BLOCK_SIZE_K=block_size_k,
        **launch_config,
    )


@triton.jit
def a100_bf16_w2_token_major_reduce_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    expert_map_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_om,
    stride_on,
    stride_twm,
    stride_twk,
    stride_tim,
    stride_tik,
    TOP_K: tl.constexpr,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    for routed_idx in tl.static_range(0, TOP_K):
        expert_id = tl.load(
            topk_ids_ptr + pid_m * stride_tim + routed_idx * stride_tik
        ).to(tl.int64)
        local_expert = expert_id
        if HAS_EXPERT_MAP:
            local_expert = tl.load(
                expert_map_ptr + expert_id,
                mask=expert_id >= 0,
                other=-1,
            ).to(tl.int64)
        valid_expert = local_expert >= 0
        route_weight = tl.load(
            topk_weights_ptr + pid_m * stride_twm + routed_idx * stride_twk,
            mask=valid_expert,
            other=0.0,
        ).to(tl.float32)

        row = pid_m * TOP_K + routed_idx
        a_ptrs = a_ptr + row * stride_am + offs_k * stride_ak
        b_ptrs = (
            b_ptr
            + local_expert * stride_be
            + offs_k[:, None] * stride_bk
            + offs_n[None, :] * stride_bn
        )

        partial = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
        for k_block in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            k_remaining = K - k_block * BLOCK_SIZE_K
            k_mask = offs_k < k_remaining
            a = tl.load(
                a_ptrs,
                mask=valid_expert & k_mask,
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=valid_expert & k_mask[:, None] & (offs_n[None, :] < N),
                other=0.0,
            )
            partial += tl.reshape(
                tl.dot(tl.expand_dims(a, 0), b),
                (BLOCK_SIZE_N,),
            )
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        accumulator += partial * route_weight

    out_ptrs = out_ptr + pid_m * stride_om + offs_n * stride_on
    tl.store(
        out_ptrs,
        accumulator.to(tl.bfloat16),
        mask=(pid_m < M) & (offs_n < N),
    )


def invoke_a100_bf16_w2_token_major_reduce_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    *,
    block_size_n: int = 64,
    block_size_k: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
) -> None:
    assert A.dtype == torch.bfloat16
    assert B.dtype == torch.bfloat16
    assert output.dtype == torch.bfloat16
    assert A.is_contiguous()
    assert B.stride(-1) == 1
    assert output.is_contiguous()
    assert topk_weights.is_contiguous()
    assert topk_ids.is_contiguous()
    assert topk_ids.dim() == 2
    assert topk_weights.shape == topk_ids.shape
    assert topk_ids.size(1) == 8
    assert A.size(0) == topk_ids.size(0) * topk_ids.size(1)
    assert A.size(1) == B.size(2)
    assert output.size(0) == topk_ids.size(0)
    assert output.size(1) == B.size(1)

    M = topk_ids.size(0)
    grid = (M, triton.cdiv(B.size(1), block_size_n))
    expert_map_arg = expert_map if expert_map is not None else topk_ids

    a100_bf16_w2_token_major_reduce_kernel[grid](
        A,
        B,
        output,
        topk_weights,
        topk_ids,
        expert_map_arg,
        M,
        B.size(1),
        B.size(2),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        output.stride(0),
        output.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        TOP_K=8,
        HAS_EXPERT_MAP=expert_map is not None,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
