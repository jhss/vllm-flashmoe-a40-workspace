# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch._subclasses.fake_tensor import FakeTensor

from vllm.triton_utils import tl, triton


@triton.jit
def moe_ep_masked_silu_and_mul_kernel(
    output_ptr,
    input_ptr,
    topk_ids_ptr,
    expert_map_ptr,
    num_rows,
    size: tl.constexpr,
    stride_out_m,
    stride_out_k,
    stride_in_m,
    stride_in_k,
    stride_topk_m,
    stride_topk_k,
    TOP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    pid_k = tl.program_id(1)
    offs_k = pid_k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    k_mask = offs_k < size

    route_idx = pid_m % TOP_K
    token_idx = pid_m // TOP_K
    row_mask = pid_m < num_rows
    expert_id = tl.load(
        topk_ids_ptr + token_idx * stride_topk_m + route_idx * stride_topk_k,
        mask=row_mask,
        other=-1,
    ).to(tl.int64)
    local_expert = tl.load(
        expert_map_ptr + expert_id,
        mask=row_mask & (expert_id >= 0),
        other=-1,
    )
    valid_row = row_mask & (local_expert >= 0)
    mask = valid_row[:, None] & k_mask[None, :]

    gate = tl.load(
        input_ptr + pid_m[:, None] * stride_in_m + offs_k[None, :] * stride_in_k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        input_ptr
        + pid_m[:, None] * stride_in_m
        + (offs_k[None, :] + size) * stride_in_k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    out = gate * tl.sigmoid(gate) * up

    tl.store(
        output_ptr + pid_m[:, None] * stride_out_m + offs_k[None, :] * stride_out_k,
        out.to(output_ptr.dtype.element_ty),
        mask=mask,
    )


def moe_ep_masked_silu_and_mul(
    output: torch.Tensor,
    input: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    assert input.ndim == 2
    assert output.ndim == 2
    assert input.is_contiguous()
    assert output.is_contiguous()
    assert topk_ids.is_contiguous()
    assert input.dtype in (torch.float16, torch.bfloat16)
    assert output.dtype == input.dtype
    assert input.size(1) == output.size(1) * 2
    assert input.size(0) == topk_ids.size(0) * top_k
    assert topk_ids.size(1) == top_k

    if not isinstance(input, FakeTensor):
        size = output.size(1)
        block_m = 1
        block_size = min(triton.next_power_of_2(size), 1024)
        grid = (triton.cdiv(input.size(0), block_m), triton.cdiv(size, block_size))
        moe_ep_masked_silu_and_mul_kernel[grid](
            output,
            input,
            topk_ids,
            expert_map,
            input.size(0),
            size,
            output.stride(0),
            output.stride(1),
            input.stride(0),
            input.stride(1),
            topk_ids.stride(0),
            topk_ids.stride(1),
            TOP_K=top_k,
            BLOCK_M=block_m,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )

    return output
