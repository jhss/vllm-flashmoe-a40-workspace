# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepEP HT local expert assignment helpers."""

import torch

from vllm import envs
from vllm.triton_utils import tl, triton


def _assert_async(condition: torch.Tensor) -> None:
    assert_async = getattr(torch, "_assert_async", None)
    if condition.device.type == "cpu":
        assert bool(condition.item())
    elif assert_async is not None:
        assert_async(condition)


@triton.jit
def _fill_expert_ids_kernel(
    expert_counts,
    expert_offsets,
    expert_ids,
    num_schedule_experts: tl.constexpr,
    num_local_experts: tl.constexpr,
    block_size_m: tl.constexpr,
    blocks_per_program: tl.constexpr,
):
    expert_idx = tl.program_id(0)
    block_chunk = tl.program_id(1)
    block_offsets = block_chunk * blocks_per_program + tl.arange(
        0, blocks_per_program
    )

    count = tl.load(expert_counts + expert_idx)
    num_blocks = (count + block_size_m - 1) // block_size_m
    start = tl.load(expert_offsets + expert_idx) // block_size_m
    expert_id = tl.where(expert_idx < num_local_experts, expert_idx, -1)
    mask = (expert_idx < num_schedule_experts) & (block_offsets < num_blocks)
    tl.store(expert_ids + start + block_offsets, expert_id, mask=mask)


@triton.jit
def _scatter_token_ids_kernel(
    topk_ids,
    expert_counts,
    expert_offsets,
    expert_write_offsets,
    sorted_token_ids,
    overflow_flag,
    num_pairs,
    num_local_experts: tl.constexpr,
    include_invalid: tl.constexpr,
    debug_validate: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < num_pairs
    expert_ids = tl.load(topk_ids + offsets, mask=mask, other=-1)
    is_valid = (expert_ids >= 0) & (expert_ids < num_local_experts)
    schedule_expert = tl.where(is_valid, expert_ids, num_local_experts)
    if include_invalid:
        should_write = mask
    else:
        should_write = mask & is_valid

    expert_pos = tl.atomic_add(
        expert_write_offsets + schedule_expert, 1, sem="relaxed", mask=should_write
    )
    if debug_validate:
        expert_count = tl.load(
            expert_counts + schedule_expert, mask=should_write, other=0
        )
        store_mask = should_write & (expert_pos < expert_count)
        overflow = should_write & (expert_pos >= expert_count)
        tl.store(overflow_flag, 1, mask=tl.any(overflow))
    else:
        store_mask = should_write
    output_offsets = tl.load(
        expert_offsets + schedule_expert, mask=should_write, other=0
    ) + expert_pos
    tl.store(sorted_token_ids + output_offsets, offsets, mask=store_mask)


def deepep_ht_remap_to_local_sentinel(
    topk_ids: torch.Tensor, num_local_experts: int
) -> torch.Tensor:
    """Convert raw DeepEP invalid IDs to the local sentinel for generic align."""
    invalid_sentinel = torch.full(
        (), num_local_experts, dtype=topk_ids.dtype, device=topk_ids.device
    )
    return torch.where(topk_ids == -1, invalid_sentinel, topk_ids)


def _make_expert_counts(
    topk_ids: torch.Tensor,
    expert_num_tokens: torch.Tensor,
    include_invalid: bool,
) -> torch.Tensor:
    counts = expert_num_tokens
    if counts.dtype != torch.int32:
        counts = counts.to(torch.int32)
    if not counts.is_contiguous():
        counts = counts.contiguous()

    if not include_invalid:
        return counts

    valid_count = counts.sum(dtype=torch.int32)
    num_pairs = counts.new_full((), topk_ids.numel(), dtype=torch.int32)
    raw_invalid_count = num_pairs - valid_count
    _assert_async(raw_invalid_count >= 0)
    invalid_count = torch.clamp(raw_invalid_count, min=0).to(torch.int32).view(1)
    return torch.cat((counts, invalid_count))


def deepep_ht_prepare_expert_assignment(
    topk_ids: torch.Tensor,
    expert_num_tokens: torch.Tensor,
    block_size_m: int,
    *,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a Triton MoE assignment schedule from DeepEP HT local metadata.

    ``topk_ids`` must already be in DeepEP HT receiver-local expert-id space,
    where invalid entries use the sentinel ``num_local_experts`` or ``-1``.
    """
    assert topk_ids.is_cuda, "DeepEP HT direct assignment requires CUDA topk_ids"
    assert expert_num_tokens.is_cuda, (
        "DeepEP HT direct assignment requires CUDA expert_num_tokens"
    )
    assert topk_ids.device == expert_num_tokens.device, (
        "topk_ids and expert_num_tokens must be on the same device"
    )
    assert topk_ids.dtype in (torch.int32, torch.int64), (
        "topk_ids must be int32 or int64"
    )
    assert expert_num_tokens.dtype in (torch.int32, torch.int64), (
        "expert_num_tokens must be int32 or int64"
    )
    assert topk_ids.is_contiguous(), "topk_ids must be contiguous"
    assert block_size_m > 0, "block_size_m must be positive"
    assert expert_num_tokens.dim() == 1, "expert_num_tokens must be 1D"
    assert expert_num_tokens.numel() > 0, "expert_num_tokens must be non-empty"

    if topk_ids.numel() == 0:
        num_schedule_experts = expert_num_tokens.numel()
        max_num_tokens_padded = num_schedule_experts * (block_size_m - 1)
        max_num_blocks = triton.cdiv(max_num_tokens_padded, block_size_m)
        return (
            torch.full(
                (max_num_tokens_padded,),
                topk_ids.numel(),
                dtype=torch.int32,
                device=topk_ids.device,
            ),
            torch.full(
                (max_num_blocks,),
                -1,
                dtype=torch.int32,
                device=topk_ids.device,
            ),
            torch.zeros((1,), dtype=torch.int32, device=topk_ids.device),
        )

    num_local_experts = expert_num_tokens.numel()
    include_invalid = not ignore_invalid_experts
    expert_counts = _make_expert_counts(
        topk_ids, expert_num_tokens, include_invalid=include_invalid
    )
    num_schedule_experts = expert_counts.numel()

    padded_counts = (
        torch.div(
            expert_counts + block_size_m - 1, block_size_m, rounding_mode="floor"
        )
        * block_size_m
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

    max_num_tokens_padded = topk_ids.numel() + num_schedule_experts * (
        block_size_m - 1
    )
    max_num_blocks = triton.cdiv(max_num_tokens_padded, block_size_m)
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
    debug_validate = envs.VLLM_DEEPEP_HT_DIRECT_ASSIGNMENT_DEBUG
    overflow_flag = (
        torch.zeros((1,), dtype=torch.int32, device=topk_ids.device)
        if debug_validate
        else expert_write_offsets
    )

    blocks_per_program = 64
    max_blocks_per_expert = triton.cdiv(topk_ids.numel(), block_size_m)
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
            block_size_m,
            blocks_per_program,
        )

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
        debug_validate,
        block_size,
    )
    if debug_validate:
        _assert_async(torch.all(overflow_flag == 0))

    return sorted_token_ids, expert_ids, num_tokens_post_padded
