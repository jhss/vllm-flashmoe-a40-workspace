# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the DeepEP HT direct expert assignment helper."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.deepep_ht_expert_assignment import (
    _make_expert_counts,
    deepep_ht_prepare_expert_assignment,
    deepep_ht_remap_to_local_sentinel,
)
from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import (
    moe_fused_mul_sum,
)


def _expert_counts(topk_ids: torch.Tensor, num_local_experts: int) -> torch.Tensor:
    flat = topk_ids.flatten()
    valid = flat[(flat >= 0) & (flat < num_local_experts)]
    return torch.bincount(valid.to(torch.int64), minlength=num_local_experts).to(
        torch.int32
    )


def _assert_assignment_matches_topk(
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    block_size_m: int,
    num_local_experts: int,
    *,
    ignore_invalid_experts: bool,
) -> None:
    flat_topk_ids = topk_ids.flatten()
    num_pairs = flat_topk_ids.numel()
    used_tokens = int(num_tokens_post_padded.item())

    actual_pairs: list[int] = []
    for block_idx in range(used_tokens // block_size_m):
        expert_id = int(expert_ids[block_idx].item())
        block = sorted_token_ids[
            block_idx * block_size_m : (block_idx + 1) * block_size_m
        ]
        valid_pairs = block[block < num_pairs]
        padding = block[block >= num_pairs]

        assert torch.all(padding == num_pairs)
        actual_pairs.extend(valid_pairs.cpu().tolist())

        if valid_pairs.numel() == 0:
            continue

        routed_experts = flat_topk_ids[valid_pairs.to(torch.long)]
        if expert_id == -1:
            assert not ignore_invalid_experts
            assert torch.all(
                (routed_experts < 0) | (routed_experts >= num_local_experts)
            )
        else:
            assert 0 <= expert_id < num_local_experts
            assert torch.all(routed_experts == expert_id)

    valid_mask = (flat_topk_ids >= 0) & (flat_topk_ids < num_local_experts)
    if ignore_invalid_experts:
        expected_pairs = torch.arange(num_pairs, device=topk_ids.device)[valid_mask]
    else:
        expected_pairs = torch.arange(num_pairs, device=topk_ids.device)

    actual_pairs_tensor = torch.tensor(
        actual_pairs, dtype=torch.int64, device=topk_ids.device
    )
    torch.testing.assert_close(
        torch.sort(actual_pairs_tensor)[0],
        torch.sort(expected_pairs.to(torch.int64))[0],
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_deepep_ht_remap_to_local_sentinel(dtype: torch.dtype):
    topk_ids = torch.tensor(
        [[-1, 0, 3], [4, -1, 2]],
        dtype=dtype,
    )

    remapped = deepep_ht_remap_to_local_sentinel(topk_ids, num_local_experts=4)

    expected = torch.tensor(
        [[4, 0, 3], [4, 4, 2]],
        dtype=dtype,
    )
    torch.testing.assert_close(remapped, expected, atol=0, rtol=0)
    assert remapped.dtype == dtype


def test_make_expert_counts_keeps_int32_with_invalid():
    topk_ids = torch.tensor([[-1, 0, 1], [2, 4, -1]], dtype=torch.int64)
    expert_num_tokens = torch.tensor([1, 1, 1, 0], dtype=torch.int64)

    expert_counts = _make_expert_counts(
        topk_ids, expert_num_tokens, include_invalid=True
    )

    torch.testing.assert_close(
        expert_counts,
        torch.tensor([1, 1, 1, 0, 3], dtype=torch.int32),
        atol=0,
        rtol=0,
    )
    assert expert_counts.dtype == torch.int32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_moe_fused_mul_sum_masks_raw_negative_topk_ids():
    inputs = torch.arange(2 * 3 * 4, device="cuda", dtype=torch.float32).view(2, 3, 4)
    topk_weights = torch.tensor(
        [[0.5, 1.0, 1.5], [2.0, 0.25, 0.75]],
        device="cuda",
        dtype=torch.float32,
    )
    topk_ids = torch.tensor(
        [[-1, 0, 2], [3, -1, 5]],
        device="cuda",
        dtype=torch.int64,
    )
    expert_map = torch.tensor([0, -1, 1, 2], device="cuda", dtype=torch.int32)
    output = torch.empty((2, 4), device="cuda", dtype=torch.float32)

    actual = moe_fused_mul_sum(inputs, topk_weights, output, topk_ids, expert_map)

    expected = torch.zeros_like(output)
    for m in range(topk_ids.size(0)):
        for k in range(topk_ids.size(1)):
            expert_id = int(topk_ids[m, k].item())
            if (
                0 <= expert_id < expert_map.numel()
                and int(expert_map[expert_id].item()) >= 0
            ):
                expected[m] += inputs[m, k] * topk_weights[m, k]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("m", [1, 17, 256])
@pytest.mark.parametrize("num_local_experts", [4, 64])
@pytest.mark.parametrize("block_size_m", [16, 64])
@pytest.mark.parametrize("topk_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("ignore_invalid_experts", [False, True])
def test_deepep_ht_prepare_expert_assignment(
    m: int,
    num_local_experts: int,
    block_size_m: int,
    topk_dtype: torch.dtype,
    ignore_invalid_experts: bool,
):
    topk = 8
    topk_ids = (
        torch.arange(m * topk, dtype=topk_dtype, device="cuda")
        .remainder(num_local_experts + 1)
        .view(m, topk)
    )
    topk_ids[0, 0] = -1
    topk_ids[-1, -1] = num_local_experts
    expert_num_tokens = _expert_counts(topk_ids, num_local_experts)

    sorted_token_ids, expert_ids, num_tokens_post_padded = (
        deepep_ht_prepare_expert_assignment(
            topk_ids,
            expert_num_tokens,
            block_size_m,
            ignore_invalid_experts=ignore_invalid_experts,
        )
    )

    valid_count = int(expert_num_tokens.sum().item())
    invalid_count = topk_ids.numel() - valid_count
    scheduled_count = valid_count if ignore_invalid_experts else topk_ids.numel()
    expected_padded = 0
    for count in expert_num_tokens.cpu().tolist():
        expected_padded += ((count + block_size_m - 1) // block_size_m) * block_size_m
    if not ignore_invalid_experts and invalid_count > 0:
        expected_padded += (
            (invalid_count + block_size_m - 1) // block_size_m
        ) * block_size_m

    assert int(num_tokens_post_padded.item()) == expected_padded
    assert (
        int((sorted_token_ids[:expected_padded] < topk_ids.numel()).sum().item())
        == scheduled_count
    )
    _assert_assignment_matches_topk(
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        block_size_m,
        num_local_experts,
        ignore_invalid_experts=ignore_invalid_experts,
    )
