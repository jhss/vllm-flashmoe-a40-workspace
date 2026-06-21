#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check DeepEP HT BLOCK_M override output closeness.

The benchmark runner measures latency in separate processes, which is the
cleanest way to avoid cross-setting state. This smoke instead keeps one layer
and one synthetic input alive per rank, runs default/M64/M32 settings, and
compares their final MoE outputs with BF16 tolerances.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.multiprocessing import spawn

os.environ.setdefault("NCCL_P2P_DISABLE", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.kernels import benchmark_moe_ep_a40 as bench  # noqa: E402
from vllm.distributed import cleanup_dist_env_and_memory  # noqa: E402
from vllm.v1.worker.workspace import (  # noqa: E402
    init_workspace_manager,
    is_workspace_manager_initialized,
)

from vllm import envs  # noqa: E402


@dataclass(frozen=True)
class CorrectnessSetting:
    name: str
    w1_block_m: int
    w2_block_m: int
    fixed_capacity: bool = False


@dataclass
class CorrectnessMetric:
    rank: int
    tokens: int
    setting: str
    reference: str
    max_abs_error: float
    mean_abs_error: float
    relative_l2_error: float
    rtol: float
    atol: float
    assert_close: bool
    assert_close_error: str | None


SETTINGS = (
    CorrectnessSetting("block_default", 0, 0),
    CorrectnessSetting("block_both_64", 64, 64),
    CorrectnessSetting("block_both_32", 32, 32),
    CorrectnessSetting("fixed_both_64", 64, 64, fixed_capacity=True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=2, choices=[1, 2])
    parser.add_argument("--backend", default="deepep_high_throughput")
    parser.add_argument("--tokens", type=int, default=320)
    parser.add_argument("--rank-tokens", type=int, nargs="+")
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=768)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--input-seed-base", type=int, default=1007)
    parser.add_argument("--rank-distinct-inputs", action="store_true", default=True)
    parser.add_argument("--deepep-ht-num-sms", type=int, default=24)
    parser.add_argument("--nccl-p2p-disable", default="0")
    parser.add_argument("--route-target-rank", type=int, default=-1)
    parser.add_argument("--fixed-capacity-num-worst-tokens", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=1.6e-2)
    parser.add_argument("--atol", type=float, default=1.0e-2)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def benchmark_args(args: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(
        world_size=args.world_size,
        backend=args.backend,
        tokens=args.tokens,
        rank_tokens=args.rank_tokens,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
        warmup=1,
        iters=1,
        seed=args.seed,
        rank_distinct_inputs=args.rank_distinct_inputs,
        input_seed_base=args.input_seed_base,
        csv=False,
        section_profile_iters=0,
        section_profile_warmup=0,
        section_profile_output=None,
        route_target_rank=args.route_target_rank,
        phase_name="correctness",
        torch_profile_iters=0,
        torch_profile_warmup=0,
        torch_profile_output=None,
        torch_profile_top_kernels=0,
        torch_profile_chrome_trace=False,
    )


def tokens_for_rank(args: argparse.Namespace, rank: int) -> int:
    if args.rank_tokens is None:
        return args.tokens
    return args.rank_tokens[rank]


def tokens_across_dp(args: argparse.Namespace) -> list[int]:
    if args.rank_tokens is None:
        return [args.tokens for _ in range(args.world_size)]
    return list(args.rank_tokens)


def configure_common_env(args: argparse.Namespace) -> None:
    os.environ["NCCL_P2P_DISABLE"] = args.nccl_p2p_disable
    os.environ["VLLM_DEEPEP_HT_NUM_SMS"] = str(args.deepep_ht_num_sms)
    os.environ["VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS"] = "1"
    os.environ["VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS_MIN_TOKENS"] = "0"
    os.environ["VLLM_MOE_TRITON_EP_MASKED_ACTIVATION"] = "0"
    os.environ["VLLM_MOE_TRITON_W1_A100_TUNED_CONFIG"] = "0"
    os.environ["VLLM_MOE_TRITON_W2_A100_TUNED_CONFIG"] = "0"
    os.environ["VLLM_MOE_A100_BF16_SPECIALIZED_KERNELS"] = "0"
    os.environ["VLLM_DEEPEP_HT_DIRECT_ASSIGNMENT"] = "0"
    os.environ["VLLM_DEEPEP_HT_FIXED_CAPACITY_DISPATCH"] = "0"
    os.environ["VLLM_DEEPEP_HT_FIXED_CAPACITY_NUM_WORST_TOKENS"] = "0"


def clear_env_cache() -> None:
    envs.disable_envs_cache()
    cache_clear = getattr(envs.__getattr__, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()


def configure_setting(setting: CorrectnessSetting, args: argparse.Namespace) -> None:
    os.environ["VLLM_MOE_TRITON_W1_BLOCK_SIZE_M_OVERRIDE"] = str(setting.w1_block_m)
    os.environ["VLLM_MOE_TRITON_W2_BLOCK_SIZE_M_OVERRIDE"] = str(setting.w2_block_m)
    os.environ["VLLM_DEEPEP_HT_FIXED_CAPACITY_DISPATCH"] = (
        "1" if setting.fixed_capacity else "0"
    )
    os.environ["VLLM_DEEPEP_HT_FIXED_CAPACITY_NUM_WORST_TOKENS"] = (
        str(args.fixed_capacity_num_worst_tokens or sum(tokens_across_dp(args)))
        if setting.fixed_capacity
        else "0"
    )
    clear_env_cache()


def relative_l2(diff: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(reference.float())
    if denominator == 0:
        return float(torch.linalg.vector_norm(diff.float()).item())
    return float((torch.linalg.vector_norm(diff.float()) / denominator).item())


def compare_outputs(
    *,
    rank: int,
    tokens: int,
    reference_name: str,
    reference: torch.Tensor,
    setting_name: str,
    output: torch.Tensor,
    rtol: float,
    atol: float,
) -> CorrectnessMetric:
    diff = output.float() - reference.float()
    assert_close = True
    assert_close_error = None
    try:
        torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)
    except AssertionError as exc:
        assert_close = False
        assert_close_error = "\n".join(str(exc).splitlines()[:4])

    return CorrectnessMetric(
        rank=rank,
        tokens=tokens,
        setting=setting_name,
        reference=reference_name,
        max_abs_error=float(diff.abs().max().item()),
        mean_abs_error=float(diff.abs().mean().item()),
        relative_l2_error=relative_l2(diff, reference),
        rtol=rtol,
        atol=atol,
        assert_close=assert_close,
        assert_close_error=assert_close_error,
    )


def gather_objects(local: dict[str, Any], world_size: int) -> list[dict[str, Any]]:
    if world_size == 1:
        return [local]
    gathered: list[dict[str, Any] | None] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, local)
    return [item or {} for item in gathered]


def write_output(
    args: argparse.Namespace,
    metrics_by_rank: Iterable[dict[str, Any]],
) -> None:
    payload = {
        "backend": args.backend,
        "world_size": args.world_size,
        "tokens": args.tokens,
        "rank_tokens": args.rank_tokens,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "weight_seed": args.seed,
        "input_seed_base": args.input_seed_base,
        "rank_distinct_inputs": args.rank_distinct_inputs,
        "route_target_rank": args.route_target_rank,
        "fixed_capacity_num_worst_tokens": args.fixed_capacity_num_worst_tokens,
        "rtol": args.rtol,
        "atol": args.atol,
        "settings": [asdict(setting) for setting in SETTINGS],
        "metrics": [
            metric
            for rank_payload in metrics_by_rank
            for metric in rank_payload["metrics"]
        ],
        "assignment_stats": [
            stat
            for rank_payload in metrics_by_rank
            for stat in rank_payload["assignment_stats"]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


def run_worker(
    local_rank: int,
    args: argparse.Namespace,
    vllm_config: Any,
) -> None:
    rank = local_rank
    bench.init_rank(vllm_config, rank=rank, local_rank=local_rank)
    device = torch.device("cuda", local_rank)

    try:
        if not is_workspace_manager_initialized():
            init_workspace_manager(device)

        bench_args = benchmark_args(args)
        layer, hidden_states, router_logits = bench.make_layer_and_inputs(
            bench_args, rank, vllm_config
        )

        def forward_once():
            with bench.make_forward_context(bench_args, vllm_config, device):
                return layer(hidden_states, router_logits)

        outputs: dict[str, torch.Tensor] = {}
        assignment_stats: list[dict[str, Any]] = []
        for setting in SETTINGS:
            configure_setting(setting, args)
            output = forward_once()
            torch.accelerator.synchronize()
            outputs[setting.name] = output.detach().clone()
            stats_by_rank = bench.collect_ep_stats(bench_args, forward_once)
            assignment_stats.append(
                {
                    "rank": rank,
                    "setting": setting.name,
                    "w1_block_m": setting.w1_block_m,
                    "w2_block_m": setting.w2_block_m,
                    "fixed_capacity": setting.fixed_capacity,
                    "local_stats": stats_by_rank[rank],
                }
            )

        reference_name = SETTINGS[0].name
        reference = outputs[reference_name]
        metrics = [
            asdict(
                compare_outputs(
                    rank=rank,
                    tokens=tokens_for_rank(args, rank),
                    reference_name=reference_name,
                    reference=reference,
                    setting_name=setting.name,
                    output=outputs[setting.name],
                    rtol=args.rtol,
                    atol=args.atol,
                )
            )
            for setting in SETTINGS[1:]
        ]
        gathered = gather_objects(
            {"metrics": metrics, "assignment_stats": assignment_stats},
            args.world_size,
        )
        torch.distributed.barrier()
        if rank == 0:
            write_output(args, gathered)
    finally:
        torch.accelerator.synchronize()
        cleanup_dist_env_and_memory()


def main() -> None:
    args = parse_args()
    if args.rank_tokens is not None and len(args.rank_tokens) != args.world_size:
        raise ValueError("--rank-tokens must provide one value per rank")
    configure_common_env(args)
    if args.num_experts % args.world_size != 0:
        raise ValueError("--num-experts must be divisible by --world-size")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.device_count() < args.world_size:
        raise RuntimeError(
            f"Need {args.world_size} CUDA devices, found {torch.cuda.device_count()}"
        )

    bench_args = benchmark_args(args)
    vllm_config = bench.make_vllm_config(bench_args)
    if args.world_size == 1:
        run_worker(0, args, vllm_config)
    else:
        spawn(run_worker, args=(args, vllm_config), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
