#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W1/W13-only Triton MoE kernel sweep for A100/SM80-style shapes."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

if "--cache-dir" in sys.argv:
    cache_dir_idx = sys.argv.index("--cache-dir") + 1
    if cache_dir_idx < len(sys.argv):
        os.environ["TRITON_CACHE_DIR"] = sys.argv[cache_dir_idx]
else:
    os.environ.setdefault(
        "TRITON_CACHE_DIR",
        tempfile.mkdtemp(prefix="vllm_moe_w1_triton_cache_"),
    )
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

import torch

from vllm.model_executor.layers.fused_moe.config import _get_config_dtype_str
from vllm.model_executor.layers.fused_moe.fused_moe import (
    _prepare_expert_assignment,
    invoke_fused_moe_triton_kernel,
    try_get_optimal_moe_config,
)
from vllm.triton_utils import tl


@dataclass(frozen=True)
class Candidate:
    name: str
    block_m: int
    block_n: int
    block_k: int
    group_m: int
    warps: int
    stages: int

    def config(self) -> dict[str, int]:
        return {
            "BLOCK_SIZE_M": self.block_m,
            "BLOCK_SIZE_N": self.block_n,
            "BLOCK_SIZE_K": self.block_k,
            "GROUP_SIZE_M": self.group_m,
            "num_warps": self.warps,
            "num_stages": self.stages,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=768)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resource-usage", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-candidates", type=int)
    return parser.parse_args()


def candidate_grid(base_config: dict[str, int]) -> list[Candidate]:
    base = Candidate(
        "base",
        base_config["BLOCK_SIZE_M"],
        base_config["BLOCK_SIZE_N"],
        base_config["BLOCK_SIZE_K"],
        base_config["GROUP_SIZE_M"],
        base_config.get("num_warps", 4),
        base_config.get("num_stages", 3),
    )
    candidates = [base]
    for block_m in (32, 64, 128):
        for block_n in (64, 128, 256):
            for block_k in (32, 64, 128):
                for warps in (4, 8):
                    for stages in (2, 3, 4):
                        candidates.append(
                            Candidate(
                                (
                                    f"m{block_m}_n{block_n}_k{block_k}"
                                    f"_w{warps}_s{stages}"
                                ),
                                block_m,
                                block_n,
                                block_k,
                                1,
                                warps,
                                stages,
                            )
                        )

    seen: set[tuple[int, int, int, int, int, int]] = set()
    unique = []
    for candidate in candidates:
        key = (
            candidate.block_m,
            candidate.block_n,
            candidate.block_k,
            candidate.group_m,
            candidate.warps,
            candidate.stages,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def make_inputs(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    hidden_states = torch.randn(
        args.tokens,
        args.hidden_size,
        device=device,
        dtype=dtype,
    ) / 10
    w1 = torch.randn(
        args.num_experts,
        2 * args.intermediate_size,
        args.hidden_size,
        device=device,
        dtype=dtype,
    ) / 10
    router_logits = torch.randn(
        args.tokens,
        args.num_experts,
        device=device,
        dtype=torch.float32,
    )
    _, topk_ids = torch.topk(
        router_logits,
        k=args.top_k,
        dim=-1,
        sorted=False,
    )
    topk_ids = topk_ids.to(torch.int32).contiguous()
    return hidden_states, w1, topk_ids


def make_assignment(
    args: argparse.Namespace,
    config: dict[str, int],
    topk_ids: torch.Tensor,
):
    return _prepare_expert_assignment(
        topk_ids,
        config,
        args.tokens,
        args.top_k,
        args.num_experts,
        None,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        block_shape=None,
    )


def launch_w1(
    args: argparse.Namespace,
    config: dict[str, int],
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    out: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
) -> None:
    invoke_fused_moe_triton_kernel(
        hidden_states,
        w1,
        out,
        None,
        None,
        None,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        False,
        args.top_k,
        config,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=None,
        B_bias=None,
    )


def time_cuda(fn, warmup: int, iters: int) -> float:
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
    return start.elapsed_time(end) * 1000.0 / iters


def cubins(cache_dir: Path) -> set[Path]:
    if not cache_dir.exists():
        return set()
    return set(cache_dir.rglob("fused_moe_kernel.cubin"))


def parse_resource_usage(cubin: Path) -> tuple[int | None, int | None, int | None]:
    cuobjdump = Path("/usr/local/cuda/bin/cuobjdump")
    if not cuobjdump.exists():
        return None, None, None
    try:
        proc = subprocess.run(
            [str(cuobjdump), "--dump-resource-usage", str(cubin)],
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return None, None, None
    reg = shared = local = None
    for item in proc.stdout.replace("\n", " ").split():
        if item.startswith("REG:"):
            reg = int(item.split(":", 1)[1])
        elif item.startswith("SHARED:"):
            shared = int(item.split(":", 1)[1])
        elif item.startswith("LOCAL:"):
            local = int(item.split(":", 1)[1])
    return reg, shared, local


def main() -> None:
    args = parse_args()
    if args.cache_dir is not None:
        os.environ["TRITON_CACHE_DIR"] = str(args.cache_dir)
    cache_dir = Path(os.environ["TRITON_CACHE_DIR"])

    torch.cuda.set_device(0)
    hidden_states, w1, topk_ids = make_inputs(args)
    w2_shape = (args.num_experts, args.hidden_size, args.intermediate_size)
    dtype_name = _get_config_dtype_str(
        use_fp8_w8a8=False,
        use_fp8_w8a16=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        ocp_mx_scheme=None,
        dtype=torch.bfloat16,
    )
    base_config = try_get_optimal_moe_config(
        tuple(w1.shape),
        w2_shape,
        args.top_k,
        dtype_name,
        args.tokens,
        block_shape=None,
    )

    base_sorted, base_experts, base_padded = make_assignment(
        args,
        base_config,
        topk_ids,
    )
    ref = torch.empty(
        args.tokens,
        args.top_k,
        2 * args.intermediate_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    launch_w1(
        args,
        base_config,
        hidden_states,
        w1,
        ref,
        base_sorted,
        base_experts,
        base_padded,
    )
    torch.cuda.synchronize()

    rows = []
    candidates = candidate_grid(base_config)
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]

    for candidate in candidates:
        config = candidate.config()
        try:
            sorted_token_ids, expert_ids, num_tokens_post_padded = make_assignment(
                args,
                config,
                topk_ids,
            )
            out = torch.empty_like(ref)
            before = cubins(cache_dir) if args.resource_usage else set()

            def once():
                launch_w1(
                    args,
                    config,
                    hidden_states,
                    w1,
                    out,
                    sorted_token_ids,
                    expert_ids,
                    num_tokens_post_padded,
                )

            once()
            torch.cuda.synchronize()
            diff = (ref.float() - out.float()).abs()
            max_abs = float(diff.max().item())
            mean_abs = float(diff.mean().item())
            torch.testing.assert_close(ref, out, rtol=3e-2, atol=8e-2)
            elapsed_us = time_cuda(once, args.warmup, args.iters)
            after = cubins(cache_dir) if args.resource_usage else set()
            new_cubins = sorted(after - before)
            reg = shared = local = None
            if args.resource_usage:
                if new_cubins:
                    cubin = new_cubins[-1]
                else:
                    existing_cubins = sorted(after, key=lambda path: path.stat().st_mtime)
                    cubin = existing_cubins[-1] if existing_cubins else None
                if cubin is not None:
                    reg, shared, local = parse_resource_usage(cubin)
            status = "ok"
        except Exception as exc:
            elapsed_us = float("nan")
            max_abs = float("nan")
            mean_abs = float("nan")
            reg = shared = local = None
            status = f"error:{type(exc).__name__}:{exc}"

        rows.append(
            {
                "name": candidate.name,
                "tokens": args.tokens,
                "hidden_size": args.hidden_size,
                "intermediate_size": args.intermediate_size,
                "num_experts": args.num_experts,
                "top_k": args.top_k,
                "block_m": candidate.block_m,
                "block_n": candidate.block_n,
                "block_k": candidate.block_k,
                "group_m": candidate.group_m,
                "warps": candidate.warps,
                "stages": candidate.stages,
                "latency_us": elapsed_us,
                "max_abs": max_abs,
                "mean_abs": mean_abs,
                "reg": reg,
                "shared": shared,
                "local": local,
                "status": status,
            }
        )

    rows.sort(
        key=lambda row: (
            row["status"] != "ok",
            float("inf") if row["latency_us"] != row["latency_us"] else row["latency_us"],
        )
    )

    fieldnames = list(rows[0].keys())
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    if args.csv:
        writer = csv.DictWriter(os.sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    else:
        for row in rows[:20]:
            print(row)


if __name__ == "__main__":
    main()
