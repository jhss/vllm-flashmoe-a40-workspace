#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a small matrix of vLLM MoE EP benchmarks.

This wrapper intentionally launches benchmark_moe_ep_a40.py in subprocesses so
each row gets a fresh torch.distributed/vLLM process-group state.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = REPO_ROOT / "benchmarks" / "kernels" / "benchmark_moe_ep_a40.py"


@dataclass(frozen=True)
class Shape:
    name: str
    num_experts: int
    hidden_size: int
    intermediate_size: int
    top_k: int
    tokens: tuple[int, ...]


PRESETS: dict[str, tuple[Shape, ...]] = {
    "qwen3": (
        Shape(
            name="qwen3_30b_a3b_bf16",
            num_experts=128,
            hidden_size=2048,
            intermediate_size=768,
            top_k=8,
            tokens=(32, 128, 512),
        ),
    ),
    "a40_quick": (
        Shape(
            name="qwen3_30b_a3b_bf16",
            num_experts=128,
            hidden_size=2048,
            intermediate_size=768,
            top_k=8,
            tokens=(32, 128, 512),
        ),
        Shape(
            name="qwen2_moe_57b_bf16",
            num_experts=60,
            hidden_size=2048,
            intermediate_size=1408,
            top_k=4,
            tokens=(32, 128),
        ),
    ),
    "a40_extended": (
        Shape(
            name="qwen3_30b_a3b_bf16",
            num_experts=128,
            hidden_size=2048,
            intermediate_size=768,
            top_k=8,
            tokens=(32, 128, 512),
        ),
        Shape(
            name="qwen2_moe_57b_bf16",
            num_experts=60,
            hidden_size=2048,
            intermediate_size=1408,
            top_k=4,
            tokens=(32, 128, 512),
        ),
        Shape(
            name="mixtral_8x7b_bf16",
            num_experts=8,
            hidden_size=4096,
            intermediate_size=7168,
            top_k=2,
            tokens=(16, 64, 128),
        ),
    ),
}


BASE_FIELDNAMES = [
    "preset",
    "shape",
    "rank",
    "world_size",
    "backend",
    "tokens",
    "hidden_size",
    "intermediate_size",
    "num_experts",
    "local_experts",
    "top_k",
    "dtype",
    "warmup",
    "iters",
    "full_forward_us",
    "topk_us",
    "dispatch_us",
    "combine_us",
    "comm_us",
    "full_forward_ms",
    "comm_ms",
    "speedup_vs_world1",
    "elapsed_s",
    "returncode",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="qwen3")
    parser.add_argument("--world-sizes", default="1,2")
    parser.add_argument("--backend", default="allgather_reducescatter")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def parse_world_sizes(value: str) -> list[int]:
    sizes = [int(v.strip()) for v in value.split(",") if v.strip()]
    unsupported = [v for v in sizes if v not in (1, 2)]
    if unsupported:
        raise ValueError(f"Only world sizes 1 and 2 are supported: {unsupported}")
    return sizes


def default_output_path(preset: str) -> Path:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "benchmarks" / "results" / f"{preset}_moe_ep_{stamp}.csv"


def command_for(
    shape: Shape,
    tokens: int,
    world_size: int,
    args: argparse.Namespace,
) -> list[str]:
    return [
        sys.executable,
        str(BENCHMARK),
        "--world-size",
        str(world_size),
        "--backend",
        args.backend,
        "--tokens",
        str(tokens),
        "--hidden-size",
        str(shape.hidden_size),
        "--intermediate-size",
        str(shape.intermediate_size),
        "--num-experts",
        str(shape.num_experts),
        "--top-k",
        str(shape.top_k),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--seed",
        str(args.seed),
        "--csv",
    ]


def parse_benchmark_csv(stdout: str) -> dict[str, str]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        if line.startswith("rank,world_size,backend,"):
            try:
                row_line = lines[idx + 1]
            except IndexError as exc:
                raise RuntimeError("CSV header found without a data row") from exc
            return next(csv.DictReader([line, row_line]))
    raise RuntimeError("benchmark CSV output was not found")


def as_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    return float(value)


def finalize_rows(rows: list[dict[str, object]]) -> None:
    baseline: dict[tuple[str, int], float] = {}
    for row in rows:
        if int(row["world_size"]) == 1 and int(row["returncode"]) == 0:
            baseline[(str(row["shape"]), int(row["tokens"]))] = float(
                row["full_forward_us"]
            )

    for row in rows:
        dispatch_us = row.get("dispatch_us")
        combine_us = row.get("combine_us")
        comm_us = None
        if dispatch_us not in (None, "") and combine_us not in (None, ""):
            comm_us = float(dispatch_us) + float(combine_us)
        row["comm_us"] = "" if comm_us is None else comm_us
        row["full_forward_ms"] = float(row["full_forward_us"]) / 1000.0
        row["comm_ms"] = "" if comm_us is None else comm_us / 1000.0

        base = baseline.get((str(row["shape"]), int(row["tokens"])))
        if base is None or int(row["world_size"]) == 1:
            row["speedup_vs_world1"] = ""
        else:
            row["speedup_vs_world1"] = base / float(row["full_forward_us"])


def write_topology(output: Path) -> None:
    topo_path = output.with_suffix(".topology.txt")
    try:
        topo = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        topo_path.write_text(topo.stdout + "\n" + query.stdout)
    except OSError:
        pass


def main() -> None:
    args = parse_args()
    world_sizes = parse_world_sizes(args.world_sizes)
    output = args.output or default_output_path(args.preset)
    output.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

    rows: list[dict[str, object]] = []
    for shape in PRESETS[args.preset]:
        for tokens in shape.tokens:
            for world_size in world_sizes:
                if shape.num_experts % world_size != 0:
                    print(
                        f"skip {shape.name} tokens={tokens} world={world_size}: "
                        "num_experts not divisible",
                        flush=True,
                    )
                    continue

                cmd = command_for(shape, tokens, world_size, args)
                print("run " + " ".join(cmd), flush=True)
                if args.dry_run:
                    continue

                start = time.perf_counter()
                completed = subprocess.run(
                    cmd,
                    cwd=REPO_ROOT,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                elapsed_s = time.perf_counter() - start
                if completed.returncode != 0:
                    print(completed.stdout[-4000:], flush=True)
                    row: dict[str, object] = {
                        "preset": args.preset,
                        "shape": shape.name,
                        "rank": "",
                        "world_size": world_size,
                        "backend": args.backend,
                        "tokens": tokens,
                        "hidden_size": shape.hidden_size,
                        "intermediate_size": shape.intermediate_size,
                        "num_experts": shape.num_experts,
                        "local_experts": "",
                        "top_k": shape.top_k,
                        "dtype": "",
                        "warmup": args.warmup,
                        "iters": args.iters,
                        "full_forward_us": "",
                        "topk_us": "",
                        "dispatch_us": "",
                        "combine_us": "",
                        "comm_us": "",
                        "full_forward_ms": "",
                        "comm_ms": "",
                        "speedup_vs_world1": "",
                        "elapsed_s": elapsed_s,
                        "returncode": completed.returncode,
                    }
                    rows.append(row)
                    if args.fail_fast:
                        raise SystemExit(completed.returncode)
                    continue

                parsed = parse_benchmark_csv(completed.stdout)
                row = {
                    "preset": args.preset,
                    "shape": shape.name,
                    "elapsed_s": elapsed_s,
                    "returncode": completed.returncode,
                }
                row.update(parsed)
                for key in ("full_forward_us", "topk_us", "dispatch_us", "combine_us"):
                    value = as_float(row, key)
                    row[key] = "" if value is None else value
                rows.append(row)

    if not args.dry_run:
        finalize_rows(rows)
        with output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=BASE_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        write_topology(output)
        print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
