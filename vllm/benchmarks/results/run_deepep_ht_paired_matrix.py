#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run paired DeepEP HT MoE EP benchmark matrices.

This wrapper keeps the expensive process-per-run benchmark simple while adding
the experiment structure needed for paired analysis: seed groups, balanced
cycle ordering, and setting-specific environment variables.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BREAK_EVEN_TOKENS = [8, 16, 32, 48, 64, 96, 128]
DEFAULT_BLOCK_M_TOKENS = [256, 320, 384, 448]
DEFAULT_BLOCK_M_SCREENING_TOKENS = [320, 448]
DEFAULT_BLOCK_M_COMBINED_ABLATION_TOKENS = [320, 448]
DEFAULT_INPUT_SEEDS = [1007, 2007, 3007, 4007, 5007]
DEFAULT_SCREENING_INPUT_SEEDS = [1007, 2007, 3007]
DEFAULT_RESULT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Setting:
    name: str
    env: dict[str, str]
    threshold: int | None = None
    w1_block_m: int | None = None
    w2_block_m: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=[
            "break-even",
            "block-m-sweep",
            "block-m-screening",
            "block-m-combined-ablation",
        ],
        default="break-even",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--commands-log", type=Path)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--tokens", type=int, nargs="+")
    parser.add_argument("--input-seed-bases", type=int, nargs="+")
    parser.add_argument("--cycles", type=int)
    parser.add_argument("--thresholds", type=int, nargs="+", default=[0])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--world-size", type=int, default=2, choices=[1, 2])
    parser.add_argument("--backend", default="deepep_high_throughput")
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=768)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--weight-seed", type=int, default=7)
    parser.add_argument("--deepep-ht-num-sms", type=int, default=24)
    parser.add_argument("--nccl-p2p-disable", default="0")
    parser.add_argument(
        "--block-m-settings",
        nargs="+",
        default=None,
        help=(
            "Block-M sweep settings. Supported values: default, w1_32, "
            "w1_64, w1_128, w2_32, w2_64, w2_128, both_32, both_64, "
            "both_128, fixed_both_64."
        ),
    )
    parser.add_argument("--section-profile-iters", type=int, default=0)
    parser.add_argument("--section-profile-warmup", type=int, default=2)
    parser.add_argument("--section-profile-dir", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args()


def default_output(mode: str) -> Path:
    if mode == "break-even":
        name = "deepep_ht_break_even_sub254_20260621_raw.csv"
    elif mode == "block-m-screening":
        name = "deepep_ht_block_m_screening_20260621_raw.csv"
    elif mode == "block-m-combined-ablation":
        name = "deepep_ht_block_m_combined_ablation_20260621_raw.csv"
    else:
        name = "deepep_ht_block_m_sweep_20260621_raw.csv"
    return DEFAULT_RESULT_DIR / name


def default_tokens(mode: str) -> list[int]:
    if mode == "break-even":
        return DEFAULT_BREAK_EVEN_TOKENS
    if mode == "block-m-screening":
        return DEFAULT_BLOCK_M_SCREENING_TOKENS
    if mode == "block-m-combined-ablation":
        return DEFAULT_BLOCK_M_COMBINED_ABLATION_TOKENS
    return DEFAULT_BLOCK_M_TOKENS


def default_input_seeds(mode: str) -> list[int]:
    if mode in ("block-m-screening", "block-m-combined-ablation"):
        return DEFAULT_SCREENING_INPUT_SEEDS
    return DEFAULT_INPUT_SEEDS


def default_cycles(mode: str) -> int:
    if mode in ("block-m-screening", "block-m-combined-ablation"):
        return 3
    return 4


def default_block_m_setting_names(mode: str) -> list[str]:
    if mode == "block-m-screening":
        return ["default", "both_64", "both_32"]
    return ["default", "w1_64", "w2_64", "both_64", "both_32"]


def common_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "NCCL_P2P_DISABLE": args.nccl_p2p_disable,
            "VLLM_DEEPEP_HT_NUM_SMS": str(args.deepep_ht_num_sms),
            "VLLM_DEEPEP_HT_FIXED_CAPACITY_DISPATCH": "0",
            "VLLM_DEEPEP_HT_FIXED_CAPACITY_NUM_WORST_TOKENS": "0",
            "VLLM_MOE_TRITON_EP_MASKED_ACTIVATION": "0",
            "VLLM_MOE_TRITON_W1_BLOCK_SIZE_M_OVERRIDE": "0",
            "VLLM_MOE_TRITON_W2_BLOCK_SIZE_M_OVERRIDE": "0",
            "VLLM_MOE_TRITON_W1_A100_TUNED_CONFIG": "0",
            "VLLM_MOE_TRITON_W2_A100_TUNED_CONFIG": "0",
            "VLLM_MOE_A100_BF16_SPECIALIZED_KERNELS": "0",
            "VLLM_DEEPEP_HT_DIRECT_ASSIGNMENT": "0",
        }
    )
    return env


def break_even_settings(threshold: int) -> list[Setting]:
    return [
        Setting(
            name="baseline",
            threshold=threshold,
            env={
                "VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS": "0",
                "VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS_MIN_TOKENS": str(threshold),
            },
        ),
        Setting(
            name="global_ignore",
            threshold=threshold,
            env={
                "VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS": "1",
                "VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS_MIN_TOKENS": str(threshold),
            },
        ),
    ]


def parse_block_setting(name: str) -> tuple[int | None, int | None]:
    if name == "default":
        return None, None
    if name.startswith("both_"):
        block_m = int(name.removeprefix("both_"))
        return block_m, block_m
    if name.startswith("w1_"):
        return int(name.removeprefix("w1_")), None
    if name.startswith("w2_"):
        return None, int(name.removeprefix("w2_"))
    raise ValueError(f"Unsupported block-m setting: {name}")


def block_m_settings(names: list[str]) -> list[Setting]:
    settings = []
    for name in names:
        fixed_capacity = name.startswith("fixed_")
        parsed_name = name.removeprefix("fixed_") if fixed_capacity else name
        w1_block_m, w2_block_m = parse_block_setting(parsed_name)
        env = {
            "VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS": "1",
            "VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS_MIN_TOKENS": "0",
        }
        if fixed_capacity:
            env["VLLM_DEEPEP_HT_FIXED_CAPACITY_DISPATCH"] = "1"
        if w1_block_m is not None:
            env["VLLM_MOE_TRITON_W1_BLOCK_SIZE_M_OVERRIDE"] = str(w1_block_m)
        if w2_block_m is not None:
            env["VLLM_MOE_TRITON_W2_BLOCK_SIZE_M_OVERRIDE"] = str(w2_block_m)
        settings.append(
            Setting(
                name=f"block_{name}",
                env=env,
                threshold=0,
                w1_block_m=w1_block_m,
                w2_block_m=w2_block_m,
            )
        )
    return settings


def block_m_combined_ablation_settings() -> list[Setting]:
    return [
        Setting(
            name="original",
            env={
                "VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS": "0",
                "VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS_MIN_TOKENS": "0",
            },
            threshold=0,
        ),
        Setting(
            name="filtering",
            env={
                "VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS": "1",
                "VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS_MIN_TOKENS": "0",
            },
            threshold=0,
        ),
        Setting(
            name="final_both_64",
            env={
                "VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS": "1",
                "VLLM_MOE_TRITON_EP_IGNORE_INVALID_EXPERTS_MIN_TOKENS": "0",
                "VLLM_MOE_TRITON_W1_BLOCK_SIZE_M_OVERRIDE": "64",
                "VLLM_MOE_TRITON_W2_BLOCK_SIZE_M_OVERRIDE": "64",
            },
            threshold=0,
            w1_block_m=64,
            w2_block_m=64,
        ),
    ]


def cycle_settings(settings: list[Setting], cycle: int) -> list[Setting]:
    if len(settings) == 2:
        pattern = (
            (settings[0], settings[1]),
            (settings[1], settings[0]),
            (settings[1], settings[0]),
            (settings[0], settings[1]),
        )
        return list(pattern[(cycle - 1) % len(pattern)])
    shift = (cycle - 1) % len(settings)
    return list(settings[shift:] + settings[:shift])


def benchmark_command(args: argparse.Namespace, tokens: int, input_seed_base: int):
    cmd = [
        sys.executable,
        "benchmarks/kernels/benchmark_moe_ep_a40.py",
        "--world-size",
        str(args.world_size),
        "--backend",
        args.backend,
        "--tokens",
        str(tokens),
        "--hidden-size",
        str(args.hidden_size),
        "--intermediate-size",
        str(args.intermediate_size),
        "--num-experts",
        str(args.num_experts),
        "--top-k",
        str(args.top_k),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--seed",
        str(args.weight_seed),
        "--rank-distinct-inputs",
        "--input-seed-base",
        str(input_seed_base),
        "--csv",
    ]
    return cmd


def prepare_run_env(
    args: argparse.Namespace,
    common: dict[str, str],
    setting: Setting,
    tokens: int,
) -> dict[str, str]:
    env = dict(common)
    env.update(setting.env)
    if (
        env.get("VLLM_DEEPEP_HT_FIXED_CAPACITY_DISPATCH") == "1"
        and int(env.get("VLLM_DEEPEP_HT_FIXED_CAPACITY_NUM_WORST_TOKENS", "0")) <= 0
    ):
        env["VLLM_DEEPEP_HT_FIXED_CAPACITY_NUM_WORST_TOKENS"] = str(
            tokens * args.world_size
        )
    return env


def section_profile_path(
    args: argparse.Namespace,
    setting: Setting,
    tokens: int,
    input_seed_base: int,
    cycle: int,
) -> Path | None:
    if args.section_profile_iters <= 0:
        return None
    output_dir = args.section_profile_dir
    if output_dir is None:
        output_dir = args.output.with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / (
        f"{args.mode}_tokens{tokens}_seed{input_seed_base}_"
        f"cycle{cycle}_{setting.name}.json"
    )


def parse_benchmark_csv(stdout: str) -> dict[str, str]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        if line.startswith("rank,world_size,backend,"):
            if idx + 1 >= len(lines):
                raise RuntimeError("Benchmark CSV header was not followed by a row")
            reader = csv.DictReader([line, lines[idx + 1]])
            return next(reader)
    raise RuntimeError(f"Benchmark CSV output not found:\n{stdout}")


def write_row(
    output: Path,
    row: dict[str, str],
    *,
    append: bool,
    fieldnames: list[str],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not append or not output.exists() or output.stat().st_size == 0
    with output.open("a" if append else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_command_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(text.rstrip())
        f.write("\n")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    args.output = args.output or default_output(args.mode)
    args.commands_log = args.commands_log or args.output.with_name(
        args.output.stem.replace("_raw", "_commands") + ".log"
    )
    tokens_list = args.tokens
    if tokens_list is None:
        tokens_list = default_tokens(args.mode)
    input_seed_bases = args.input_seed_bases or default_input_seeds(args.mode)
    cycles = args.cycles or default_cycles(args.mode)
    block_m_setting_names = args.block_m_settings or default_block_m_setting_names(
        args.mode
    )

    if args.mode == "break-even":
        settings_groups = [
            break_even_settings(threshold) for threshold in args.thresholds
        ]
    elif args.mode == "block-m-combined-ablation":
        settings_groups = [block_m_combined_ablation_settings()]
    else:
        settings_groups = [block_m_settings(block_m_setting_names)]

    common = common_env(args)
    fieldnames: list[str] | None = None
    runs = 0
    append = args.append

    for settings in settings_groups:
        for input_seed_base in input_seed_bases:
            for cycle in range(1, cycles + 1):
                for tokens in tokens_list:
                    cycle_order = cycle_settings(settings, cycle)
                    for order_idx, setting in enumerate(cycle_order):
                        runs += 1
                        if args.limit and runs > args.limit:
                            return

                        env = prepare_run_env(args, common, setting, tokens)
                        cmd = benchmark_command(args, tokens, input_seed_base)
                        profile_path = section_profile_path(
                            args, setting, tokens, input_seed_base, cycle
                        )
                        if profile_path is not None:
                            cmd.extend(
                                [
                                    "--section-profile-iters",
                                    str(args.section_profile_iters),
                                    "--section-profile-warmup",
                                    str(args.section_profile_warmup),
                                    "--section-profile-output",
                                    str(profile_path),
                                ]
                            )

                        env_items = " ".join(
                            f"{key}={env[key]}"
                            for key in sorted(setting.env | common)
                            if key.startswith("VLLM_") or key == "NCCL_P2P_DISABLE"
                        )
                        command_text = f"{env_items} {' '.join(cmd)}"
                        print(
                            f"[{runs}] {setting.name} seed={input_seed_base} "
                            f"cycle={cycle} tokens={tokens}",
                            flush=True,
                        )
                        append_command_log(args.commands_log, command_text)
                        if args.print_only:
                            continue

                        completed = subprocess.run(
                            cmd,
                            cwd=repo_root,
                            env=env,
                            text=True,
                            capture_output=True,
                            check=False,
                        )
                        if completed.returncode != 0:
                            sys.stderr.write(completed.stdout)
                            sys.stderr.write(completed.stderr)
                            raise subprocess.CalledProcessError(
                                completed.returncode,
                                cmd,
                                output=completed.stdout,
                                stderr=completed.stderr,
                            )
                        benchmark_row = parse_benchmark_csv(completed.stdout)
                        extra = {
                            "setting": setting.name,
                            "threshold": (
                                ""
                                if setting.threshold is None
                                else str(setting.threshold)
                            ),
                            "input_seed_group": str(input_seed_base),
                            "cycle": str(cycle),
                            "cycle_order": str(order_idx),
                            "w1_block_m": (
                                ""
                                if setting.w1_block_m is None
                                else str(setting.w1_block_m)
                            ),
                            "w2_block_m": (
                                ""
                                if setting.w2_block_m is None
                                else str(setting.w2_block_m)
                            ),
                        }
                        row = extra | benchmark_row
                        if fieldnames is None:
                            fieldnames = list(row)
                        write_row(
                            args.output,
                            row,
                            append=append,
                            fieldnames=fieldnames,
                        )
                        append = True


if __name__ == "__main__":
    main()
