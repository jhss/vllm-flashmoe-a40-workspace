#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize DeepEP HT paired ignore-invalid benchmark CSV files."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DEFAULT_CSV = "deepep_ht_paired_ignore_20260620_raw.csv"
SETTING_ORDER = (
    "original",
    "filtering",
    "final_both_64",
    "baseline",
    "ignore_off",
    "global_ignore",
    "local_id_ignore",
    "block_default",
    "block_w1_32",
    "block_w1_64",
    "block_w1_128",
    "block_w2_32",
    "block_w2_64",
    "block_w2_128",
    "block_both_32",
    "block_both_64",
    "block_fixed_both_64",
    "block_both_128",
)
INT_FIELDS = {
    "cycle_order",
    "threshold",
    "input_seed_group",
    "w1_block_m",
    "w2_block_m",
    "cycle",
    "rank",
    "world_size",
    "tokens",
    "hidden_size",
    "intermediate_size",
    "num_experts",
    "local_experts",
    "top_k",
    "warmup",
    "iters",
    "weight_seed",
    "input_seed_base",
    "input_seed_rank0",
    "input_seed_rank1",
    "input_tokens",
    "received_tokens_rank0",
    "received_tokens_rank1",
    "ep_ignore_num_tokens_rank0",
    "ep_ignore_num_tokens_rank1",
    "valid_route_pairs_rank0",
    "valid_route_pairs_rank1",
    "invalid_route_pairs_rank0",
    "invalid_route_pairs_rank1",
    "expert_tokens_min",
    "expert_tokens_max",
    "expert_tokens_zero",
}
FLOAT_FIELDS = {
    "full_forward_us",
    "rank0_forward_us",
    "rank1_forward_us",
    "critical_path_us",
    "topk_us",
    "dispatch_us",
    "combine_us",
    "expert_tokens_mean",
    "expert_tokens_cv",
}
BOOL_FIELDS = {
    "rank_distinct_inputs",
    "ep_ignore_enabled_rank0",
    "ep_ignore_enabled_rank1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name(DEFAULT_CSV),
    )
    parser.add_argument(
        "--outlier-threshold-us",
        type=float,
        default=0.0,
        help="Print paired rows whose delta is above this threshold.",
    )
    return parser.parse_args()


def median(values: Iterable[float]) -> float:
    return statistics.median(list(values))


def iqr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return q3 - q1


def parse_value(name: str, value: str) -> Any:
    if value == "":
        return None
    if name in INT_FIELDS:
        return int(value)
    if name in FLOAT_FIELDS:
        return float(value)
    if name in BOOL_FIELDS:
        return value == "True"
    return value


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append({name: parse_value(name, value) for name, value in row.items()})
    return rows


def ordered_settings(rows: list[dict[str, Any]]) -> list[str]:
    present = {str(row["setting"]) for row in rows}
    ordered = [setting for setting in SETTING_ORDER if setting in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def baseline_setting(rows: list[dict[str, Any]]) -> str:
    settings = ordered_settings(rows)
    if "baseline" in settings:
        return "baseline"
    if "ignore_off" in settings:
        return "ignore_off"
    if "original" in settings:
        return "original"
    if "block_default" in settings:
        return "block_default"
    return settings[0]


def group_columns(rows: list[dict[str, Any]]) -> list[str]:
    columns = ["tokens"]
    if any("threshold" in row for row in rows):
        columns.insert(0, "threshold")
    return columns


def pair_columns(rows: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    if any("threshold" in row for row in rows):
        columns.append("threshold")
    if any("input_seed_group" in row for row in rows):
        columns.append("input_seed_group")
    columns.extend(["cycle", "tokens"])
    return columns


def row_key(row: dict[str, Any], columns: list[str]) -> tuple[Any, ...]:
    return tuple(row.get(column) for column in columns)


def label_from_key(columns: list[str], key: tuple[Any, ...]) -> str:
    return " ".join(f"{column}={value}" for column, value in zip(columns, key))


def grouped_rows(
    rows: list[dict[str, Any]],
    columns: list[str],
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row_key(row, columns), []).append(row)
    return grouped


def validate_rows(rows: list[dict[str, Any]]) -> None:
    settings = ordered_settings(rows)
    baseline = baseline_setting(rows)
    pair_cols = pair_columns(rows)
    seen = {(row_key(row, pair_cols), str(row["setting"])) for row in rows}
    pair_keys = sorted({row_key(row, pair_cols) for row in rows})
    missing = [
        (pair_key, setting)
        for pair_key in pair_keys
        for setting in settings
        if (pair_key, setting) not in seen
    ]
    print(f"rows={len(rows)}")
    print(f"settings={settings}")
    print(f"baseline={baseline}")
    print(f"pair_key={pair_cols}")
    for name in ("threshold", "input_seed_group", "cycle", "tokens"):
        values = sorted({row.get(name) for row in rows if name in row})
        if values:
            print(f"{name}_values={values}")
    print(f"missing={missing}")
    print()


def critical_rank(row: dict[str, Any]) -> str:
    rank0 = row.get("rank0_forward_us")
    rank1 = row.get("rank1_forward_us")
    if rank0 is None or rank1 is None:
        return "rank0"
    if rank0 == rank1:
        return "tie"
    return "rank0" if rank0 > rank1 else "rank1"


def print_absolute_table(rows: list[dict[str, Any]]) -> None:
    settings = ordered_settings(rows)
    group_cols = group_columns(rows)
    print("## Critical Path Absolute")
    header = "| " + " | ".join(group_cols) + " | setting | median | IQR | min | max |"
    print(header)
    print("|" + "---|" * (len(group_cols) + 5))
    for group_key, subset in sorted(grouped_rows(rows, group_cols).items()):
        for setting in settings:
            vals = [
                float(row["critical_path_us"])
                for row in subset
                if row["setting"] == setting
            ]
            if not vals:
                continue
            cells = [str(value) for value in group_key]
            cells.extend(
                [
                    setting,
                    f"{median(vals):.1f}",
                    f"{iqr(vals):.1f}",
                    f"{min(vals):.1f}",
                    f"{max(vals):.1f}",
                ]
            )
            print("| " + " | ".join(cells) + " |")
    print()


def paired_deltas(
    rows: list[dict[str, Any]],
    target: str,
    group_key: tuple[Any, ...],
    group_cols: list[str],
) -> list[tuple[dict[str, Any], dict[str, Any], float, float]]:
    baseline = baseline_setting(rows)
    pair_cols = pair_columns(rows)
    filtered = [
        row
        for row in rows
        if row_key(row, group_cols) == group_key
        and row["setting"] in (baseline, target)
    ]
    by_key = {(row_key(row, pair_cols), str(row["setting"])): row for row in filtered}
    pair_keys = sorted({row_key(row, pair_cols) for row in filtered})
    deltas = []
    for pair_key in pair_keys:
        base_row = by_key.get((pair_key, baseline))
        target_row = by_key.get((pair_key, target))
        if base_row is None or target_row is None:
            continue
        base = float(base_row["critical_path_us"])
        target_value = float(target_row["critical_path_us"])
        delta = target_value - base
        pct = delta / base * 100.0
        deltas.append((base_row, target_row, delta, pct))
    return deltas


def print_paired_table(rows: list[dict[str, Any]]) -> None:
    baseline = baseline_setting(rows)
    targets = [setting for setting in ordered_settings(rows) if setting != baseline]
    group_cols = group_columns(rows)
    print("## Paired Delta")
    print(
        "| "
        + " | ".join(group_cols)
        + " | setting | min recv median | critical recv median | "
        "median delta | delta/median baseline | median pair pct | "
        "IQR | min | max | wins |"
    )
    print("|" + "---|" * (len(group_cols) + 10))
    for group_key in sorted(grouped_rows(rows, group_cols)):
        base_vals = [
            float(row["critical_path_us"])
            for row in rows
            if row_key(row, group_cols) == group_key and row["setting"] == baseline
        ]
        if not base_vals:
            continue
        base_median = median(base_vals)
        for target in targets:
            entries = paired_deltas(rows, target, group_key, group_cols)
            if not entries:
                continue
            deltas = [entry[2] for entry in entries]
            pcts = [entry[3] for entry in entries]
            min_recv = [
                min(
                    int(entry[1]["received_tokens_rank0"]),
                    int(entry[1]["received_tokens_rank1"]),
                )
                for entry in entries
            ]
            critical_recv = []
            for _, target_row, _, _ in entries:
                if critical_rank(target_row) == "rank0":
                    critical_recv.append(int(target_row["received_tokens_rank0"]))
                else:
                    critical_recv.append(int(target_row["received_tokens_rank1"]))
            delta_median = median(deltas)
            cells = [str(value) for value in group_key]
            cells.extend(
                [
                    target,
                    f"{median(min_recv):.0f}",
                    f"{median(critical_recv):.0f}",
                    f"{delta_median:+.1f}",
                    f"{delta_median / base_median * 100.0:+.2f}%",
                    f"{median(pcts):+.2f}%",
                    f"{iqr(deltas):.1f}",
                    f"{min(deltas):+.1f}",
                    f"{max(deltas):+.1f}",
                    f"{sum(delta < 0 for delta in deltas)}/{len(deltas)}",
                ]
            )
            print("| " + " | ".join(cells) + " |")
    print()


def print_seed_table(rows: list[dict[str, Any]]) -> None:
    if not any("input_seed_group" in row for row in rows):
        return
    baseline = baseline_setting(rows)
    targets = [setting for setting in ordered_settings(rows) if setting != baseline]
    group_cols = group_columns(rows)
    print("## Seed-Level Paired Median")
    print(
        "| "
        + " | ".join(group_cols)
        + " | input_seed_group | setting | median delta | wins |"
    )
    print("|" + "---|" * (len(group_cols) + 4))
    seed_groups = grouped_rows(rows, group_cols + ["input_seed_group"])
    for key in sorted(seed_groups):
        group_key = key[: len(group_cols)]
        seed = key[-1]
        for target in targets:
            entries = paired_deltas(rows, target, group_key, group_cols)
            entries = [
                entry for entry in entries if entry[0].get("input_seed_group") == seed
            ]
            if not entries:
                continue
            deltas = [entry[2] for entry in entries]
            cells = [str(value) for value in group_key]
            cells.extend(
                [
                    str(seed),
                    target,
                    f"{median(deltas):+.1f}",
                    f"{sum(delta < 0 for delta in deltas)}/{len(deltas)}",
                ]
            )
            print("| " + " | ".join(cells) + " |")
    print()


def print_ignore_table(rows: list[dict[str, Any]]) -> None:
    settings = ordered_settings(rows)
    group_cols = group_columns(rows)
    print("## Rank Activation")
    print(
        "| "
        + " | ".join(group_cols)
        + " | setting | rank0 true/total | rank1 true/total | "
        "num_tokens r0/r1 | recv r0/r1 | critical r0/r1/tie |"
    )
    print("|" + "---|" * (len(group_cols) + 6))
    for group_key, subset in sorted(grouped_rows(rows, group_cols).items()):
        for setting in settings:
            setting_rows = [row for row in subset if row["setting"] == setting]
            if not setting_rows:
                continue
            rank0 = Counter(row.get("ep_ignore_enabled_rank0") for row in setting_rows)
            rank1 = Counter(row.get("ep_ignore_enabled_rank1") for row in setting_rows)
            num0 = [row.get("ep_ignore_num_tokens_rank0") for row in setting_rows]
            num1 = [row.get("ep_ignore_num_tokens_rank1") for row in setting_rows]
            recv0 = [row.get("received_tokens_rank0") for row in setting_rows]
            recv1 = [row.get("received_tokens_rank1") for row in setting_rows]
            crit = Counter(critical_rank(row) for row in setting_rows)
            cells = [str(value) for value in group_key]
            cells.extend(
                [
                    setting,
                    f"{rank0.get(True, 0)}/{len(setting_rows)}",
                    f"{rank1.get(True, 0)}/{len(setting_rows)}",
                    f"{median(num0):.0f}/{median(num1):.0f}",
                    f"{median(recv0):.0f}/{median(recv1):.0f}",
                    f"{crit.get('rank0', 0)}/"
                    f"{crit.get('rank1', 0)}/{crit.get('tie', 0)}",
                ]
            )
            print("| " + " | ".join(cells) + " |")
    print()


def print_route_table(rows: list[dict[str, Any]]) -> None:
    def format_pair_median(
        left_values: list[float | None],
        right_values: list[float | None],
    ) -> str:
        left = [value for value in left_values if value is not None]
        right = [value for value in right_values if value is not None]
        if not left or not right:
            return "n/a"
        return f"{median(left):.0f}/{median(right):.0f}"

    baseline = baseline_setting(rows)
    group_cols = group_columns(rows)
    print("## Route Stats")
    print(
        "| " + " | ".join(group_cols) + " | valid pairs r0/r1 | invalid pairs r0/r1 |"
    )
    print("|" + "---|" * (len(group_cols) + 2))
    for group_key, subset in sorted(grouped_rows(rows, group_cols).items()):
        baseline_rows = [row for row in subset if row["setting"] == baseline]
        if not baseline_rows:
            continue
        valid0 = [row["valid_route_pairs_rank0"] for row in baseline_rows]
        valid1 = [row["valid_route_pairs_rank1"] for row in baseline_rows]
        invalid0 = [row["invalid_route_pairs_rank0"] for row in baseline_rows]
        invalid1 = [row["invalid_route_pairs_rank1"] for row in baseline_rows]
        cells = [str(value) for value in group_key]
        cells.extend(
            [
                format_pair_median(valid0, valid1),
                format_pair_median(invalid0, invalid1),
            ]
        )
        print("| " + " | ".join(cells) + " |")
    print()


def print_outliers(rows: list[dict[str, Any]], threshold_us: float) -> None:
    baseline = baseline_setting(rows)
    targets = [setting for setting in ordered_settings(rows) if setting != baseline]
    group_cols = group_columns(rows)
    print("## Positive Delta Outliers")
    print(
        "| "
        "group | input_seed_group | cycle | setting | delta | base | target | "
        "recv r0/r1 | active r0/r1 |"
    )
    print("|---|---:|---:|---|---:|---:|---:|---:|---:|")
    any_outlier = False
    for group_key in sorted(grouped_rows(rows, group_cols)):
        for target in targets:
            for base_row, target_row, delta, _ in paired_deltas(
                rows, target, group_key, group_cols
            ):
                if delta <= threshold_us:
                    continue
                any_outlier = True
                print(
                    "| "
                    f"{label_from_key(group_cols, group_key)} | "
                    f"{base_row.get('input_seed_group', '')} | "
                    f"{base_row.get('cycle', '')} | {target} | "
                    f"{delta:+.1f} | "
                    f"{float(base_row['critical_path_us']):.1f} | "
                    f"{float(target_row['critical_path_us']):.1f} | "
                    f"{target_row.get('received_tokens_rank0')}/"
                    f"{target_row.get('received_tokens_rank1')} | "
                    f"{target_row.get('ep_ignore_enabled_rank0')}/"
                    f"{target_row.get('ep_ignore_enabled_rank1')} |"
                )
    if not any_outlier:
        print("| none |  |  |  |  |  |  |  |  |")
    print()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.csv_path)
    validate_rows(rows)
    print_absolute_table(rows)
    print_paired_table(rows)
    print_seed_table(rows)
    print_ignore_table(rows)
    print_route_table(rows)
    print_outliers(rows, args.outlier_threshold_us)


if __name__ == "__main__":
    main()
