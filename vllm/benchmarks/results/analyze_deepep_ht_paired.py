#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize DeepEP HT paired ignore-invalid benchmark CSV."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter
from pathlib import Path


SETTINGS = ("baseline", "global_ignore", "local_id_ignore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name(
            "deepep_ht_paired_ignore_20260620_raw.csv"
        ),
    )
    return parser.parse_args()


def median(values: list[float]) -> float:
    return statistics.median(values)


def iqr(values: list[float]) -> float:
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return q3 - q1


def read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            parsed: dict[str, object] = dict(row)
            parsed["cycle"] = int(row["cycle"])
            parsed["tokens"] = int(row["tokens"])
            for name in (
                "critical_path_us",
                "rank0_forward_us",
                "rank1_forward_us",
            ):
                parsed[name] = float(row[name])
            for name in (
                "received_tokens_rank0",
                "received_tokens_rank1",
                "valid_route_pairs_rank0",
                "valid_route_pairs_rank1",
                "invalid_route_pairs_rank0",
                "invalid_route_pairs_rank1",
            ):
                parsed[name] = int(row[name]) if row[name] else None
            rows.append(parsed)
    return rows


def validate_rows(rows: list[dict[str, object]]) -> None:
    cycles = sorted({int(row["cycle"]) for row in rows})
    token_sizes = sorted({int(row["tokens"]) for row in rows})
    seen = {
        (int(row["cycle"]), int(row["tokens"]), str(row["setting"]))
        for row in rows
    }
    missing = [
        (cycle, tokens, setting)
        for cycle in cycles
        for tokens in token_sizes
        for setting in SETTINGS
        if (cycle, tokens, setting) not in seen
    ]
    print(f"rows={len(rows)}")
    print(f"cycles={cycles}")
    print(f"tokens={token_sizes}")
    print(f"missing={missing}")
    print()


def by_tokens_setting(
    rows: list[dict[str, object]],
    tokens: int,
    setting: str,
) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if int(row["tokens"]) == tokens and str(row["setting"]) == setting
    ]


def print_absolute_table(rows: list[dict[str, object]]) -> None:
    print("## Critical Path Absolute")
    print(
        "| tokens | baseline median us (IQR/min/max) | "
        "global-ignore median us (IQR/min/max) | "
        "local-ID-ignore median us (IQR/min/max) |"
    )
    print("|---:|---:|---:|---:|")
    for tokens in sorted({int(row["tokens"]) for row in rows}):
        parts = [str(tokens)]
        for setting in SETTINGS:
            subset = by_tokens_setting(rows, tokens, setting)
            vals = [float(row["critical_path_us"]) for row in subset]
            parts.append(
                f"{median(vals):.1f} ({iqr(vals):.1f}/"
                f"{min(vals):.1f}/{max(vals):.1f})"
            )
        print("| " + " | ".join(parts) + " |")
    print()


def print_paired_table(rows: list[dict[str, object]]) -> None:
    print("## Paired Delta")
    print(
        "| tokens | global-ignore - baseline | "
        "local-ID-ignore - baseline | local-ID - global |"
    )
    print("|---:|---:|---:|---:|")
    cycles = sorted({int(row["cycle"]) for row in rows})
    for tokens in sorted({int(row["tokens"]) for row in rows}):
        by_key = {
            (int(row["cycle"]), str(row["setting"])): float(
                row["critical_path_us"]
            )
            for row in rows
            if int(row["tokens"]) == tokens
        }
        base_median = median([by_key[(cycle, "baseline")] for cycle in cycles])
        parts = [str(tokens)]
        for setting in ("global_ignore", "local_id_ignore"):
            deltas = [
                by_key[(cycle, setting)] - by_key[(cycle, "baseline")]
                for cycle in cycles
            ]
            delta_median = median(deltas)
            pct = delta_median / base_median * 100.0
            parts.append(
                f"{delta_median:+.1f} us ({pct:+.2f}%, "
                f"IQR {iqr(deltas):.1f}, "
                f"min/max {min(deltas):+.1f}/{max(deltas):+.1f})"
            )
        deltas = [
            by_key[(cycle, "local_id_ignore")]
            - by_key[(cycle, "global_ignore")]
            for cycle in cycles
        ]
        parts.append(
            f"{median(deltas):+.1f} us (IQR {iqr(deltas):.1f}, "
            f"min/max {min(deltas):+.1f}/{max(deltas):+.1f})"
        )
        print("| " + " | ".join(parts) + " |")
    print()


def print_ignore_table(rows: list[dict[str, object]]) -> None:
    print("## Ignore Enabled")
    print("| tokens | setting | rank0 true/total | rank1 true/total |")
    print("|---:|---|---:|---:|")
    for tokens in sorted({int(row["tokens"]) for row in rows}):
        for setting in SETTINGS:
            subset = by_tokens_setting(rows, tokens, setting)
            rank0 = Counter(str(row["ep_ignore_enabled_rank0"]) for row in subset)
            rank1 = Counter(str(row["ep_ignore_enabled_rank1"]) for row in subset)
            print(
                f"| {tokens} | {setting} | "
                f"{rank0.get('True', 0)}/{len(subset)} | "
                f"{rank1.get('True', 0)}/{len(subset)} |"
            )
    print()


def print_rank_stats_table(rows: list[dict[str, object]]) -> None:
    print("## Rank Stats")
    print(
        "| tokens | setting | critical rank r0/r1/tie | "
        "rank0 median us | rank1 median us | recv r0/r1 median |"
    )
    print("|---:|---|---:|---:|---:|---:|")
    for tokens in sorted({int(row["tokens"]) for row in rows}):
        for setting in SETTINGS:
            subset = by_tokens_setting(rows, tokens, setting)
            rank0 = [float(row["rank0_forward_us"]) for row in subset]
            rank1 = [float(row["rank1_forward_us"]) for row in subset]
            critical = Counter(
                "tie"
                if a == b
                else ("rank0" if a > b else "rank1")
                for a, b in zip(rank0, rank1)
            )
            recv0 = [
                int(row["received_tokens_rank0"])
                for row in subset
                if row["received_tokens_rank0"] is not None
            ]
            recv1 = [
                int(row["received_tokens_rank1"])
                for row in subset
                if row["received_tokens_rank1"] is not None
            ]
            print(
                f"| {tokens} | {setting} | "
                f"{critical.get('rank0', 0)}/"
                f"{critical.get('rank1', 0)}/"
                f"{critical.get('tie', 0)} | "
                f"{median(rank0):.1f} | {median(rank1):.1f} | "
                f"{median(recv0):.0f}/{median(recv1):.0f} |"
            )
    print()


def print_route_stats_table(rows: list[dict[str, object]]) -> None:
    print("## Route Stats")
    print(
        "| tokens | received tokens r0/r1 | valid route pairs r0/r1 | "
        "invalid route pairs r0/r1 |"
    )
    print("|---:|---:|---:|---:|")
    for tokens in sorted({int(row["tokens"]) for row in rows}):
        subset = by_tokens_setting(rows, tokens, "baseline")
        recv0 = [int(row["received_tokens_rank0"]) for row in subset]
        recv1 = [int(row["received_tokens_rank1"]) for row in subset]
        valid0 = [int(row["valid_route_pairs_rank0"]) for row in subset]
        valid1 = [int(row["valid_route_pairs_rank1"]) for row in subset]
        invalid0 = [int(row["invalid_route_pairs_rank0"]) for row in subset]
        invalid1 = [int(row["invalid_route_pairs_rank1"]) for row in subset]
        print(
            f"| {tokens} | {median(recv0):.0f}/{median(recv1):.0f} | "
            f"{median(valid0):.0f}/{median(valid1):.0f} | "
            f"{median(invalid0):.0f}/{median(invalid1):.0f} |"
        )
    print()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.csv_path)
    validate_rows(rows)
    print_absolute_table(rows)
    print_paired_table(rows)
    print_ignore_table(rows)
    print_rank_stats_table(rows)
    print_route_stats_table(rows)


if __name__ == "__main__":
    main()
