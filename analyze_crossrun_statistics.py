"""Paired block-bootstrap statistics for final-method cross-run ablations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_paired_statistics import _apply_holm, _paired_ci, _rmse


NATURAL = ("hk", "deep", "harsh", "phone")
FAULT = (
    "novatel_step",
    "novatel_ramp",
    "ublox_f9p_step",
    "ublox_f9p_ramp",
    "ublox_m8t_step",
    "ublox_m8t_ramp",
    "distributed_novatel_step_f9p_ramp",
)


def _row(
    scope: str,
    case: str,
    full_root: Path,
    comparator_root: Path,
    comparator: str,
    block: int,
    trials: int,
    seed: int,
) -> dict[str, object]:
    report = json.loads(
        (full_root / f"{case}.json").read_text(encoding="utf-8")
    )
    start = int(report["evaluation_start_index"])
    with np.load(full_root / f"{case}.npz", allow_pickle=False) as full:
        truth = np.asarray(full["truth_positions_enu_m"])[start:]
        candidate = np.asarray(full["trajectory__dda_fc"])[start:]
    with np.load(
        comparator_root / f"{case}.npz", allow_pickle=False
    ) as ablation:
        estimate = np.asarray(ablation["trajectory__dda_fc"])[start:]
    candidate_squared = np.sum((candidate - truth) ** 2, axis=1)
    comparator_squared = np.sum((estimate - truth) ** 2, axis=1)
    return {
        "case": f"{scope}__{case}",
        "baseline": comparator,
        "evaluation_epoch_count": len(truth),
        "candidate_rmse_m": _rmse(candidate_squared),
        "baseline_rmse_m": _rmse(comparator_squared),
        **_paired_ci(
            candidate_squared,
            comparator_squared,
            block,
            trials,
            seed,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-natural-root", type=Path, required=True)
    parser.add_argument("--full-fault-root", type=Path, required=True)
    parser.add_argument("--component-root", type=Path, required=True)
    parser.add_argument("--factor-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--block", type=int, default=30)
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    comparators = (
        (
            "motion_single",
            args.component_root / "motion_single" / "natural",
            args.component_root / "motion_single" / "fault",
        ),
        (
            "separation_pair",
            args.component_root / "separation_pair" / "natural",
            args.component_root / "separation_pair" / "fault",
        ),
        (
            "selected_pair_independent_factors",
            args.factor_root / "natural",
            args.factor_root / "fault",
        ),
    )
    rows: list[dict[str, object]] = []
    for comparator, natural_root, fault_root in comparators:
        for case in NATURAL:
            rows.append(
                _row(
                    "natural",
                    case,
                    args.full_natural_root,
                    natural_root,
                    comparator,
                    args.block,
                    args.trials,
                    args.seed + len(rows),
                )
            )
        for case in FAULT:
            rows.append(
                _row(
                    "fault",
                    case,
                    args.full_fault_root,
                    fault_root,
                    comparator,
                    args.block,
                    args.trials,
                    args.seed + len(rows),
                )
            )
    _apply_holm(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "paired moving-block bootstrap across frozen runs",
        "trials": args.trials,
        "block_epochs": args.block,
        "rows": rows,
    }
    (args.output_root / "report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (args.output_root / "report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
