"""Paired moving-block bootstrap for the frozen CI baselines.

The proposed and covariance-intersection trajectories are evaluated on the
same state timestamps, hidden truth, and post-initialization interval.  This
script reads only already optimized trajectories and cannot change factors or
method parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.analysis.analyze_paired_statistics import _apply_holm, _paired_ci, _rmse


NATURAL_CASES = ("hk", "deep", "harsh", "phone")


def _case_paths(
    name: str,
    natural_root: Path,
    fault_root: Path,
) -> tuple[Path, Path]:
    root = natural_root if name in NATURAL_CASES else fault_root
    return root / f"{name}.json", root / f"{name}.npz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--natural-root", type=Path, required=True)
    parser.add_argument("--fault-root", type=Path, required=True)
    parser.add_argument("--ci-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument("--block", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    fault_cases = sorted(path.stem for path in args.fault_root.glob("*.json"))
    cases = [*NATURAL_CASES, *fault_cases]
    baselines = ("global_ci__cauchy", "provenance_ci__cauchy")
    rows: list[dict[str, object]] = []

    for case_index, name in enumerate(cases):
        report_path, proposed_path = _case_paths(
            name, args.natural_root, args.fault_root
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        start = int(report["evaluation_start_index"])
        ci_path = args.ci_root / f"{name}.npz"
        with np.load(proposed_path, allow_pickle=False) as proposed_archive, np.load(
            ci_path, allow_pickle=False
        ) as ci_archive:
            proposed_times = np.asarray(proposed_archive["timestamps_s"])
            ci_times = np.asarray(ci_archive["timestamps_s"])
            if not np.array_equal(proposed_times, ci_times):
                raise ValueError(f"timestamp mismatch for {name}")
            proposed_truth = np.asarray(
                proposed_archive["truth_positions_enu_m"]
            )
            ci_truth = np.asarray(ci_archive["truth_positions_enu_m"])
            if not np.array_equal(proposed_truth, ci_truth):
                raise ValueError(f"truth mismatch for {name}")

            truth = proposed_truth[start:]
            proposed = np.asarray(proposed_archive["trajectory__dda_fc"])[start:]
            proposed_squared = np.sum((proposed - truth) ** 2, axis=1)
            scope = "natural" if name in NATURAL_CASES else "fault"
            for baseline_index, baseline in enumerate(baselines):
                estimate = np.asarray(
                    ci_archive[f"trajectory__{baseline}"]
                )[start:]
                baseline_squared = np.sum((estimate - truth) ** 2, axis=1)
                rows.append(
                    {
                        "case": f"{scope}__{name}",
                        "baseline": baseline,
                        "evaluation_epoch_count": len(truth),
                        "candidate_rmse_m": _rmse(proposed_squared),
                        "baseline_rmse_m": _rmse(baseline_squared),
                        **_paired_ci(
                            proposed_squared,
                            baseline_squared,
                            args.block,
                            args.trials,
                            args.seed + 10 * case_index + baseline_index,
                        ),
                    }
                )

    _apply_holm(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "paired_moving_block_bootstrap_for_ci_baselines",
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
