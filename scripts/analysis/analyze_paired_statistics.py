"""Moving-block bootstrap for the final topology candidate.

The script consumes only already frozen trajectories and hidden truth stored in
their result archives.  It does not change method parameters or trajectories.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _rmse(error_squared: np.ndarray) -> float:
    return float(np.sqrt(np.mean(error_squared)))


def _paired_ci(
    candidate_squared: np.ndarray,
    baseline_squared: np.ndarray,
    block: int,
    trials: int,
    seed: int,
) -> dict[str, float | bool]:
    count = len(candidate_squared)
    starts = np.arange(max(count - block + 1, 1))
    blocks_needed = int(np.ceil(count / block))
    rng = np.random.default_rng(seed)
    values = np.empty(trials)
    for trial in range(trials):
        chosen = rng.choice(starts, blocks_needed, replace=True)
        indices = np.concatenate(
            [
                np.arange(start, min(start + block, count))
                for start in chosen
            ]
        )[:count]
        values[trial] = _rmse(candidate_squared[indices]) - _rmse(
            baseline_squared[indices]
        )
    observed = _rmse(candidate_squared) - _rmse(baseline_squared)
    low, high = np.percentile(values, [2.5, 97.5])
    return {
        "candidate_minus_baseline_rmse_m": observed,
        "ci95_low_m": float(low),
        "ci95_high_m": float(high),
        "strict_improvement_ci_below_zero": bool(high < 0.0),
        "one_sided_bootstrap_p": float(
            (1 + np.count_nonzero(values >= 0.0)) / (trials + 1)
        ),
    }


def _apply_holm(rows: list[dict[str, object]], alpha: float = 0.05) -> None:
    for scope in ("natural", "fault"):
        baselines = sorted(
            {
                str(row["baseline"])
                for row in rows
                if str(row["case"]).startswith(f"{scope}__")
            }
        )
        for baseline in baselines:
            family = [
                row
                for row in rows
                if str(row["case"]).startswith(f"{scope}__")
                and row["baseline"] == baseline
            ]
            ordered = sorted(
                family, key=lambda row: float(row["one_sided_bootstrap_p"])
            )
            still_rejecting = True
            count = len(ordered)
            for rank, row in enumerate(ordered):
                threshold = alpha / (count - rank)
                reject = bool(
                    still_rejecting
                    and float(row["one_sided_bootstrap_p"]) <= threshold
                    and float(row["candidate_minus_baseline_rmse_m"]) < 0.0
                )
                row["holm_family"] = f"{scope}__{baseline}"
                row["holm_rank"] = rank + 1
                row["holm_threshold"] = threshold
                row["holm_significant_0p05"] = reject
                if not reject:
                    still_rejecting = False


def _analyze(
    name: str,
    report_path: Path,
    trajectory_path: Path,
    trials: int,
    block: int,
    seed: int,
) -> list[dict[str, object]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    start = int(report["evaluation_start_index"])
    with np.load(trajectory_path, allow_pickle=False) as archive:
        truth = np.asarray(archive["truth_positions_enu_m"])[start:]
        candidate = np.asarray(archive["trajectory__dda_fc"])[start:]
        candidate_squared = np.sum((candidate - truth) ** 2, axis=1)
        rows = []
        baselines = [
            "grouped_physical_receivers__direct",
            "grouped_physical_receivers__cauchy",
            "grouped_physical_receivers__switch_irls",
            "grouped_physical_receivers__dcs_irls",
            "hierarchical_consensus__cauchy",
        ]
        baselines.extend(
            sorted(
                key.removeprefix("trajectory__")
                for key in archive.files
                if key.startswith("trajectory__fixed_pair_consensus__")
            )
        )
        for baseline in baselines:
            estimate = np.asarray(archive[f"trajectory__{baseline}"])[start:]
            baseline_squared = np.sum((estimate - truth) ** 2, axis=1)
            rows.append(
                {
                    "case": name,
                    "baseline": baseline,
                    "evaluation_epoch_count": len(truth),
                    "candidate_rmse_m": _rmse(candidate_squared),
                    "baseline_rmse_m": _rmse(baseline_squared),
                    **_paired_ci(
                        candidate_squared,
                        baseline_squared,
                        block,
                        trials,
                        seed + len(rows),
                    ),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--natural-root", type=Path, required=True)
    parser.add_argument("--fault-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument("--block", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for panel in ("hk", "deep", "harsh", "phone"):
        rows.extend(
            _analyze(
                f"natural__{panel}",
                args.natural_root / f"{panel}.json",
                args.natural_root / f"{panel}.npz",
                args.trials,
                args.block,
                args.seed + len(rows),
            )
        )
    for report_path in sorted(args.fault_root.glob("*.json")):
        rows.extend(
            _analyze(
                f"fault__{report_path.stem}",
                report_path,
                report_path.with_suffix(".npz"),
                args.trials,
                args.block,
                args.seed + len(rows),
            )
        )

    _apply_holm(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "moving_block_bootstrap",
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
