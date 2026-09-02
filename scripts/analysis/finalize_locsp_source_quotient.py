"""Score the preregistered LOCSP physical-source quotient experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METHOD = "grouped_physical_receivers__cauchy"
KEY = f"trajectory__{METHOD}"


def _rmse(squared_errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(squared_errors)))


def _metrics(trajectory: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = trajectory - truth
    distance = np.linalg.norm(error, axis=1)
    horizontal = np.linalg.norm(error[:, :2], axis=1)
    return {
        "rmse_3d_m": _rmse(distance**2),
        "rmse_xy_m": _rmse(horizontal**2),
        "rmse_z_m": _rmse(error[:, 2] ** 2),
        "median_3d_m": float(np.median(distance)),
        "p95_3d_m": float(np.percentile(distance, 95.0)),
        "maximum_3d_m": float(np.max(distance)),
    }


def _paired_masked_bootstrap(
    candidate_squared: np.ndarray,
    baseline_squared: np.ndarray,
    timestamps: np.ndarray,
    block: int,
    trials: int,
    seed: int,
) -> dict[str, float | bool | int]:
    """Moving-block bootstrap that never bridges a reference outage."""

    count = len(candidate_squared)
    if count < block:
        raise ValueError("fewer valid reference states than one bootstrap block")
    starts = np.asarray(
        [
            start
            for start in range(count - block + 1)
            if np.max(np.diff(timestamps[start : start + block])) <= 1.5
        ],
        dtype=int,
    )
    if not len(starts):
        raise ValueError("no valid within-segment bootstrap blocks")
    rng = np.random.default_rng(seed)
    blocks_needed = int(np.ceil(count / block))
    values = np.empty(trials)
    for trial in range(trials):
        chosen = rng.choice(starts, blocks_needed, replace=True)
        indices = np.concatenate(
            [np.arange(start, start + block) for start in chosen]
        )[:count]
        values[trial] = _rmse(candidate_squared[indices]) - _rmse(
            baseline_squared[indices]
        )
    low, high = np.percentile(values, [2.5, 97.5])
    return {
        "candidate_minus_baseline_rmse_m": (
            _rmse(candidate_squared) - _rmse(baseline_squared)
        ),
        "ci95_low_m": float(low),
        "ci95_high_m": float(high),
        "strict_improvement_ci_below_zero": bool(high < 0.0),
        "one_sided_bootstrap_p": float(
            (1 + np.count_nonzero(values >= 0.0)) / (trials + 1)
        ),
        "eligible_within_segment_block_starts": int(len(starts)),
    }


def _load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    for prefix in ("two-source", "source-aware", "naive"):
        parser.add_argument(f"--{prefix}-report", type=Path, required=True)
        parser.add_argument(f"--{prefix}-trajectory", type=Path, required=True)
    parser.add_argument("--stream-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-label", default="LOCSP CR2")
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument("--block", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    reports = {
        "two": _load_report(args.two_source_report),
        "aware": _load_report(args.source_aware_report),
        "naive": _load_report(args.naive_report),
    }
    starts = {int(report["evaluation_start_index"]) for report in reports.values()}
    if len(starts) != 1:
        raise ValueError(f"evaluation start mismatch: {starts}")
    start = starts.pop()

    paths = (
        args.two_source_trajectory,
        args.source_aware_trajectory,
        args.naive_trajectory,
    )
    with np.load(paths[0], allow_pickle=False) as two, np.load(
        paths[1], allow_pickle=False
    ) as aware, np.load(paths[2], allow_pickle=False) as naive:
        for archive in (aware, naive):
            if not np.array_equal(two["timestamps_s"], archive["timestamps_s"]):
                raise ValueError("timestamp mismatch")
            if not np.array_equal(
                two["truth_positions_enu_m"],
                archive["truth_positions_enu_m"],
                equal_nan=True,
            ):
                raise ValueError("truth mismatch")
        all_truth = np.asarray(two["truth_positions_enu_m"])
        valid = np.all(np.isfinite(all_truth), axis=1)
        valid[:start] = False
        valid_times = np.asarray(two["timestamps_s"])[valid]
        truth = all_truth[valid]
        trajectories = {
            "two": np.asarray(two[KEY])[valid],
            "aware": np.asarray(aware[KEY])[valid],
            "naive": np.asarray(naive[KEY])[valid],
        }

    metric = {name: _metrics(value, truth) for name, value in trajectories.items()}
    replay_difference = np.linalg.norm(
        trajectories["aware"] - trajectories["two"], axis=1
    )
    aware_squared = np.sum((trajectories["aware"] - truth) ** 2, axis=1)
    naive_squared = np.sum((trajectories["naive"] - truth) ** 2, axis=1)
    bootstrap = _paired_masked_bootstrap(
        aware_squared,
        naive_squared,
        valid_times,
        args.block,
        args.trials,
        args.seed,
    )
    reduction = 100.0 * (
        metric["naive"]["rmse_3d_m"] - metric["aware"]["rmse_3d_m"]
    ) / metric["naive"]["rmse_3d_m"]
    p95_reduction = 100.0 * (
        metric["naive"]["p95_3d_m"] - metric["aware"]["p95_3d_m"]
    ) / metric["naive"]["p95_3d_m"]
    payload = {
        "dataset": args.dataset_label,
        "dataset_doi": "10.57745/YCXRWF",
        "candidate_physical_sources": ["u-blox M8T", "Septentrio"],
        "reference_only": "NovAtel INSPVA",
        "lidar_frontend": "KISS-ICP on Hesai PandarXT-32",
        "evaluation_epoch_count": int(len(truth)),
        "reference_valid_state_fraction": float(len(truth) / len(all_truth)),
        "evaluation_start_index": start,
        "stream_equivalence_audit": _load_report(args.stream_audit),
        "source_quotient_with_replay": metric["aware"],
        "three_stream_naive": metric["naive"],
        "two_physical_sources_without_replay": metric["two"],
        "replay_invariance": {
            "maximum_trajectory_difference_m": float(np.max(replay_difference)),
            "mean_trajectory_difference_m": float(np.mean(replay_difference)),
            "registered_tolerance_m": 1e-9,
            "pass": bool(np.max(replay_difference) <= 1e-9),
        },
        "source_quotient_vs_naive": {
            "rmse_reduction_percent": reduction,
            "p95_reduction_percent": p95_reduction,
            **bootstrap,
        },
        "scope": (
            "Independent external validation of physical-source replay "
            "invariance and information multiplicity; not validation of the "
            "three-receiver triangle selector or pair-collapse rule."
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows = [
        {"method": "source_quotient_with_replay", **metric["aware"]},
        {"method": "three_stream_naive", **metric["naive"]},
        {"method": "two_physical_sources_without_replay", **metric["two"]},
    ]
    with (args.output_root / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
