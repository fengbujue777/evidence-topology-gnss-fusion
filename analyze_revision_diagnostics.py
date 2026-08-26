"""Diagnostics added for the covariance-union and topology review audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from paper_pipeline.covariance_union import pair_covariance_union
from paper_pipeline.loewner_envelope import provenance_loewner_envelope
from run_ci_baseline import _covariance_intersection


NATURAL_CASES = ("hk", "deep", "harsh", "phone")
FAULT_CASES = (
    "novatel_step",
    "novatel_ramp",
    "ublox_f9p_step",
    "ublox_f9p_ramp",
    "ublox_m8t_step",
    "ublox_m8t_ramp",
    "distributed_novatel_step_f9p_ramp",
)
CONFIDENCE_LEVELS = (0.50, 0.90, 0.95, 0.99)


def _floor_covariance(covariance: np.ndarray, floor_m: float = 1.0) -> np.ndarray:
    covariance = 0.5 * (np.asarray(covariance) + np.asarray(covariance).T)
    values, vectors = np.linalg.eigh(covariance)
    return (vectors * np.maximum(values, floor_m**2)) @ vectors.T


def _factor_candidates(
    points: np.ndarray, covariances: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    count = len(points)
    proposed_center, proposed_covariance, _ = provenance_loewner_envelope(
        points,
        covariances,
        floor_m=1.0,
        center_mode="arithmetic",
        envelope_mode="positive_part",
    )
    union_center, union_covariance, _ = pair_covariance_union(
        points, covariances, floor_m=1.0
    )
    ci_center, ci_covariance, _ = _covariance_intersection(
        points, covariances
    )
    regularized = np.asarray(
        [_floor_covariance(covariance) for covariance in covariances]
    )
    average_center = np.mean(points, axis=0)
    independence_covariance = _floor_covariance(
        np.sum(regularized, axis=0) / float(count**2)
    )
    marginal_average_covariance = _floor_covariance(
        np.mean(regularized, axis=0)
    )
    return {
        "proposed_envelope": (proposed_center, proposed_covariance),
        "same_subset_cu": (union_center, union_covariance),
        "same_subset_ci": (ci_center, ci_covariance),
        "independence_average": (average_center, independence_covariance),
        "marginal_average": (average_center, marginal_average_covariance),
    }


def _normalized_squared_error(
    center: np.ndarray, covariance: np.ndarray, truth: np.ndarray
) -> float:
    error = np.asarray(center) - np.asarray(truth)
    return float(error @ np.linalg.solve(covariance, error))


def _ellipsoid_volume(covariance: np.ndarray, probability: float = 0.95) -> float:
    radius_squared = float(chi2.ppf(probability, df=3))
    determinant = max(float(np.linalg.det(covariance)), 0.0)
    return float(
        (4.0 * np.pi / 3.0)
        * radius_squared ** 1.5
        * np.sqrt(determinant)
    )


def _calibration_case(
    name: str, group: str, root: Path
) -> list[dict[str, object]]:
    factor_path = root / f"{name}.selected_factor_inputs.npz"
    trajectory_path = root / f"{name}.npz"
    report_path = root / f"{name}.json"
    with np.load(factor_path, allow_pickle=False) as factors:
        factor_arrays = {key: factors[key] for key in factors.files}
    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        state_times = np.asarray(trajectory["timestamps_s"], dtype=float)
        state_truth = np.asarray(
            trajectory["truth_positions_enu_m"], dtype=float
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    evaluation_start = float(
        state_times[int(report["evaluation_start_index"])]
    )
    factor_times = np.asarray(factor_arrays["timestamps_s"], dtype=float)
    keep = factor_times >= evaluation_start
    truth = np.column_stack(
        [
            np.interp(factor_times, state_times, state_truth[:, dimension])
            for dimension in range(3)
        ]
    )
    records: dict[str, dict[str, list[float]]] = {}

    def append(method: str, q_value: float, volume: float) -> None:
        target = records.setdefault(method, {"q": [], "volume": []})
        target["q"].append(float(q_value))
        target["volume"].append(float(volume))

    for epoch in np.flatnonzero(keep):
        count = int(factor_arrays["member_counts"][epoch])
        points = np.asarray(
            factor_arrays["member_points_enu_m"][epoch, :count],
            dtype=float,
        )
        covariances = np.asarray(
            factor_arrays["member_covariances_m2"][epoch, :count],
            dtype=float,
        )
        candidates = _factor_candidates(points, covariances)
        for method, (center, covariance) in candidates.items():
            append(
                method,
                _normalized_squared_error(center, covariance, truth[epoch]),
                _ellipsoid_volume(covariance),
            )
        for point, covariance in zip(points, covariances):
            covariance = _floor_covariance(covariance)
            append(
                "selected_member_marginal",
                _normalized_squared_error(point, covariance, truth[epoch]),
                _ellipsoid_volume(covariance),
            )

    rows = []
    quantiles = {
        probability: float(chi2.ppf(probability, df=3))
        for probability in CONFIDENCE_LEVELS
    }
    for method, values in records.items():
        q_values = np.asarray(values["q"], dtype=float)
        volumes = np.asarray(values["volume"], dtype=float)
        row: dict[str, object] = {
            "case": name,
            "case_group": group,
            "method": method,
            "sample_count": int(len(q_values)),
            "q_median": float(np.median(q_values)),
            "q_p95": float(np.percentile(q_values, 95.0)),
            "median_95_volume_m3": float(np.median(volumes)),
        }
        for probability, threshold in quantiles.items():
            row[f"coverage_{int(100 * probability)}"] = float(
                np.mean(q_values <= threshold)
            )
        rows.append(row)
    return rows


def _aggregate_calibration(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    # Recompute pooled results from per-case factor files in the caller; the
    # table below is an equal-case descriptive summary, not an iid interval.
    output = []
    for group in ("natural", "controlled_fault"):
        group_rows = [row for row in rows if row["case_group"] == group]
        for method in sorted({str(row["method"]) for row in group_rows}):
            selected = [row for row in group_rows if row["method"] == method]
            result: dict[str, object] = {
                "case_group": group,
                "method": method,
                "case_count": len(selected),
                "aggregation": "equal-case descriptive mean",
                "median_95_volume_m3": float(
                    np.mean([row["median_95_volume_m3"] for row in selected])
                ),
            }
            for probability in CONFIDENCE_LEVELS:
                key = f"coverage_{int(100 * probability)}"
                result[key] = float(np.mean([row[key] for row in selected]))
            output.append(result)
    return output


def _faulty_receivers(case: str) -> set[str]:
    if case == "distributed_novatel_step_f9p_ramp":
        return {"novatel", "ublox_f9p"}
    if case.startswith("novatel"):
        return {"novatel"}
    if case.startswith("ublox_f9p"):
        return {"ublox_f9p"}
    if case.startswith("ublox_m8t"):
        return {"ublox_m8t"}
    raise ValueError(f"unknown fault case: {case}")


def _fault_start(case_root: Path, faulty_receivers: set[str]) -> float:
    starts = []
    for path in case_root.glob("*.hidden_labels.npz"):
        token = path.name.lower()
        group = (
            "novatel" if "novatel" in token else
            "ublox_f9p" if "f9p" in token else
            "ublox_m8t" if "m8t" in token else ""
        )
        if group not in faulty_receivers:
            continue
        with np.load(path, allow_pickle=False) as labels:
            start = float(np.asarray(labels["fault_start_s"]).reshape(()))
        if np.isfinite(start):
            starts.append(start)
    if not starts:
        manifest = json.loads(
            (case_root / "fault_injection_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        if "start_timestamp_s" in manifest:
            return float(manifest["start_timestamp_s"])
        start_fraction = float(manifest["start_fraction"])
        for path in case_root.glob("*.estimator_input.npz"):
            token = path.name.lower()
            group = (
                "novatel" if "novatel" in token else
                "ublox_f9p" if "f9p" in token else
                "ublox_m8t" if "m8t" in token else ""
            )
            if group not in faulty_receivers:
                continue
            with np.load(path, allow_pickle=False) as estimator:
                timestamps = np.asarray(
                    estimator["gnss_timestamps_s"], dtype=float
                )
            start_index = min(
                max(int(np.floor(start_fraction * len(timestamps))), 1),
                len(timestamps) - 1,
            )
            starts.append(float(timestamps[start_index]))
    if not starts:
        raise RuntimeError(f"fault start unavailable in {case_root}")
    return max(starts)


def _topology_diagnostics(
    root: Path, fault_case_root: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    natural_rows = []
    for name in NATURAL_CASES:
        report = json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
        windows = report["candidate_subset_windows"]
        subsets = [tuple(window["selected_receivers"]) for window in windows]
        switches = sum(left != right for left, right in zip(subsets, subsets[1:]))
        natural_rows.append(
            {
                "case": name,
                "window_count": len(windows),
                "singleton_fraction": float(
                    np.mean([len(subset) == 1 for subset in subsets])
                ),
                "pair_fraction": float(
                    np.mean([len(subset) == 2 for subset in subsets])
                ),
                "topology_switch_count": int(switches),
                "switch_fraction_per_transition": (
                    float(switches / (len(subsets) - 1))
                    if len(subsets) > 1 else 0.0
                ),
                "subset_counts": report["candidate_subset_counts"],
            }
        )

    fault_rows = []
    all_receivers = {"novatel", "ublox_f9p", "ublox_m8t"}
    for name in FAULT_CASES:
        report = json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
        factors = np.load(
            root / f"{name}.selected_factor_inputs.npz",
            allow_pickle=False,
        )
        faulty = _faulty_receivers(name)
        healthy = all_receivers - faulty
        onset = _fault_start(fault_case_root / name / "cases", faulty)
        windows = report["candidate_subset_windows"]
        timestamps = np.asarray(factors["timestamps_s"], dtype=float)
        post_windows = []
        for window in windows:
            start_epoch = int(window["start_epoch"])
            if start_epoch >= len(timestamps):
                continue
            if float(timestamps[start_epoch]) >= onset:
                post_windows.append(window)
        epoch_mask = timestamps >= onset
        epoch_subsets = []
        for epoch in np.flatnonzero(epoch_mask):
            count = int(factors["member_counts"][epoch])
            epoch_subsets.append(
                set(str(item) for item in factors["receiver_names"][epoch, :count])
            )
        safe = [faulty.isdisjoint(subset) for subset in epoch_subsets]
        healthy_pair = [
            subset == healthy and len(healthy) == 2 for subset in epoch_subsets
        ]
        healthy_singleton = [
            len(subset) == 1 and subset.issubset(healthy)
            for subset in epoch_subsets
        ]
        healthy_retention = [
            len(subset & healthy) / max(len(healthy), 1)
            for subset in epoch_subsets
        ]
        first_safe_offset = None
        for epoch, is_safe in zip(np.flatnonzero(epoch_mask), safe):
            if is_safe:
                first_safe_offset = float(timestamps[epoch] - onset)
                break
        fault_rows.append(
            {
                "case": name,
                "faulty_receivers": sorted(faulty),
                "post_onset_epoch_count": len(epoch_subsets),
                "post_onset_full_window_count": len(post_windows),
                "fault_exclusion_rate": float(np.mean(safe)) if safe else None,
                "fault_inclusion_rate": float(1.0 - np.mean(safe)) if safe else None,
                "correct_healthy_pair_rate": (
                    float(np.mean(healthy_pair)) if healthy_pair else None
                ),
                "healthy_singleton_rate": (
                    float(np.mean(healthy_singleton))
                    if healthy_singleton else None
                ),
                "mean_healthy_receiver_retention": (
                    float(np.mean(healthy_retention))
                    if healthy_retention else None
                ),
                "first_safe_assigned_epoch_offset_s": first_safe_offset,
                "latency_interpretation": (
                    "offline same-window assignment; not a causal deployment delay"
                ),
                "recovery_delay_s": None,
                "recovery_delay_note": "not identifiable: injected faults persist to route end",
                "post_onset_subset_counts": {
                    ",".join(window["selected_receivers"]): sum(
                        tuple(other["selected_receivers"])
                        == tuple(window["selected_receivers"])
                        for other in post_windows
                    )
                    for window in post_windows
                },
            }
        )
        factors.close()
    return natural_rows, fault_rows


def _rmse(report: dict[str, object], metric: str) -> float:
    return float(report["metrics"][metric]["rmse_3d_m"])


def _cu_and_route_balanced(
    proposed_natural_root: Path,
    proposed_fault_root: Path,
    cu_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    comparison = []
    values: dict[str, dict[str, float]] = {}
    for group, names, proposed_root in (
        ("natural", NATURAL_CASES, proposed_natural_root),
        ("controlled_fault", FAULT_CASES, proposed_fault_root),
    ):
        for name in names:
            proposed = json.loads(
                (proposed_root / f"{name}.json").read_text(encoding="utf-8")
            )
            cu = json.loads((cu_root / f"{name}.json").read_text(encoding="utf-8"))
            proposed_rmse = _rmse(proposed, "motion_margin_subset")
            cu_rmse = _rmse(cu, "motion_margin_subset")
            comparison.append(
                {
                    "case": name,
                    "case_group": group,
                    "proposed_envelope_rmse_m": proposed_rmse,
                    "same_subset_cu_rmse_m": cu_rmse,
                    "proposed_minus_cu_m": proposed_rmse - cu_rmse,
                }
            )
            values[name] = {
                "proposed": proposed_rmse,
                "all_receiver": _rmse(
                    proposed, "grouped_physical_receivers__cauchy"
                ),
            }

    # Equal weight for three route clusters.  Medium natural plus its seven
    # fault derivatives form one cluster; Harsh and phones share one route.
    clusters = {
        "medium_route": ["hk", *FAULT_CASES],
        "deep_route": ["deep"],
        "harsh_shared_route": ["harsh", "phone"],
    }
    route_rows = []
    for cluster, names in clusters.items():
        route_rows.append(
            {
                "route_cluster": cluster,
                "tested_case_count": len(names),
                "proposed_mean_rmse_m": float(
                    np.mean([values[name]["proposed"] for name in names])
                ),
                "all_receiver_mean_rmse_m": float(
                    np.mean([values[name]["all_receiver"] for name in names])
                ),
            }
        )
    route_rows.append(
        {
            "route_cluster": "equal_route_cluster_mean",
            "tested_case_count": sum(len(names) for names in clusters.values()),
            "proposed_mean_rmse_m": float(
                np.mean([row["proposed_mean_rmse_m"] for row in route_rows])
            ),
            "all_receiver_mean_rmse_m": float(
                np.mean([row["all_receiver_mean_rmse_m"] for row in route_rows])
            ),
        }
    )
    return comparison, route_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposed-natural-root", type=Path, required=True)
    parser.add_argument("--proposed-fault-root", type=Path, required=True)
    parser.add_argument("--cu-root", type=Path, required=True)
    parser.add_argument("--fault-case-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    calibration_rows = []
    for group, names in (
        ("natural", NATURAL_CASES),
        ("controlled_fault", FAULT_CASES),
    ):
        for name in names:
            calibration_rows.extend(_calibration_case(name, group, args.cu_root))
    calibration_summary = _aggregate_calibration(calibration_rows)
    natural_topology, fault_topology = _topology_diagnostics(
        args.cu_root, args.fault_case_root
    )
    cu_comparison, route_balanced = _cu_and_route_balanced(
        args.proposed_natural_root,
        args.proposed_fault_root,
        args.cu_root,
    )
    payload = {
        "calibration_interpretation": (
            "factor-level normalized squared error; descriptive coverage, "
            "not posterior NIS/NEES"
        ),
        "calibration_by_case": calibration_rows,
        "calibration_summary": calibration_summary,
        "natural_topology": natural_topology,
        "controlled_fault_topology": fault_topology,
        "same_subset_cu_comparison": cu_comparison,
        "route_balanced_aggregate": route_balanced,
        "independence_note": (
            "The seven injected cases share the Medium route; Harsh and "
            "phones share a route. Route-balanced rows are descriptive and "
            "do not create additional independent routes."
        ),
    }
    (args.output_root / "revision_diagnostics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output_root / "calibration_by_case.csv", calibration_rows)
    _write_csv(args.output_root / "calibration_summary.csv", calibration_summary)
    _write_csv(args.output_root / "natural_topology.csv", natural_topology)
    _write_csv(args.output_root / "fault_topology.csv", fault_topology)
    _write_csv(args.output_root / "same_subset_cu.csv", cu_comparison)
    _write_csv(args.output_root / "route_balanced.csv", route_balanced)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
