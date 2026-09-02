"""Exploratory KISS fusion with simultaneous heterogeneous GNSS solutions.

This is a mechanism scout, not confirmatory evidence.  It tests whether the
receiver/constellation diversity already present in UrbanNav can form a useful
consensus position stream before a paper method is designed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evidence_topology.fusion_graph_backend import _build_graph, _optimize, _key
from paper_pipeline.alignment import (
    estimate_truth_free_alignment,
    interpolate_poses,
    transform_poses,
)
from paper_pipeline.cases import load_estimator_input


def _nearest_indices(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right = np.searchsorted(source, target)
    right = np.clip(right, 0, len(source) - 1)
    left = np.clip(right - 1, 0, len(source) - 1)
    choose_left = np.abs(source[left] - target) <= np.abs(source[right] - target)
    indices = np.where(choose_left, left, right)
    return indices, np.abs(source[indices] - target)


def _geometric_median(points: np.ndarray) -> np.ndarray:
    estimate = np.median(points, axis=0)
    for _ in range(50):
        distances = np.linalg.norm(points - estimate, axis=1)
        if np.min(distances) < 1e-9:
            return points[int(np.argmin(distances))].copy()
        weights = 1.0 / np.maximum(distances, 1e-9)
        updated = np.sum(weights[:, None] * points, axis=0) / np.sum(weights)
        if np.linalg.norm(updated - estimate) < 1e-9:
            return updated
        estimate = updated
    return estimate


def _consensus_epoch(
    points: np.ndarray, covariances: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "geometric_median":
        center = _geometric_median(points)
    elif mode == "coordinate_median":
        center = np.median(points, axis=0)
    elif mode == "covariance_mean":
        information = np.zeros((3, 3))
        rhs = np.zeros(3)
        for point, covariance in zip(points, covariances):
            values, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
            covariance = (vectors * np.maximum(values, 1.0)) @ vectors.T
            inverse = np.linalg.inv(covariance)
            information += inverse
            rhs += inverse @ point
        center = np.linalg.solve(information, rhs)
    else:
        raise ValueError(mode)
    deviations = points - center
    scatter = (
        deviations.T @ deviations / max(len(points) - 1, 1)
        if len(points) > 1
        else np.zeros((3, 3))
    )
    # Do not divide the base covariance by receiver count: urban multipath is
    # partly common-mode.  Receiver disagreement is added conservatively.
    base = np.median(covariances, axis=0)
    covariance = 0.5 * (base + base.T) + scatter
    values, vectors = np.linalg.eigh(covariance)
    covariance = (vectors * np.maximum(values, 1.0)) @ vectors.T
    return center, covariance


def _hardware_group(path: Path) -> str:
    name = path.name.lower()
    if "m8t_" in name:
        return "ublox_m8t"
    if "f9p" in name:
        return "ublox_f9p"
    if "novatel" in name:
        return "novatel"
    if "trimble" in name:
        return "trimble"
    if "ublox" in name:
        return "ublox"
    return path.name.split("__")[1]


def _hierarchical_consensus(
    group_points: dict[str, list[np.ndarray]],
    group_covariances: dict[str, list[np.ndarray]],
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    centers = []
    covariances = []
    within_mode = (
        "geometric_median"
        if mode == "hierarchical_geometric"
        else "coordinate_median"
    )
    for group in sorted(group_points):
        center, covariance = _consensus_epoch(
            np.asarray(group_points[group]),
            np.asarray(group_covariances[group]),
            within_mode,
        )
        centers.append(center)
        covariances.append(covariance)
    return _consensus_epoch(
        np.asarray(centers),
        np.asarray(covariances),
        within_mode,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance-s", type=float, default=0.2)
    parser.add_argument("--min-solutions", type=int, default=3)
    args = parser.parse_args()

    inputs = sorted(args.case_root.glob("*.estimator_input.npz"))
    datasets = [load_estimator_input(path) for path in inputs]
    hardware_groups = [_hardware_group(path) for path in inputs]
    base = datasets[0]
    all_times = np.unique(
        np.concatenate([data.gnss_timestamps_s for data in datasets])
    )
    all_times = all_times[
        (all_times >= base.lidar_timestamps_s[0])
        & (all_times <= base.lidar_timestamps_s[-1])
    ]
    epoch_points = []
    epoch_covariances = []
    epoch_counts = []
    epoch_group_counts = []
    epoch_group_points = []
    epoch_group_covariances = []
    retained_times = []
    for timestamp in all_times:
        points = []
        covariances = []
        group_points: dict[str, list[np.ndarray]] = {}
        group_covariances: dict[str, list[np.ndarray]] = {}
        for data, group in zip(datasets, hardware_groups):
            indices, differences = _nearest_indices(
                data.gnss_timestamps_s, np.asarray([timestamp])
            )
            if differences[0] <= args.tolerance_s:
                index = int(indices[0])
                points.append(data.gnss_positions_enu_m[index])
                covariances.append(data.gnss_covariances_m2[index])
                group_points.setdefault(group, []).append(
                    data.gnss_positions_enu_m[index]
                )
                group_covariances.setdefault(group, []).append(
                    data.gnss_covariances_m2[index]
                )
        if (
            len(points) >= args.min_solutions
            and len(group_points) >= min(args.min_solutions, 2)
        ):
            retained_times.append(timestamp)
            epoch_points.append(np.asarray(points))
            epoch_covariances.append(np.asarray(covariances))
            epoch_counts.append(len(points))
            epoch_group_counts.append(len(group_points))
            epoch_group_points.append(group_points)
            epoch_group_covariances.append(group_covariances)
    times = np.asarray(retained_times)
    alignment_positions = np.asarray(
        [
            _hierarchical_consensus(
                group_points,
                group_covariances,
                "hierarchical_geometric",
            )[0]
            for group_points, group_covariances in zip(
                epoch_group_points, epoch_group_covariances
            )
        ]
    )
    alignment = estimate_truth_free_alignment(
        base.lidar_timestamps_s,
        base.lidar_poses_local,
        times,
        alignment_positions,
        ransac_threshold_m=20.0,
        minimum_count=30,
        minimum_duration_s=60.0,
    )
    lidar_enu = transform_poses(
        alignment.transform_enu_from_lidar, base.lidar_poses_local
    )
    lidar_at_gnss = interpolate_poses(
        base.lidar_timestamps_s, lidar_enu, times
    )
    hidden_path = inputs[0].with_name(
        inputs[0].name.replace(".estimator_input.npz", ".hidden_labels.npz")
    )
    with np.load(hidden_path, allow_pickle=False) as hidden:
        truth = interpolate_poses(
            np.asarray(hidden["truth_timestamps_s"]),
            np.asarray(hidden["truth_poses_enu"]),
            times,
        )[:, :3, 3]

    results = {}
    for mode in (
        "hierarchical_geometric",
        "hierarchical_coordinate",
        "geometric_median",
        "coordinate_median",
        "covariance_mean",
    ):
        positions = []
        covariances = []
        for epoch, (points, covariance_set) in enumerate(
            zip(epoch_points, epoch_covariances)
        ):
            if mode.startswith("hierarchical_"):
                position, covariance = _hierarchical_consensus(
                    epoch_group_points[epoch],
                    epoch_group_covariances[epoch],
                    mode,
                )
            else:
                position, covariance = _consensus_epoch(
                    points, covariance_set, mode
                )
            positions.append(position)
            covariances.append(covariance)
        positions = np.asarray(positions)
        covariances = np.asarray(covariances)
        graph, initial, _ = _build_graph(
            lidar_at_gnss,
            positions,
            covariances,
            "cauchy",
            20,
            20260727,
        )
        optimized = _optimize(graph, initial)
        estimates = np.asarray(
            [
                np.asarray(optimized.atPose3(_key(i)).translation())
                for i in range(len(times))
            ]
        )
        error = np.linalg.norm(
            estimates[alignment.used_count :] - truth[alignment.used_count :],
            axis=1,
        )
        gnss_error = np.linalg.norm(
            positions[alignment.used_count :] - truth[alignment.used_count :],
            axis=1,
        )
        results[mode] = {
            "batch_cauchy_ate_rmse_m": float(np.sqrt(np.mean(error**2))),
            "consensus_gnss_rmse_m": float(np.sqrt(np.mean(gnss_error**2))),
            "consensus_gnss_median_m": float(np.median(gnss_error)),
        }
    payload = {
        "evidentiary_status": "EXPLORATORY_MECHANISM_SCOUT",
        "receiver_solution_count": len(datasets),
        "physical_receiver_group_count": len(set(hardware_groups)),
        "physical_receiver_groups": sorted(set(hardware_groups)),
        "retained_epoch_count": len(times),
        "minimum_solutions_per_epoch": int(np.min(epoch_counts)),
        "median_solutions_per_epoch": float(np.median(epoch_counts)),
        "maximum_solutions_per_epoch": int(np.max(epoch_counts)),
        "minimum_groups_per_epoch": int(np.min(epoch_group_counts)),
        "median_groups_per_epoch": float(np.median(epoch_group_counts)),
        "maximum_groups_per_epoch": int(np.max(epoch_group_counts)),
        "alignment_used_count": int(alignment.used_count),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
