"""Fair same-epoch panel for dependency-aware GNSS receiver consensus.

This program is deliberately a development/adjudication experiment.  It uses
only estimator-side data to construct every method and reads hidden truth only
after all trajectories have been estimated.  All compared streams use the
same epochs and are scored after a common truth-free alignment prefix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from evidence_topology.fusion_graph_backend import _build_graph, _key, _optimize
from evidence_topology.receiver_consensus import (
    _consensus_epoch,
    _geometric_median,
    _hardware_group,
    _nearest_indices,
)
from paper_pipeline.alignment import (
    estimate_truth_free_alignment,
    interpolate_poses,
    transform_poses,
)
from paper_pipeline.cases import load_estimator_input


def _regularize(covariance: np.ndarray, floor_m: float = 1.0) -> np.ndarray:
    covariance = 0.5 * (covariance + covariance.T)
    values, vectors = np.linalg.eigh(covariance)
    return (vectors * np.maximum(values, floor_m**2)) @ vectors.T


def _aggregate(
    points: np.ndarray,
    covariances: np.ndarray,
    center_mode: str,
    covariance_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if center_mode == "geometric":
        center = _geometric_median(points)
    elif center_mode == "coordinate":
        center = np.median(points, axis=0)
    elif center_mode == "information_mean":
        information = np.zeros((3, 3))
        rhs = np.zeros(3)
        for point, covariance in zip(points, covariances):
            inverse = np.linalg.inv(_regularize(covariance))
            information += inverse
            rhs += inverse @ point
        center = np.linalg.solve(information, rhs)
    else:
        raise ValueError(center_mode)

    deviations = points - center
    scatter = (
        deviations.T @ deviations / max(len(points) - 1, 1)
        if len(points) > 1
        else np.zeros((3, 3))
    )
    base = np.median(covariances, axis=0)
    if covariance_mode == "nonshrinking":
        covariance = base + scatter
    elif covariance_mode == "independence":
        covariance = (base + scatter) / max(len(points), 1)
    elif covariance_mode == "base_only":
        covariance = base
    else:
        raise ValueError(covariance_mode)
    return center, _regularize(covariance)


def _hierarchical_epoch(
    group_points: dict[str, list[np.ndarray]],
    group_covariances: dict[str, list[np.ndarray]],
    center_mode: str,
    covariance_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    group_centers = []
    group_uncertainties = []
    for group in sorted(group_points):
        center, covariance = _aggregate(
            np.asarray(group_points[group]),
            np.asarray(group_covariances[group]),
            center_mode,
            covariance_mode,
        )
        group_centers.append(center)
        group_uncertainties.append(covariance)
    return _aggregate(
        np.asarray(group_centers),
        np.asarray(group_uncertainties),
        center_mode,
        covariance_mode,
    )


def _metrics(estimates: np.ndarray, truth: np.ndarray, start: int) -> dict[str, float]:
    difference = estimates[start:, :3] - truth[start:, :3]
    error_3d = np.linalg.norm(difference, axis=1)
    error_xy = np.linalg.norm(difference[:, :2], axis=1)
    return {
        "rmse_3d_m": float(np.sqrt(np.mean(error_3d**2))),
        "rmse_xy_m": float(np.sqrt(np.mean(error_xy**2))),
        "median_3d_m": float(np.median(error_3d)),
        "p95_3d_m": float(np.percentile(error_3d, 95.0)),
        "maximum_3d_m": float(np.max(error_3d)),
    }


def _solution_label(path: Path) -> str:
    suffix = ".estimator_input.npz"
    return path.name[: -len(suffix)] if path.name.endswith(suffix) else path.stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance-s", type=float, default=0.2)
    parser.add_argument(
        "--require-all-groups",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()

    input_paths = sorted(args.case_root.glob("*.estimator_input.npz"))
    if len(input_paths) < 2:
        raise ValueError("at least two simultaneous GNSS solution files are required")
    datasets = [load_estimator_input(path) for path in input_paths]
    groups = [_hardware_group(path) for path in input_paths]
    labels = [_solution_label(path) for path in input_paths]
    group_names = sorted(set(groups))
    base = datasets[0]
    # Case construction crops the common KISS trajectory to each receiver's
    # availability interval.  Verify the overlapping trajectory instead of
    # incorrectly requiring identical array lengths.
    lidar_start = max(data.lidar_timestamps_s[0] for data in datasets)
    lidar_end = min(data.lidar_timestamps_s[-1] for data in datasets)
    audit_times = np.linspace(lidar_start, lidar_end, 100)
    base_audit = interpolate_poses(
        base.lidar_timestamps_s, base.lidar_poses_local, audit_times
    )
    for data in datasets[1:]:
        other_audit = interpolate_poses(
            data.lidar_timestamps_s, data.lidar_poses_local, audit_times
        )
        if not np.allclose(base_audit, other_audit, atol=1e-5):
            raise ValueError(
                "case files do not share the same overlapping LiDAR trajectory"
            )

    all_times = np.unique(
        np.concatenate([data.gnss_timestamps_s for data in datasets])
    )
    all_times = all_times[
        (all_times >= lidar_start) & (all_times <= lidar_end)
    ]
    retained_times: list[float] = []
    retained_group_points: list[dict[str, list[np.ndarray]]] = []
    retained_group_covariances: list[dict[str, list[np.ndarray]]] = []
    retained_solution_points: list[dict[str, np.ndarray]] = []
    retained_solution_covariances: list[dict[str, np.ndarray]] = []
    for timestamp in all_times:
        group_points: dict[str, list[np.ndarray]] = {}
        group_covariances: dict[str, list[np.ndarray]] = {}
        solution_points: dict[str, np.ndarray] = {}
        solution_covariances: dict[str, np.ndarray] = {}
        for data, group, label in zip(datasets, groups, labels):
            indices, differences = _nearest_indices(
                data.gnss_timestamps_s, np.asarray([timestamp])
            )
            if differences[0] > args.tolerance_s:
                continue
            index = int(indices[0])
            point = np.asarray(data.gnss_positions_enu_m[index], dtype=float)
            covariance = np.asarray(data.gnss_covariances_m2[index], dtype=float)
            group_points.setdefault(group, []).append(point)
            group_covariances.setdefault(group, []).append(covariance)
            solution_points[label] = point
            solution_covariances[label] = covariance
        enough = (
            len(group_points) == len(group_names)
            if args.require_all_groups
            else len(group_points) >= 2
        )
        if enough:
            retained_times.append(float(timestamp))
            retained_group_points.append(group_points)
            retained_group_covariances.append(group_covariances)
            retained_solution_points.append(solution_points)
            retained_solution_covariances.append(solution_covariances)
    times = np.asarray(retained_times)
    if len(times) < 100:
        raise ValueError(f"too few common receiver epochs: {len(times)}")

    streams: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    method_specs = {
        "proposed_hierarchical_geometric_nonshrinking": (
            "hierarchical",
            "geometric",
            "nonshrinking",
        ),
        "ablation_flat_geometric_nonshrinking": (
            "flat",
            "geometric",
            "nonshrinking",
        ),
        "ablation_hierarchical_coordinate_nonshrinking": (
            "hierarchical",
            "coordinate",
            "nonshrinking",
        ),
        "ablation_hierarchical_geometric_independence": (
            "hierarchical",
            "geometric",
            "independence",
        ),
        "baseline_information_mean": (
            "flat",
            "information_mean",
            "independence",
        ),
    }
    for name, (level, center_mode, covariance_mode) in method_specs.items():
        positions = []
        covariances = []
        for group_points, group_covariances in zip(
            retained_group_points, retained_group_covariances
        ):
            if level == "hierarchical":
                center, covariance = _hierarchical_epoch(
                    group_points,
                    group_covariances,
                    center_mode,
                    covariance_mode,
                )
            else:
                center, covariance = _aggregate(
                    np.asarray(
                        [
                            point
                            for group in sorted(group_points)
                            for point in group_points[group]
                        ]
                    ),
                    np.asarray(
                        [
                            covariance
                            for group in sorted(group_covariances)
                            for covariance in group_covariances[group]
                        ]
                    ),
                    center_mode,
                    covariance_mode,
                )
            positions.append(center)
            covariances.append(covariance)
        streams[name] = (np.asarray(positions), np.asarray(covariances))

    # A physical receiver is one vote.  A receiver that exposes several
    # constellation solutions is collapsed before it becomes a baseline.
    for group in group_names:
        positions = []
        covariances = []
        for group_points, group_covariances in zip(
            retained_group_points, retained_group_covariances
        ):
            center, covariance = _aggregate(
                np.asarray(group_points[group]),
                np.asarray(group_covariances[group]),
                "geometric",
                "nonshrinking",
            )
            positions.append(center)
            covariances.append(covariance)
        streams[f"single_physical_receiver__{group}"] = (
            np.asarray(positions),
            np.asarray(covariances),
        )

    # Raw-solution baselines are included only when a stream exists at every
    # retained epoch, avoiding an easier/different evaluation interval.
    for label in labels:
        if all(label in points for points in retained_solution_points):
            streams[f"single_raw_solution__{label}"] = (
                np.asarray([points[label] for points in retained_solution_points]),
                np.asarray(
                    [covariances[label] for covariances in retained_solution_covariances]
                ),
            )

    local_at_epochs = interpolate_poses(
        base.lidar_timestamps_s, base.lidar_poses_local, times
    )
    alignments = {}
    lidar_streams = {}
    for name, (positions, _) in streams.items():
        alignment = estimate_truth_free_alignment(
            base.lidar_timestamps_s,
            base.lidar_poses_local,
            times,
            positions,
            ransac_threshold_m=20.0,
            minimum_count=30,
            minimum_duration_s=60.0,
            random_seed=args.seed,
        )
        alignments[name] = alignment
        lidar_streams[name] = transform_poses(
            alignment.transform_enu_from_lidar, local_at_epochs
        )
    common_start = max(alignment.used_count for alignment in alignments.values())

    hidden_path = input_paths[0].with_name(
        input_paths[0].name.replace(
            ".estimator_input.npz", ".hidden_labels.npz"
        )
    )
    with np.load(hidden_path, allow_pickle=False) as hidden:
        truth_poses = interpolate_poses(
            np.asarray(hidden["truth_timestamps_s"], dtype=float),
            np.asarray(hidden["truth_poses_enu"], dtype=float),
            times,
        )
    truth = truth_poses[:, :3, 3]

    results: dict[str, object] = {}
    for name, (positions, covariances) in streams.items():
        families = ("direct", "huber", "cauchy")
        entry: dict[str, object] = {
            "alignment_used_count": int(alignments[name].used_count),
            "alignment_inlier_fraction": float(alignments[name].inlier_fraction),
            "gnss_only": _metrics(positions, truth, common_start),
            "kiss_icp_alignment_only": _metrics(
                lidar_streams[name][:, :3, 3], truth, common_start
            ),
            "fusion": {},
        }
        for family in families:
            started = time.perf_counter()
            graph, initial, _ = _build_graph(
                lidar_streams[name],
                positions,
                covariances,
                family,
                20,
                args.seed,
            )
            optimized = _optimize(graph, initial)
            estimates = np.asarray(
                [
                    np.asarray(
                        optimized.atPose3(_key(index)).translation(), dtype=float
                    )
                    for index in range(len(times))
                ]
            )
            entry["fusion"][family] = {
                **_metrics(estimates, truth, common_start),
                "runtime_s": float(time.perf_counter() - started),
            }
        results[name] = entry

    payload = {
        "evidentiary_status": "DEVELOPMENT_STRONG_SAME_EPOCH_PANEL",
        "truth_access_policy": (
            "Truth was loaded only after every estimator-side stream and "
            "truth-free alignment had been constructed."
        ),
        "receiver_solution_count": len(datasets),
        "physical_receiver_group_count": len(group_names),
        "physical_receiver_groups": group_names,
        "common_epoch_count": len(times),
        "common_evaluation_start_index": int(common_start),
        "common_evaluation_epoch_count": int(len(times) - common_start),
        "time_start_s": float(times[0]),
        "time_end_s": float(times[-1]),
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
