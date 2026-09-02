"""Standard covariance-intersection baselines for a frozen QM-TAFC panel.

The script uses exactly the same common epochs, truth-free alignment prefix,
KISS-ICP trajectory, GTSAM graph, covariance floor, and hidden-truth scoring
policy as the frozen panel.  It adds two classical baselines:

* global_ci: all solution streams are conservatively fused as one unknown-
  correlation evidence unit;
* provenance_ci: solution streams are fused by CI inside each known physical
  receiver, while different physical receivers remain separate factors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from evidence_topology import topology_factor_graph as base
from evidence_topology.receiver_diversity_baselines import (
    _metrics,
    _regularize,
)
from evidence_topology.receiver_consensus import (
    _hardware_group,
    _nearest_indices,
)
from paper_pipeline.alignment import (
    estimate_truth_free_alignment,
    interpolate_poses,
    transform_poses,
)
from paper_pipeline.cases import load_estimator_input


def _deduplicate(
    points: np.ndarray, covariances: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    signatures = np.column_stack(
        [
            np.asarray(points, dtype=float),
            np.asarray(covariances, dtype=float).reshape(len(points), -1),
        ]
    )
    _, indices = np.unique(signatures, axis=0, return_index=True)
    indices = np.sort(indices)
    return np.asarray(points)[indices], np.asarray(covariances)[indices]


def _covariance_intersection(
    points: np.ndarray, covariances: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points, covariances = _deduplicate(points, covariances)
    covariances = np.asarray(
        [_regularize(covariance) for covariance in covariances]
    )
    inverses = np.asarray(
        [np.linalg.inv(covariance) for covariance in covariances]
    )
    count = len(points)
    if count == 1:
        return points[0], covariances[0], np.ones(1)

    def objective(weights: np.ndarray) -> float:
        information = np.einsum("i,ijk->jk", weights, inverses)
        sign, logdet = np.linalg.slogdet(information)
        return float(-logdet) if sign > 0 else 1e12

    result = minimize(
        objective,
        np.full(count, 1.0 / count),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints={
            "type": "eq",
            "fun": lambda weights: float(np.sum(weights) - 1.0),
        },
        options={"ftol": 1e-10, "maxiter": 200},
    )
    if not result.success:
        weights = np.full(count, 1.0 / count)
    else:
        weights = np.maximum(np.asarray(result.x, dtype=float), 0.0)
        weights /= np.sum(weights)
    information = np.einsum("i,ijk->jk", weights, inverses)
    covariance = np.linalg.inv(information)
    rhs = np.sum(
        [
            weight * inverse @ point
            for weight, inverse, point in zip(
                weights, inverses, points
            )
        ],
        axis=0,
    )
    center = covariance @ rhs
    return center, _regularize(covariance), weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-output", type=Path, required=True)
    parser.add_argument("--tolerance-s", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--epoch-policy",
        choices=("all_sources", "all_streams"),
        default="all_sources",
    )
    args = parser.parse_args()

    input_paths = sorted(args.case_root.glob("*.estimator_input.npz"))
    datasets = [load_estimator_input(path) for path in input_paths]
    groups = [_hardware_group(path) for path in input_paths]
    group_names = sorted(set(groups))
    if len(group_names) < 2:
        raise ValueError("CI panel requires at least two physical receivers")

    start = max(data.lidar_timestamps_s[0] for data in datasets)
    end = min(data.lidar_timestamps_s[-1] for data in datasets)
    base_data = datasets[0]
    all_times = np.unique(
        np.concatenate([data.gnss_timestamps_s for data in datasets])
    )
    all_times = all_times[(all_times >= start) & (all_times <= end)]

    times = []
    epoch_points: list[np.ndarray] = []
    epoch_covariances: list[np.ndarray] = []
    epoch_group_points: list[dict[str, np.ndarray]] = []
    epoch_group_covariances: list[dict[str, np.ndarray]] = []
    alignment_group_positions: list[np.ndarray] = []
    for timestamp in all_times:
        raw_points: dict[str, list[np.ndarray]] = {}
        raw_covariances: dict[str, list[np.ndarray]] = {}
        flat_points = []
        flat_covariances = []
        for data, group in zip(datasets, groups):
            indices, differences = _nearest_indices(
                data.gnss_timestamps_s, np.asarray([timestamp])
            )
            if differences[0] > args.tolerance_s:
                continue
            index = int(indices[0])
            point = np.asarray(
                data.gnss_positions_enu_m[index], dtype=float
            )
            covariance = np.asarray(
                data.gnss_covariances_m2[index], dtype=float
            )
            raw_points.setdefault(group, []).append(point)
            raw_covariances.setdefault(group, []).append(covariance)
            flat_points.append(point)
            flat_covariances.append(covariance)
        if (
            args.epoch_policy == "all_streams"
            and len(flat_points) != len(datasets)
        ):
            continue
        if (
            args.epoch_policy == "all_sources"
            and len(raw_points) != len(group_names)
        ):
            continue
        group_points = {}
        group_covariances = {}
        for group in group_names:
            unique_points, unique_covariances = _deduplicate(
                np.asarray(raw_points[group]),
                np.asarray(raw_covariances[group]),
            )
            (
                group_points[group],
                group_covariances[group],
                _,
            ) = _covariance_intersection(
                unique_points,
                unique_covariances,
            )
        times.append(timestamp)
        epoch_points.append(np.asarray(flat_points))
        epoch_covariances.append(np.asarray(flat_covariances))
        epoch_group_points.append(group_points)
        epoch_group_covariances.append(group_covariances)
        alignment_group_positions.append(
            base._canonical_alignment_reference(
                raw_points, raw_covariances
            )
        )

    times = np.asarray(times, dtype=float)
    if len(times) < 100:
        raise ValueError(f"too few common epochs: {len(times)}")

    global_positions = []
    global_covariances = []
    global_active_weight_counts = []
    for points, covariances in zip(epoch_points, epoch_covariances):
        position, covariance, weights = _covariance_intersection(
            points, covariances
        )
        global_positions.append(position)
        global_covariances.append(covariance)
        global_active_weight_counts.append(
            int(np.count_nonzero(weights > 1e-4))
        )
    global_positions = np.asarray(global_positions)
    global_covariances = np.asarray(global_covariances)

    alignment_group_positions = np.asarray(alignment_group_positions)

    alignment = estimate_truth_free_alignment(
        base_data.lidar_timestamps_s,
        base_data.lidar_poses_local,
        times,
        alignment_group_positions,
        ransac_threshold_m=20.0,
        minimum_count=30,
        minimum_duration_s=60.0,
        random_seed=args.seed,
    )
    regular_times = np.arange(times[0], times[-1] + 1e-9, 1.0)
    state_times = np.unique(np.concatenate([regular_times, times]))
    state_times = state_times[
        (state_times >= times[0]) & (state_times <= times[-1])
    ]
    measurement_indices = np.searchsorted(state_times, times)
    local_states = interpolate_poses(
        base_data.lidar_timestamps_s,
        base_data.lidar_poses_local,
        state_times,
    )
    lidar = transform_poses(
        alignment.transform_enu_from_lidar, local_states
    )

    trajectories = {}
    runtimes = {}
    for family in ("direct", "huber", "cauchy"):
        name = f"global_ci__{family}"
        trajectories[name], runtimes[name] = base._optimize_single(
            lidar,
            state_times,
            measurement_indices,
            global_positions,
            global_covariances,
            family,
        )
        name = f"provenance_ci__{family}"
        trajectories[name], runtimes[name] = base._optimize_grouped(
            lidar,
            state_times,
            measurement_indices,
            epoch_group_points,
            epoch_group_covariances,
            family,
        )

    hidden_path = input_paths[0].with_name(
        input_paths[0].name.replace(
            ".estimator_input.npz", ".hidden_labels.npz"
        )
    )
    with np.load(hidden_path, allow_pickle=False) as hidden:
        truth = interpolate_poses(
            np.asarray(hidden["truth_timestamps_s"], dtype=float),
            np.asarray(hidden["truth_poses_enu"], dtype=float),
            state_times,
        )[:, :3, 3]
    evaluation_start = int(
        np.searchsorted(state_times, times[alignment.used_count])
    )
    metrics = {
        name: {
            **_metrics(trajectory, truth, evaluation_start),
            "runtime_s": runtimes[name],
        }
        for name, trajectory in trajectories.items()
    }
    payload = {
        "evidentiary_status": "POST_FROZEN_STRONG_BASELINE",
        "epoch_policy": args.epoch_policy,
        "truth_access_policy": (
            "All CI weights, alignment, and trajectories were completed "
            "before hidden truth was opened."
        ),
        "physical_receiver_groups": group_names,
        "common_epoch_count": len(times),
        "evaluation_epoch_count": len(state_times) - evaluation_start,
        "alignment_prefix_count": alignment.used_count,
        "global_ci_active_weight_count": {
            "median": float(np.median(global_active_weight_counts)),
            "minimum": int(np.min(global_active_weight_counts)),
            "maximum": int(np.max(global_active_weight_counts)),
        },
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        args.trajectory_output,
        timestamps_s=state_times,
        truth_positions_enu_m=truth,
        **{
            f"trajectory__{name}": trajectory
            for name, trajectory in trajectories.items()
        },
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
