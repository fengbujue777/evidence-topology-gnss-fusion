"""Evaluate a contrast-identified correlated GNSS solution-stream factor.

All calibration, shrinkage selection, alignment, and trajectory optimization
are completed using estimator inputs only.  Hidden labels are opened once, at
the end, solely for scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from evidence_topology import topology_factor_graph as base
from evidence_topology.receiver_diversity_baselines import _metrics
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
from paper_pipeline.contrast_identified_factor import (
    condensed_gls_factor,
    fit_contrast_model,
)


def _build_common_panel(
    input_paths: list[Path], tolerance_s: float
) -> tuple[
    list[object],
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    datasets = [load_estimator_input(path) for path in input_paths]
    groups = [_hardware_group(path) for path in input_paths]
    start = max(data.lidar_timestamps_s[0] for data in datasets)
    end = min(data.lidar_timestamps_s[-1] for data in datasets)
    all_times = np.unique(
        np.concatenate([data.gnss_timestamps_s for data in datasets])
    )
    all_times = all_times[(all_times >= start) & (all_times <= end)]
    retained_times = []
    retained_points = []
    retained_covariances = []
    alignment_positions = []
    for timestamp in all_times:
        points = []
        covariances = []
        for data in datasets:
            indices, differences = _nearest_indices(
                data.gnss_timestamps_s, np.asarray([timestamp])
            )
            if differences[0] > tolerance_s:
                break
            index = int(indices[0])
            points.append(
                np.asarray(data.gnss_positions_enu_m[index], dtype=float)
            )
            covariances.append(
                np.asarray(data.gnss_covariances_m2[index], dtype=float)
            )
        if len(points) != len(datasets):
            continue
        group_points: dict[str, list[np.ndarray]] = {}
        group_covariances: dict[str, list[np.ndarray]] = {}
        for point, covariance, group in zip(points, covariances, groups):
            group_points.setdefault(group, []).append(point)
            group_covariances.setdefault(group, []).append(covariance)
        group_centers = []
        group_uncertainties = []
        for group in sorted(group_points):
            unique_points, unique_covariances = (
                base._deduplicate_exact_measurements(
                    group_points[group], group_covariances[group]
                )
            )
            center, covariance = base._aggregate(
                unique_points,
                unique_covariances,
                "geometric",
                "nonshrinking",
            )
            group_centers.append(center)
            group_uncertainties.append(covariance)
        alignment_position, _ = base._aggregate(
            np.asarray(group_centers),
            np.asarray(group_uncertainties),
            "geometric",
            "independence",
        )
        retained_times.append(timestamp)
        retained_points.append(points)
        retained_covariances.append(covariances)
        alignment_positions.append(alignment_position)
    return (
        datasets,
        groups,
        np.asarray(retained_times),
        np.asarray(retained_points),
        np.asarray(retained_covariances),
        np.asarray(alignment_positions),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-output", type=Path, required=True)
    parser.add_argument("--tolerance-s", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    input_paths = sorted(args.case_root.glob("*.estimator_input.npz"))
    if len(input_paths) < 2:
        raise ValueError("at least two solution streams are required")
    (
        datasets,
        groups,
        times,
        points,
        covariances,
        alignment_positions,
    ) = _build_common_panel(input_paths, args.tolerance_s)
    if len(times) < 100:
        raise ValueError(f"too few common epochs: {len(times)}")

    alignment = estimate_truth_free_alignment(
        datasets[0].lidar_timestamps_s,
        datasets[0].lidar_poses_local,
        times,
        alignment_positions,
        ransac_threshold_m=20.0,
        minimum_count=30,
        minimum_duration_s=60.0,
        random_seed=args.seed,
    )
    calibration_count = max(
        int(alignment.used_count), 30, int(np.ceil(0.25 * len(times)))
    )
    calibration_count = min(calibration_count, len(times) - 20)
    model = fit_contrast_model(
        points[:calibration_count],
        covariances[:calibration_count],
    )
    factor_positions = []
    factor_covariances = []
    for epoch_points in points:
        position, covariance = condensed_gls_factor(epoch_points, model)
        factor_positions.append(position)
        factor_covariances.append(covariance)
    factor_positions = np.asarray(factor_positions)
    factor_covariances = np.asarray(factor_covariances)

    regular_times = np.arange(times[0], times[-1] + 1e-9, 1.0)
    state_times = np.unique(np.concatenate([regular_times, times]))
    state_times = state_times[
        (state_times >= times[0]) & (state_times <= times[-1])
    ]
    measurement_indices = np.searchsorted(state_times, times)
    local_states = interpolate_poses(
        datasets[0].lidar_timestamps_s,
        datasets[0].lidar_poses_local,
        state_times,
    )
    lidar = transform_poses(
        alignment.transform_enu_from_lidar, local_states
    )

    trajectories = {"aligned_kiss_icp": lidar[:, :3, 3]}
    runtimes = {"aligned_kiss_icp": 0.0}
    for family in ("direct", "huber", "cauchy"):
        started = time.perf_counter()
        trajectory, backend_runtime = base._optimize_single(
            lidar,
            state_times,
            measurement_indices,
            factor_positions,
            factor_covariances,
            family,
        )
        trajectories[f"cicf__{family}"] = trajectory
        runtimes[f"cicf__{family}"] = (
            float(time.perf_counter() - started) + backend_runtime
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
        np.searchsorted(state_times, times[calibration_count])
    )
    metrics = {
        name: {
            **_metrics(trajectory, truth, evaluation_start),
            "runtime_s": runtimes[name],
        }
        for name, trajectory in trajectories.items()
    }
    payload = {
        "evidentiary_status": "EXPLORATORY_CONTRAST_IDENTIFIED_FACTOR",
        "truth_access_policy": (
            "Relative-bias calibration, contrast covariance fitting, "
            "shrinkage selection, alignment, and all trajectories were "
            "completed before hidden truth was opened."
        ),
        "input_streams": [path.name for path in input_paths],
        "physical_receiver_groups": groups,
        "common_epoch_count": int(len(times)),
        "calibration_epoch_count": int(calibration_count),
        "evaluation_epoch_count": int(len(state_times) - evaluation_start),
        "selected_shrinkage": model.shrinkage,
        "shrinkage_validation_nll": {
            str(key): value for key, value in model.validation_nll.items()
        },
        "joint_covariance_effective_rank": model.effective_rank,
        "joint_covariance_dimension": int(
            model.joint_covariance.shape[0]
        ),
        "relative_biases_m": model.relative_biases.tolist(),
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
        joint_covariance=model.joint_covariance,
        relative_biases_m=model.relative_biases,
        **{
            f"trajectory__{name}": trajectory
            for name, trajectory in trajectories.items()
        },
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
