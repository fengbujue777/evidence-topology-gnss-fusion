"""Information-preserving common-mode/contrast GNSS factor screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from paper_pipeline.alignment import (
    estimate_truth_free_alignment,
    interpolate_poses,
    transform_poses,
)
from paper_pipeline.cases import load_estimator_input


FROZEN_FIXED_LAG_BASELINES = {
    "urbannav_hk_medium_urban_1__novatel_flexpak6__real_spp": 5.7629865225171955,
    "urbannav_hk_medium_urban_1__ublox_f9p__real_spp": 8.363387378907813,
    "urbannav_hk_medium_urban_1__ublox_m8t_gc__real_spp": 11.48769872659415,
    "urbannav_hk_medium_urban_1__ublox_m8t_gej__real_spp": 17.59205664615761,
    "urbannav_hk_medium_urban_1__ublox_m8t_gr__real_spp": 25.666736857919837,
}


def _gtsam():
    import gtsam

    return gtsam


def _key(index: int) -> int:
    return _gtsam().symbol("x", index)


def _regularize_covariance(
    covariance: np.ndarray, floor_m: float = 1.0
) -> np.ndarray:
    covariance = 0.5 * (
        np.asarray(covariance, dtype=float)
        + np.asarray(covariance, dtype=float).T
    )
    values, vectors = np.linalg.eigh(covariance)
    return (vectors * np.maximum(values, floor_m**2)) @ vectors.T


def _inverse_sqrt(covariance: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(
        0.5 * (covariance + covariance.T)
    )
    return (vectors * (1.0 / np.sqrt(np.maximum(values, 1e-9)))) @ vectors.T


def _robust_noise(dimension: int, family: str):
    gtsam = _gtsam()
    base = gtsam.noiseModel.Isotropic.Sigma(dimension, 1.0)
    if family == "direct":
        return base
    if family == "huber":
        estimator = gtsam.noiseModel.mEstimator.Huber.Create(1.345)
    elif family == "cauchy":
        estimator = gtsam.noiseModel.mEstimator.Cauchy.Create(2.5)
    else:
        raise ValueError(f"unsupported robust family: {family}")
    return gtsam.noiseModel.Robust.Create(estimator, base)


def _gps_noise(
    covariance: np.ndarray,
    family: str,
    *,
    floor_m: float = 1.0,
):
    gtsam = _gtsam()
    base = gtsam.noiseModel.Gaussian.Covariance(
        _regularize_covariance(covariance, floor_m=floor_m)
    )
    if family == "direct":
        return base
    if family == "huber":
        estimator = gtsam.noiseModel.mEstimator.Huber.Create(1.345)
    elif family == "cauchy":
        estimator = gtsam.noiseModel.mEstimator.Cauchy.Create(2.5)
    else:
        raise ValueError(f"unsupported family: {family}")
    return gtsam.noiseModel.Robust.Create(estimator, base)


def _scalar_mode_factor(
    indices: np.ndarray,
    positions: np.ndarray,
    mode_row: np.ndarray,
):
    """Whitened scalar mode over a block of pose translations."""

    gtsam = _gtsam()
    indices = np.asarray(indices, dtype=int).copy()
    positions = np.asarray(positions, dtype=float).copy()
    mode_row = np.asarray(mode_row, dtype=float).reshape(-1).copy()
    keys = [_key(int(index)) for index in indices]
    block_size = len(indices)
    if mode_row.shape != (3 * block_size,):
        raise ValueError("mode row has an invalid shape")

    def error_function(this, values, jacobians):
        residual = 0.0
        for local_index, key in enumerate(keys):
            pose = values.atPose3(key)
            row = mode_row[
                3 * local_index : 3 * (local_index + 1)
            ]
            residual += row @ (
                np.asarray(pose.translation(), dtype=float)
                - positions[local_index]
            )
            if jacobians is not None:
                jacobian = np.zeros((1, 6), dtype=float)
                jacobian[0, 3:6] = (
                    row @ np.asarray(pose.rotation().matrix(), dtype=float)
                )
                jacobians[local_index] = jacobian
        return np.asarray([residual], dtype=float)

    return gtsam.CustomFactor(
        _robust_noise(1, "cauchy"), keys, error_function
    )


def _factorized_gnss(
    graph,
    gnss_positions: np.ndarray,
    gnss_covariances: np.ndarray,
    window_epochs: int,
    random_control: bool,
    random_seed: int,
) -> tuple[int, int]:
    rng = np.random.default_rng(random_seed)
    count = len(gnss_positions)
    common_factors = 0
    contrast_factors = 0
    for start in range(0, count, window_epochs):
        end = min(start + window_epochs, count)
        if end - start < 2:
            graph.add(
                _gtsam().GPSFactor(
                    _key(start),
                    gnss_positions[start],
                    _gps_noise(gnss_covariances[start], "cauchy"),
                )
            )
            common_factors += 3
            continue
        indices = np.arange(start, end, dtype=int)
        block_size = len(indices)
        whitener = np.zeros((3 * block_size, 3 * block_size))
        for local_index, index in enumerate(indices):
            block = _inverse_sqrt(
                _regularize_covariance(gnss_covariances[index])
            )
            sl = slice(3 * local_index, 3 * (local_index + 1))
            whitener[sl, sl] = block

        if random_control:
            temporal = rng.normal(size=block_size)
            temporal /= max(np.linalg.norm(temporal), 1e-12)
        else:
            temporal = np.ones(block_size, dtype=float)
        bias_basis = np.kron(temporal[:, None], np.eye(3))
        q_basis, _ = np.linalg.qr(
            whitener @ bias_basis, mode="complete"
        )
        transformed_rows = q_basis.T @ whitener
        for mode_index, row in enumerate(transformed_rows):
            graph.add(
                _scalar_mode_factor(
                    indices,
                    gnss_positions[indices],
                    row,
                )
            )
            if mode_index < 3:
                common_factors += 1
            else:
                contrast_factors += 1
    return common_factors, contrast_factors


def _build_graph(
    lidar_poses_enu: np.ndarray,
    gnss_positions: np.ndarray,
    gnss_covariances: np.ndarray,
    variant: str,
    window_epochs: int,
    random_seed: int,
):
    gtsam = _gtsam()
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    rotation_sigma = np.deg2rad(0.5)
    odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.asarray(
            [
                rotation_sigma,
                rotation_sigma,
                rotation_sigma,
                0.10,
                0.10,
                0.10,
            ]
        )
    )
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.asarray(
            [
                np.deg2rad(5.0),
                np.deg2rad(5.0),
                np.deg2rad(5.0),
                0.5,
                0.5,
                0.8,
            ]
        )
    )
    for index, matrix in enumerate(lidar_poses_enu):
        initial.insert(_key(index), gtsam.Pose3(matrix))
    graph.add(
        gtsam.PriorFactorPose3(
            _key(0), gtsam.Pose3(lidar_poses_enu[0]), prior_noise
        )
    )
    for index in range(1, len(lidar_poses_enu)):
        relative = (
            np.linalg.inv(lidar_poses_enu[index - 1])
            @ lidar_poses_enu[index]
        )
        graph.add(
            gtsam.BetweenFactorPose3(
                _key(index - 1),
                _key(index),
                gtsam.Pose3(relative),
                odometry_noise,
            )
        )

    counts = {"common_modes": 0, "contrast_modes": 0}
    if variant in {"direct", "huber", "cauchy"}:
        for index, (position, covariance) in enumerate(
            zip(gnss_positions, gnss_covariances)
        ):
            graph.add(
                gtsam.GPSFactor(
                    _key(index),
                    position,
                    _gps_noise(covariance, variant),
                )
            )
    elif variant in {"common_contrast", "random_control"}:
        common, contrast = _factorized_gnss(
            graph,
            gnss_positions,
            gnss_covariances,
            window_epochs,
            variant == "random_control",
            random_seed,
        )
        counts = {"common_modes": common, "contrast_modes": contrast}
    else:
        raise ValueError(f"unsupported variant: {variant}")
    return graph, initial, counts


def _optimize(graph, initial):
    gtsam = _gtsam()
    parameters = gtsam.LevenbergMarquardtParams()
    parameters.setMaxIterations(30)
    parameters.setRelativeErrorTol(1e-6)
    parameters.setAbsoluteErrorTol(1e-6)
    return gtsam.LevenbergMarquardtOptimizer(
        graph, initial, parameters
    ).optimize()


def _evaluate(
    result,
    count: int,
    truth_positions: np.ndarray,
    start: int,
) -> tuple[float, np.ndarray]:
    estimates = np.asarray(
        [
            np.asarray(result.atPose3(_key(index)).translation(), dtype=float)
            for index in range(count)
        ]
    )
    error = np.linalg.norm(estimates[start:] - truth_positions[start:], axis=1)
    return float(np.sqrt(np.mean(error**2))), estimates


def run_case(
    input_path: Path,
    hidden_path: Path,
    output_directory: Path,
    window_epochs: int,
    random_seed: int,
) -> dict[str, object]:
    data = load_estimator_input(input_path)
    with np.load(hidden_path, allow_pickle=False) as hidden:
        truth_timestamps_s = np.asarray(
            hidden["truth_timestamps_s"], dtype=float
        )
        truth_poses = np.asarray(hidden["truth_poses_enu"], dtype=float)
    alignment = estimate_truth_free_alignment(
        data.lidar_timestamps_s,
        data.lidar_poses_local,
        data.gnss_timestamps_s,
        data.gnss_positions_enu_m,
        ransac_threshold_m=20.0,
        minimum_count=30,
        minimum_duration_s=60.0,
    )
    lidar_poses_enu = transform_poses(
        alignment.transform_enu_from_lidar, data.lidar_poses_local
    )
    lidar_at_gnss = interpolate_poses(
        data.lidar_timestamps_s,
        lidar_poses_enu,
        data.gnss_timestamps_s,
    )
    truth_at_gnss = interpolate_poses(
        truth_timestamps_s, truth_poses, data.gnss_timestamps_s
    )[:, :3, 3]

    case_id = input_path.name.removesuffix(".estimator_input.npz")
    row: dict[str, object] = {
        "case_id": case_id,
        "epochs": len(data.gnss_timestamps_s),
        "alignment_used_count": alignment.used_count,
        "frozen_fixed_lag_best_ate_m": FROZEN_FIXED_LAG_BASELINES[case_id],
        "variants": {},
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    for variant in (
        "direct",
        "huber",
        "cauchy",
        "common_contrast",
        "random_control",
    ):
        graph, initial, counts = _build_graph(
            lidar_at_gnss,
            data.gnss_positions_enu_m,
            data.gnss_covariances_m2,
            variant,
            window_epochs,
            random_seed,
        )
        started = time.perf_counter()
        result = _optimize(graph, initial)
        elapsed_s = time.perf_counter() - started
        ate, estimates = _evaluate(
            result,
            len(data.gnss_timestamps_s),
            truth_at_gnss,
            alignment.used_count,
        )
        row["variants"][variant] = {
            "ate_rmse_m": ate,
            "runtime_s": elapsed_s,
            "factor_count": graph.size(),
            **counts,
        }
        np.savez_compressed(
            output_directory / f"{case_id}__{variant}.npz",
            timestamps_s=data.gnss_timestamps_s,
            positions_enu_m=estimates,
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--window-epochs", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20260727)
    args = parser.parse_args()

    rows = []
    for input_path in sorted(args.case_root.glob("*.estimator_input.npz")):
        hidden_path = input_path.with_name(
            input_path.name.replace(
                ".estimator_input.npz", ".hidden_labels.npz"
            )
        )
        row = run_case(
            input_path,
            hidden_path,
            args.output_root / "trajectories",
            args.window_epochs,
            args.random_seed,
        )
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    conventional_gains = []
    random_gains = []
    frozen_gains = []
    for row in rows:
        candidate = row["variants"]["common_contrast"]["ate_rmse_m"]
        conventional = min(
            row["variants"][name]["ate_rmse_m"]
            for name in ("direct", "huber", "cauchy")
        )
        random_control = row["variants"]["random_control"]["ate_rmse_m"]
        frozen = row["frozen_fixed_lag_best_ate_m"]
        conventional_gains.append(conventional - candidate)
        random_gains.append(random_control - candidate)
        frozen_gains.append(frozen - candidate)
    verdict = {
        "candidate_vs_batch_conventional": {
            "wins": int(
                np.count_nonzero(np.asarray(conventional_gains) > 0.0)
            ),
            "mean_gain_m": float(np.mean(conventional_gains)),
            "minimum_gain_m": float(np.min(conventional_gains)),
        },
        "candidate_vs_random_mode_control": {
            "wins": int(np.count_nonzero(np.asarray(random_gains) > 0.0)),
            "mean_gain_m": float(np.mean(random_gains)),
            "minimum_gain_m": float(np.min(random_gains)),
        },
        "candidate_vs_frozen_fixed_lag": {
            "wins": int(np.count_nonzero(np.asarray(frozen_gains) > 0.0)),
            "mean_gain_m": float(np.mean(frozen_gains)),
            "minimum_gain_m": float(np.min(frozen_gains)),
        },
    }
    passed = all(
        item["wins"] >= 4 and item["mean_gain_m"] > 0.0
        for item in verdict.values()
    )
    payload = {
        "status": "DEVELOPMENT_ONLY_FROZEN_INFORMATION_PRESERVING_SCREEN",
        "window_epochs": args.window_epochs,
        "random_seed": args.random_seed,
        "rows": rows,
        "verdict": verdict,
        "passed": passed,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "screen.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"verdict": verdict, "passed": passed}, indent=2))


if __name__ == "__main__":
    main()
