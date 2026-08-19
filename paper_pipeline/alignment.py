"""Truth-free initialization of the LiDAR-local to ENU transformation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class AlignmentResult:
    transform_enu_from_lidar: np.ndarray
    used_count: int
    duration_s: float
    displacement_m: float
    heading_excitation_deg: float
    inlier_fraction: float
    residual_median_m: float
    parameter_covariance_se2: np.ndarray


def interpolate_poses(
    source_timestamps_s: np.ndarray,
    source_poses: np.ndarray,
    target_timestamps_s: np.ndarray,
) -> np.ndarray:
    source_timestamps_s = np.asarray(source_timestamps_s, dtype=float)
    target_timestamps_s = np.asarray(target_timestamps_s, dtype=float)
    source_poses = np.asarray(source_poses, dtype=float)
    if target_timestamps_s[0] < source_timestamps_s[0] or target_timestamps_s[-1] > source_timestamps_s[-1]:
        raise ValueError("target timestamps extend beyond the pose trajectory")
    translations = np.column_stack(
        [
            np.interp(target_timestamps_s, source_timestamps_s, source_poses[:, axis, 3])
            for axis in range(3)
        ]
    )
    rotations = Rotation.from_matrix(source_poses[:, :3, :3])
    interpolated_rotations = Slerp(source_timestamps_s, rotations)(target_timestamps_s)
    result = np.repeat(np.eye(4)[None, :, :], len(target_timestamps_s), axis=0)
    result[:, :3, :3] = interpolated_rotations.as_matrix()
    result[:, :3, 3] = translations
    return result


def _heading_excitation_deg(points: np.ndarray) -> float:
    differences = np.diff(points[:, :2], axis=0)
    valid = np.linalg.norm(differences, axis=1) > 0.1
    if np.count_nonzero(valid) < 2:
        return 0.0
    headings = np.unwrap(np.arctan2(differences[valid, 1], differences[valid, 0]))
    return float(np.degrees(np.max(headings) - np.min(headings)))


def initialization_prefix(
    timestamps_s: np.ndarray,
    local_positions_m: np.ndarray,
    minimum_count: int = 3,
    minimum_duration_s: float = 30.0,
    minimum_displacement_m: float = 30.0,
    minimum_heading_excitation_deg: float = 15.0,
) -> int:
    if minimum_count < 3:
        raise ValueError("minimum_count must be at least 3")
    for count in range(minimum_count, len(timestamps_s) + 1):
        duration = timestamps_s[count - 1] - timestamps_s[0]
        displacement = np.linalg.norm(
            local_positions_m[count - 1, :2] - local_positions_m[0, :2]
        )
        excitation = _heading_excitation_deg(local_positions_m[:count])
        if (
            duration >= minimum_duration_s
            and displacement >= minimum_displacement_m
            and excitation >= minimum_heading_excitation_deg
        ):
            return count
    raise ValueError(
        "trajectory never satisfies the registered truth-free alignment excitation"
    )


def _fit_se2(source_xy: np.ndarray, target_xy: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = weights / np.sum(weights)
    source_center = np.sum(source_xy * weights[:, None], axis=0)
    target_center = np.sum(target_xy * weights[:, None], axis=0)
    source_zero = source_xy - source_center
    target_zero = target_xy - target_center
    covariance = (source_zero * weights[:, None]).T @ target_zero
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def _refine_se3(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    initial_transform: np.ndarray,
    vertical_weight: float,
    robust_scale_m: float,
) -> np.ndarray:
    """Robust anisotropic SE(3) trajectory alignment."""

    initial = np.concatenate(
        [
            Rotation.from_matrix(initial_transform[:3, :3]).as_rotvec(),
            initial_transform[:3, 3],
        ]
    )
    axis_scale = np.array(
        [1.0, 1.0, np.sqrt(max(vertical_weight, 1e-6))]
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        rotation = Rotation.from_rotvec(parameters[:3]).as_matrix()
        predicted = (
            rotation @ source_xyz.T
        ).T + parameters[3:6]
        return ((predicted - target_xyz) * axis_scale).reshape(-1)

    optimized = least_squares(
        residual,
        initial,
        loss="soft_l1",
        f_scale=max(float(robust_scale_m), 0.25),
        max_nfev=300,
    )
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(
        optimized.x[:3]
    ).as_matrix()
    transform[:3, 3] = optimized.x[3:6]
    return transform


def estimate_truth_free_alignment(
    lidar_timestamps_s: np.ndarray,
    lidar_poses_local: np.ndarray,
    gnss_timestamps_s: np.ndarray,
    gnss_positions_enu_m: np.ndarray,
    ransac_threshold_m: float = 3.0,
    ransac_trials: int = 500,
    random_seed: int = 20260726,
    minimum_count: int = 3,
    minimum_duration_s: float = 30.0,
    minimum_displacement_m: float = 30.0,
    minimum_heading_excitation_deg: float = 15.0,
    bootstrap_trials: int = 200,
    bootstrap_block_epochs: int = 5,
    alignment_model: str = "se2",
    se3_vertical_weight: float = 0.1,
) -> AlignmentResult:
    matched_lidar = interpolate_poses(
        lidar_timestamps_s, lidar_poses_local, gnss_timestamps_s
    )
    local_positions = matched_lidar[:, :3, 3]
    count = initialization_prefix(
        gnss_timestamps_s,
        local_positions,
        minimum_count=minimum_count,
        minimum_duration_s=minimum_duration_s,
        minimum_displacement_m=minimum_displacement_m,
        minimum_heading_excitation_deg=minimum_heading_excitation_deg,
    )
    source = local_positions[:count, :2]
    target = np.asarray(gnss_positions_enu_m[:count, :2], dtype=float)
    rng = np.random.default_rng(random_seed)

    best_inliers = np.zeros(count, dtype=bool)
    best_median = np.inf
    for _ in range(ransac_trials):
        indices = rng.choice(count, 2, replace=False)
        source_vector = source[indices[1]] - source[indices[0]]
        target_vector = target[indices[1]] - target[indices[0]]
        if np.linalg.norm(source_vector) < 3.0 or np.linalg.norm(target_vector) < 3.0:
            continue
        yaw = np.arctan2(target_vector[1], target_vector[0]) - np.arctan2(
            source_vector[1], source_vector[0]
        )
        rotation = np.array(
            [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
        )
        translation = target[indices[0]] - rotation @ source[indices[0]]
        residual = np.linalg.norm((rotation @ source.T).T + translation - target, axis=1)
        inliers = residual <= ransac_threshold_m
        median = float(np.median(residual[inliers])) if np.any(inliers) else np.inf
        if (np.count_nonzero(inliers), -median) > (
            np.count_nonzero(best_inliers),
            -best_median,
        ):
            best_inliers = inliers
            best_median = median
    if np.count_nonzero(best_inliers) < max(3, count // 3):
        raise ValueError("truth-free GNSS/LiDAR alignment has too few RANSAC inliers")

    weights = best_inliers.astype(float)
    for _ in range(5):
        rotation_2d, translation_2d = _fit_se2(source, target, weights)
        residual = np.linalg.norm(
            (rotation_2d @ source.T).T + translation_2d - target, axis=1
        )
        huber = np.minimum(1.0, ransac_threshold_m / np.maximum(residual, 1e-9))
        weights = best_inliers.astype(float) * huber
    yaw = np.arctan2(rotation_2d[1, 0], rotation_2d[0, 0])
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    transform[:2, 3] = translation_2d
    rotated_z = (transform[:3, :3] @ local_positions[:count].T).T[:, 2]
    transform[2, 3] = np.median(
        gnss_positions_enu_m[:count, 2] - rotated_z
    )
    if alignment_model == "se3":
        transform = _refine_se3(
            local_positions[:count][best_inliers],
            np.asarray(gnss_positions_enu_m[:count], dtype=float)[
                best_inliers
            ],
            transform,
            se3_vertical_weight,
            ransac_threshold_m,
        )
    elif alignment_model != "se2":
        raise ValueError(f"unsupported alignment_model: {alignment_model}")
    inlier_indices = np.flatnonzero(best_inliers)
    bootstrap_rng = np.random.default_rng(random_seed + 1)
    bootstrap_parameters: list[np.ndarray] = []
    block_length = max(1, min(bootstrap_block_epochs, len(inlier_indices)))
    nominal_source = source[inlier_indices]
    nominal_source_center = np.mean(nominal_source, axis=0)
    nominal_spread = float(
        np.mean(
            np.sum(
                (nominal_source - nominal_source_center) ** 2,
                axis=1,
            )
        )
    )
    for _ in range(max(bootstrap_trials, 0)):
        selected: list[int] = []
        while len(selected) < len(inlier_indices):
            block_start = int(
                bootstrap_rng.integers(
                    0, max(len(inlier_indices) - block_length + 1, 1)
                )
            )
            selected.extend(
                inlier_indices[
                    block_start : block_start + block_length
                ].tolist()
            )
        sample_indices = np.asarray(
            selected[: len(inlier_indices)], dtype=int
        )
        sample_source = source[sample_indices]
        sample_source_center = np.mean(sample_source, axis=0)
        sample_spread = float(
            np.mean(
                np.sum(
                    (sample_source - sample_source_center) ** 2,
                    axis=1,
                )
            )
        )
        if sample_spread < 0.25 * max(nominal_spread, 1e-6):
            continue
        try:
            sample_rotation, sample_translation = _fit_se2(
                source[sample_indices],
                target[sample_indices],
                np.ones(len(sample_indices)),
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        sample_yaw = np.arctan2(
            sample_rotation[1, 0], sample_rotation[0, 0]
        )
        sample_rotated_z = (
            Rotation.from_euler("z", sample_yaw).as_matrix()
            @ local_positions[sample_indices].T
        ).T[:, 2]
        sample_z = np.median(
            gnss_positions_enu_m[sample_indices, 2] - sample_rotated_z
        )
        yaw_delta = np.arctan2(
            np.sin(sample_yaw - yaw), np.cos(sample_yaw - yaw)
        )
        if abs(yaw_delta) > np.deg2rad(20.0):
            continue
        bootstrap_parameters.append(
            np.array(
                [
                    sample_translation[0],
                    sample_translation[1],
                    sample_z,
                    yaw + yaw_delta,
                ]
            )
        )
    if len(bootstrap_parameters) >= 5:
        parameter_covariance = np.cov(
            np.asarray(bootstrap_parameters).T, ddof=1
        )
    else:
        parameter_covariance = np.diag(
            [0.5**2, 0.5**2, 0.8**2, np.deg2rad(5.0) ** 2]
        )
    parameter_covariance = 0.5 * (
        parameter_covariance + parameter_covariance.T
    )
    covariance_values, covariance_vectors = np.linalg.eigh(
        parameter_covariance
    )
    parameter_covariance = (
        covariance_vectors
        * np.maximum(covariance_values, 1e-8)
    ) @ covariance_vectors.T
    return AlignmentResult(
        transform,
        count,
        float(gnss_timestamps_s[count - 1] - gnss_timestamps_s[0]),
        float(np.linalg.norm(source[count - 1] - source[0])),
        _heading_excitation_deg(local_positions[:count]),
        float(np.mean(best_inliers)),
        float(best_median),
        parameter_covariance,
    )


def transform_poses(transform: np.ndarray, poses: np.ndarray) -> np.ndarray:
    return np.einsum("ij,njk->nik", transform, poses)
