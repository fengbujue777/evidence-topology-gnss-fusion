from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _strictly_increasing(values: np.ndarray) -> bool:
    return len(values) < 2 or bool(np.all(np.diff(values) > 0))


@dataclass(frozen=True)
class EstimatorInput:
    """Inputs that an estimator is allowed to read.

    Ground truth and injected-fault labels deliberately do not appear in this
    type.  The final runner serializes this object separately from
    :class:`EvaluationTruth`.
    """

    lidar_timestamps_s: np.ndarray
    lidar_poses_local: np.ndarray
    gnss_timestamps_s: np.ndarray
    gnss_positions_enu_m: np.ndarray
    gnss_covariances_m2: np.ndarray

    def validate(self) -> None:
        n_lidar = len(self.lidar_timestamps_s)
        n_gnss = len(self.gnss_timestamps_s)
        if self.lidar_poses_local.shape != (n_lidar, 4, 4):
            raise ValueError("lidar_poses_local must have shape (N, 4, 4)")
        if self.gnss_positions_enu_m.shape != (n_gnss, 3):
            raise ValueError("gnss_positions_enu_m must have shape (M, 3)")
        if self.gnss_covariances_m2.shape != (n_gnss, 3, 3):
            raise ValueError("gnss_covariances_m2 must have shape (M, 3, 3)")
        if not _strictly_increasing(self.lidar_timestamps_s):
            raise ValueError("LiDAR timestamps are not strictly increasing")
        if not _strictly_increasing(self.gnss_timestamps_s):
            raise ValueError("GNSS timestamps are not strictly increasing")
        if not np.all(np.isfinite(self.lidar_poses_local)):
            raise ValueError("LiDAR poses contain non-finite values")
        if not np.all(np.isfinite(self.gnss_positions_enu_m)):
            raise ValueError("GNSS positions contain non-finite values")
        if np.any(np.linalg.eigvalsh(self.gnss_covariances_m2) <= 0):
            raise ValueError("GNSS covariance must be positive definite")


@dataclass(frozen=True)
class EvaluationTruth:
    """Evaluation-only data, unavailable to estimator code."""

    timestamps_s: np.ndarray
    poses_enu: np.ndarray

    def validate(self) -> None:
        n = len(self.timestamps_s)
        if self.poses_enu.shape != (n, 4, 4):
            raise ValueError("poses_enu must have shape (N, 4, 4)")
        if not _strictly_increasing(self.timestamps_s):
            raise ValueError("truth timestamps are not strictly increasing")
        if not np.all(np.isfinite(self.poses_enu)):
            raise ValueError("truth poses contain non-finite values")


@dataclass(frozen=True)
class FaultRealization:
    """Controlled GNSS realization and hidden evaluation labels."""

    timestamps_s: np.ndarray
    positions_enu_m: np.ndarray
    covariances_m2: np.ndarray
    bias_m: np.ndarray
    faulty: np.ndarray
    detectable: np.ndarray
    fault_start_s: float | None
    fault_end_s: float | None
    scenario: str
    seed: int

    def estimator_view(
        self, lidar_timestamps_s: np.ndarray, lidar_poses_local: np.ndarray
    ) -> EstimatorInput:
        result = EstimatorInput(
            lidar_timestamps_s=np.asarray(lidar_timestamps_s, dtype=float),
            lidar_poses_local=np.asarray(lidar_poses_local, dtype=float),
            gnss_timestamps_s=np.asarray(self.timestamps_s, dtype=float),
            gnss_positions_enu_m=np.asarray(self.positions_enu_m, dtype=float),
            gnss_covariances_m2=np.asarray(self.covariances_m2, dtype=float),
        )
        result.validate()
        return result
