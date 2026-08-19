"""Build truth-isolated estimator cases from cached real trajectories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .alignment import estimate_truth_free_alignment
from .faults import FaultConfig, generate_controlled_gnss
from .rtklib import parse_solution
from .schema import EstimatorInput, EvaluationTruth, FaultRealization


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_estimator_input(path: Path) -> EstimatorInput:
    with np.load(path, allow_pickle=False) as archive:
        result = EstimatorInput(
            archive["lidar_timestamps_s"],
            archive["lidar_poses_local"],
            archive["gnss_timestamps_s"],
            archive["gnss_positions_enu_m"],
            archive["gnss_covariances_m2"],
        )
    result.validate()
    return result


def load_evaluation_truth(path: Path) -> EvaluationTruth:
    with np.load(path, allow_pickle=False) as archive:
        result = EvaluationTruth(
            archive["truth_timestamps_s"], archive["truth_poses_enu"]
        )
    result.validate()
    return result


def _common_interval(
    lidar_timestamps_s: np.ndarray, truth_timestamps_s: np.ndarray
) -> tuple[float, float]:
    start = max(float(lidar_timestamps_s[0]), float(truth_timestamps_s[0]))
    end = min(float(lidar_timestamps_s[-1]), float(truth_timestamps_s[-1]))
    if end - start < 120.0:
        raise ValueError(
            f"LiDAR/truth overlap is only {end - start:.1f} s; need >=120 s"
        )
    return start, end


def _covering_slice(
    timestamps_s: np.ndarray, start_s: float, end_s: float
) -> slice:
    """Return samples spanning the requested interval, including brackets."""

    first = max(int(np.searchsorted(timestamps_s, start_s, side="right")) - 1, 0)
    last = min(
        int(np.searchsorted(timestamps_s, end_s, side="left")),
        len(timestamps_s) - 1,
    )
    if timestamps_s[first] > start_s or timestamps_s[last] < end_s:
        raise ValueError("source trajectory does not bracket requested interval")
    return slice(first, last + 1)


def _regular_sample_indices(
    timestamps_s: np.ndarray, minimum_interval_s: float
) -> np.ndarray:
    """Choose observations nearest to a fixed-rate time grid.

    Anchoring targets at the first observation avoids the rate-dependent drift
    that would arise from repeatedly accepting ``last + interval``.  A zero
    interval preserves every source observation.
    """

    if minimum_interval_s <= 0.0:
        return np.arange(len(timestamps_s), dtype=int)
    targets = np.arange(
        float(timestamps_s[0]),
        float(timestamps_s[-1]) + 0.5 * minimum_interval_s,
        minimum_interval_s,
    )
    right = np.searchsorted(timestamps_s, targets, side="left")
    right = np.clip(right, 0, len(timestamps_s) - 1)
    left = np.maximum(right - 1, 0)
    choose_right = (
        np.abs(timestamps_s[right] - targets)
        < np.abs(timestamps_s[left] - targets)
    )
    selected = np.where(choose_right, right, left)
    return np.unique(selected.astype(int))


def _save_estimator_input(path: Path, estimator_input: EstimatorInput) -> None:
    estimator_input.validate()
    np.savez_compressed(
        path,
        lidar_timestamps_s=estimator_input.lidar_timestamps_s,
        lidar_poses_local=estimator_input.lidar_poses_local,
        gnss_timestamps_s=estimator_input.gnss_timestamps_s,
        gnss_positions_enu_m=estimator_input.gnss_positions_enu_m,
        gnss_covariances_m2=estimator_input.gnss_covariances_m2,
    )


def _save_hidden_labels(
    path: Path,
    truth: EvaluationTruth,
    fault: FaultRealization,
) -> None:
    truth.validate()
    np.savez_compressed(
        path,
        truth_timestamps_s=truth.timestamps_s,
        truth_poses_enu=truth.poses_enu,
        gnss_timestamps_s=fault.timestamps_s,
        bias_m=fault.bias_m,
        faulty=fault.faulty,
        detectable=fault.detectable,
        fault_start_s=np.nan
        if fault.fault_start_s is None
        else fault.fault_start_s,
        fault_end_s=np.nan if fault.fault_end_s is None else fault.fault_end_s,
    )


def build_controlled_case(
    dataset_id: str,
    sensor_cache_path: Path,
    truth_cache_path: Path,
    scenario: str,
    seed: int,
    output_directory: Path,
) -> dict[str, object]:
    """Create separate estimator and hidden-label artifacts.

    This is the only stage allowed to read both the reference trajectory and
    estimator sensor cache.  Estimator executables receive only the first
    artifact; evaluators receive both after the output file is sealed.
    """

    with np.load(sensor_cache_path, allow_pickle=False) as sensor:
        lidar_timestamps = np.asarray(sensor["lidar_timestamps_s"], dtype=float)
        lidar_poses = np.asarray(sensor["lidar_poses_local"], dtype=float)
    truth = load_evaluation_truth(truth_cache_path)
    start, end = _common_interval(lidar_timestamps, truth.timestamps_s)
    start = float(np.ceil(start))
    end = float(np.floor(end))
    if end - start < 120.0:
        raise ValueError("integer-second common interval is shorter than 120 s")
    lidar_selection = _covering_slice(lidar_timestamps, start, end)
    truth_selection = _covering_slice(truth.timestamps_s, start, end)
    cropped_truth = EvaluationTruth(
        truth.timestamps_s[truth_selection], truth.poses_enu[truth_selection]
    )
    clean_for_initialization = generate_controlled_gnss(
        cropped_truth.timestamps_s,
        cropped_truth.poses_enu[:, :3, 3],
        "clean",
        seed,
    )
    clean_alignment = estimate_truth_free_alignment(
        lidar_timestamps[lidar_selection],
        lidar_poses[lidar_selection],
        clean_for_initialization.timestamps_s,
        clean_for_initialization.positions_enu_m,
    )
    earliest_fault_start_s = (
        clean_for_initialization.timestamps_s[clean_alignment.used_count - 1]
        + 20.0
    )
    fault = generate_controlled_gnss(
        cropped_truth.timestamps_s,
        cropped_truth.poses_enu[:, :3, 3],
        scenario,
        seed,
        FaultConfig(earliest_fault_start_s=earliest_fault_start_s),
    )
    estimator_input = fault.estimator_view(
        lidar_timestamps[lidar_selection], lidar_poses[lidar_selection]
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset_id}__{scenario}__{seed}"
    estimator_path = output_directory / f"{stem}.estimator_input.npz"
    hidden_path = output_directory / f"{stem}.hidden_labels.npz"
    _save_estimator_input(estimator_path, estimator_input)
    _save_hidden_labels(hidden_path, cropped_truth, fault)
    manifest = {
        "case_id": stem,
        "dataset_id": dataset_id,
        "scenario": scenario,
        "seed": int(seed),
        "estimator_input": str(estimator_path),
        "hidden_labels": str(hidden_path),
        "estimator_input_sha256": sha256(estimator_path),
        "hidden_labels_sha256": sha256(hidden_path),
        "lidar_frames": int(len(estimator_input.lidar_timestamps_s)),
        "gnss_epochs": int(len(estimator_input.gnss_timestamps_s)),
        "duration_s": float(
            estimator_input.gnss_timestamps_s[-1]
            - estimator_input.gnss_timestamps_s[0]
        ),
        "registered_initialization_epochs": int(clean_alignment.used_count),
        "earliest_fault_start_s": float(earliest_fault_start_s),
    }
    manifest_path = output_directory / f"{stem}.case_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_real_gnss_case(
    dataset_id: str,
    sensor_cache_path: Path,
    truth_cache_path: Path,
    rtklib_solution_path: Path,
    output_directory: Path,
    use_full_gnss_covariance: bool = False,
    minimum_gnss_interval_s: float = 0.0,
) -> dict[str, object]:
    """Build a real-SPP estimator case without precomputing degradation labels."""

    with np.load(sensor_cache_path, allow_pickle=False) as sensor:
        lidar_timestamps = np.asarray(sensor["lidar_timestamps_s"], dtype=float)
        lidar_poses = np.asarray(sensor["lidar_poses_local"], dtype=float)
    with np.load(truth_cache_path, allow_pickle=False) as truth_archive:
        truth = EvaluationTruth(
            np.asarray(truth_archive["truth_timestamps_s"], dtype=float),
            np.asarray(truth_archive["truth_poses_enu"], dtype=float),
        )
        origin = tuple(
            float(value)
            for value in truth_archive["enu_origin_lat_lon_alt"]
        )
    solution = parse_solution(
        rtklib_solution_path,
        origin,
        use_full_covariance=use_full_gnss_covariance,
    )
    gnss_timestamps = np.asarray(solution["timestamps_s"], dtype=float)
    gnss_positions = np.asarray(solution["positions_enu_m"], dtype=float)
    gnss_covariances = np.asarray(
        solution["covariances_m2"], dtype=float
    )
    start = max(
        float(lidar_timestamps[0]),
        float(truth.timestamps_s[0]),
        float(gnss_timestamps[0]),
    )
    end = min(
        float(lidar_timestamps[-1]),
        float(truth.timestamps_s[-1]),
        float(gnss_timestamps[-1]),
    )
    if end - start < 120.0:
        raise ValueError("real GNSS/LiDAR/truth overlap is shorter than 120 s")
    gnss_selection = (gnss_timestamps >= start) & (gnss_timestamps <= end)
    selected_gnss_timestamps = gnss_timestamps[gnss_selection]
    selected_source_indices = np.flatnonzero(gnss_selection)
    regular_indices = _regular_sample_indices(
        selected_gnss_timestamps, minimum_gnss_interval_s
    )
    selected_source_indices = selected_source_indices[regular_indices]
    selected_gnss_timestamps = gnss_timestamps[selected_source_indices]
    lidar_selection = _covering_slice(
        lidar_timestamps,
        float(selected_gnss_timestamps[0]),
        float(selected_gnss_timestamps[-1]),
    )
    truth_selection = _covering_slice(
        truth.timestamps_s,
        float(selected_gnss_timestamps[0]),
        float(selected_gnss_timestamps[-1]),
    )
    estimator_input = EstimatorInput(
        lidar_timestamps[lidar_selection],
        lidar_poses[lidar_selection],
        selected_gnss_timestamps,
        gnss_positions[selected_source_indices],
        gnss_covariances[selected_source_indices],
    )
    estimator_input.validate()
    cropped_truth = EvaluationTruth(
        truth.timestamps_s[truth_selection], truth.poses_enu[truth_selection]
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset_id}__real_spp"
    estimator_path = output_directory / f"{stem}.estimator_input.npz"
    hidden_path = output_directory / f"{stem}.hidden_labels.npz"
    _save_estimator_input(estimator_path, estimator_input)
    np.savez_compressed(
        hidden_path,
        truth_timestamps_s=cropped_truth.timestamps_s,
        truth_poses_enu=cropped_truth.poses_enu,
        gnss_timestamps_s=estimator_input.gnss_timestamps_s,
        fault_start_s=np.nan,
        fault_end_s=np.nan,
    )
    manifest = {
        "case_id": stem,
        "dataset_id": dataset_id,
        "scenario": "real_spp",
        "seed": 0,
        "estimator_input": str(estimator_path),
        "hidden_labels": str(hidden_path),
        "estimator_input_sha256": sha256(estimator_path),
        "hidden_labels_sha256": sha256(hidden_path),
        "lidar_frames": int(len(estimator_input.lidar_timestamps_s)),
        "gnss_epochs": int(len(estimator_input.gnss_timestamps_s)),
        "duration_s": float(
            estimator_input.gnss_timestamps_s[-1]
            - estimator_input.gnss_timestamps_s[0]
        ),
        "rtklib_solution": str(rtklib_solution_path),
        "sensor_cache_sha256": sha256(sensor_cache_path),
        "truth_cache_sha256": sha256(truth_cache_path),
        "rtklib_solution_sha256": sha256(rtklib_solution_path),
        "use_full_gnss_covariance": use_full_gnss_covariance,
        "minimum_gnss_interval_s": float(minimum_gnss_interval_s),
    }
    (output_directory / f"{stem}.case_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest
