"""Frozen DDA-FC method and its same-epoch baseline panel."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np

from paper_pipeline.covariance_union import pair_covariance_union

from evidence_topology.fusion_graph_backend import (
    _gps_noise,
    _gtsam,
    _key,
    _optimize,
)
from evidence_topology.receiver_diversity_baselines import (
    _aggregate,
    _metrics,
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
from paper_pipeline.loewner_envelope import geometric_median


def _trajectory(result, count: int) -> np.ndarray:
    return np.asarray(
        [
            np.asarray(result.atPose3(_key(index)).translation(), dtype=float)
            for index in range(count)
        ]
    )


def _optimize_single(
    lidar: np.ndarray,
    state_times: np.ndarray,
    measurement_indices: np.ndarray,
    positions: np.ndarray,
    covariances: np.ndarray,
    family: str,
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    graph, initial = _base_graph(lidar, state_times)
    gtsam = _gtsam()
    for index, position, covariance in zip(
        measurement_indices, positions, covariances
    ):
        graph.add(
            gtsam.GPSFactor(
                _key(int(index)),
                position,
                _gps_noise(covariance, family),
            )
        )
    return (
        _trajectory(_optimize(graph, initial), len(lidar)),
        float(time.perf_counter() - started),
    )


def _optimize_grouped(
    lidar: np.ndarray,
    state_times: np.ndarray,
    measurement_indices: np.ndarray,
    group_positions: list[dict[str, np.ndarray]],
    group_covariances: list[dict[str, np.ndarray]],
    family: str,
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    graph, initial = _base_graph(lidar, state_times)
    gtsam = _gtsam()
    for index, positions, covariances in zip(
        measurement_indices, group_positions, group_covariances
    ):
        for group in sorted(positions):
            graph.add(
                gtsam.GPSFactor(
                    _key(int(index)),
                    positions[group],
                    _gps_noise(covariances[group], family),
                )
            )
    return (
        _trajectory(_optimize(graph, initial), len(lidar)),
        float(time.perf_counter() - started),
    )


def _optimize_grouped_irls(
    lidar: np.ndarray,
    state_times: np.ndarray,
    measurement_indices: np.ndarray,
    group_positions: list[dict[str, np.ndarray]],
    group_covariances: list[dict[str, np.ndarray]],
    method: str,
    *,
    switch_prior: float = 4.0,
    dcs_phi: float = 4.0,
    outer_iterations: int = 4,
) -> tuple[np.ndarray, float]:
    """Strong all-receiver factor-wise robustification baseline."""

    if method not in {"switch_irls", "dcs_irls"}:
        raise ValueError(f"unsupported grouped IRLS method: {method}")
    started = time.perf_counter()
    group_names = sorted(group_positions[0])
    weights = {
        (epoch, group): 1.0
        for epoch in range(len(group_positions))
        for group in group_names
    }
    result = None
    for outer in range(outer_iterations):
        graph, initial = _base_graph(lidar, state_times)
        gtsam = _gtsam()
        for epoch, (index, positions, covariances) in enumerate(
            zip(
                measurement_indices,
                group_positions,
                group_covariances,
            )
        ):
            for group in group_names:
                weight = max(weights[(epoch, group)], 1e-4)
                graph.add(
                    gtsam.GPSFactor(
                        _key(int(index)),
                        positions[group],
                        _gps_noise(covariances[group] / weight, "direct"),
                    )
                )
        result = _optimize(graph, initial if result is None else result)
        if outer + 1 == outer_iterations:
            break
        for epoch, index in enumerate(measurement_indices):
            estimate = np.asarray(
                result.atPose3(_key(int(index))).translation(), dtype=float
            )
            for group in group_names:
                residual = estimate - group_positions[epoch][group]
                covariance = group_covariances[epoch][group]
                squared = float(
                    residual
                    @ np.linalg.pinv(covariance, hermitian=True)
                    @ residual
                )
                if method == "switch_irls":
                    switch = switch_prior / (switch_prior + squared)
                    weight = switch**2
                else:
                    scale = min(
                        1.0, 2.0 * dcs_phi / (dcs_phi + squared)
                    )
                    weight = scale**2
                weights[(epoch, group)] = float(
                    np.clip(weight, 1e-4, 1.0)
                )
    if result is None:
        raise RuntimeError("grouped IRLS produced no estimate")
    return (
        _trajectory(result, len(lidar)),
        float(time.perf_counter() - started),
    )


def _optimize_adjudicated(
    lidar: np.ndarray,
    state_times: np.ndarray,
    measurement_indices: np.ndarray,
    group_positions: list[dict[str, np.ndarray]],
    group_covariances: list[dict[str, np.ndarray]],
    consensus_positions: np.ndarray,
    consensus_covariances: np.ndarray,
    windows: list[dict[str, object]],
) -> tuple[np.ndarray, float]:
    """Optimize a graph whose receiver topology is selected per time window."""

    started = time.perf_counter()
    graph, initial = _base_graph(lidar, state_times)
    gtsam = _gtsam()
    for window in windows:
        start = int(window["start_epoch"])
        stop = int(window["stop_epoch"])
        receiver = window["selected_receiver"]
        family = str(window["selected_family"])
        for epoch in range(start, stop):
            index = int(measurement_indices[epoch])
            if window.get("factor_topology") == "receiver_subset_independent":
                selected_receivers = [
                    str(group)
                    for group in window.get("selected_receivers", [])
                ]
                if len(selected_receivers) < 2:
                    raise ValueError(
                        "receiver_subset_independent requires at least two "
                        "physical receivers"
                    )
                for group in selected_receivers:
                    graph.add(
                        gtsam.GPSFactor(
                            _key(index),
                            group_positions[epoch][group],
                            _gps_noise(
                                group_covariances[epoch][group], family
                            ),
                        )
                    )
                continue
            if window.get("factor_topology") == "receiver_subset_consensus":
                selected_receivers = [
                    str(group)
                    for group in window.get("selected_receivers", [])
                ]
                if len(selected_receivers) < 2:
                    raise ValueError(
                        "receiver_subset_consensus requires at least two "
                        "physical receivers"
                    )
                position, covariance = _aggregate(
                    np.asarray(
                        [
                            group_positions[epoch][group]
                            for group in selected_receivers
                        ]
                    ),
                    np.asarray(
                        [
                            group_covariances[epoch][group]
                            for group in selected_receivers
                        ]
                    ),
                    "geometric",
                    "nonshrinking",
                )
                graph.add(
                    gtsam.GPSFactor(
                        _key(index),
                        position,
                        _gps_noise(covariance, family),
                    )
                )
                continue
            if window.get("factor_topology") == "receiver_subset_covariance_union":
                selected_receivers = [
                    str(group)
                    for group in window.get("selected_receivers", [])
                ]
                if len(selected_receivers) != 2:
                    raise ValueError(
                        "receiver_subset_covariance_union requires exactly "
                        "two physical receivers"
                    )
                position, covariance, _ = pair_covariance_union(
                    np.asarray(
                        [
                            group_positions[epoch][group]
                            for group in selected_receivers
                        ]
                    ),
                    np.asarray(
                        [
                            group_covariances[epoch][group]
                            for group in selected_receivers
                        ]
                    ),
                )
                graph.add(
                    gtsam.GPSFactor(
                        _key(index),
                        position,
                        _gps_noise(covariance, family),
                    )
                )
                continue
            if window.get("factor_topology") == "hierarchical_consensus":
                graph.add(
                    gtsam.GPSFactor(
                        _key(index),
                        consensus_positions[epoch],
                        _gps_noise(
                            consensus_covariances[epoch], family
                        ),
                    )
                )
                continue
            if receiver is None:
                selected_groups = sorted(group_positions[epoch])
            else:
                selected_groups = [str(receiver)]
            for group in selected_groups:
                graph.add(
                    gtsam.GPSFactor(
                        _key(index),
                        group_positions[epoch][group],
                        _gps_noise(
                            group_covariances[epoch][group], family
                        ),
                    )
                )
    return (
        _trajectory(_optimize(graph, initial), len(lidar)),
        float(time.perf_counter() - started),
    )


def _base_graph(
    lidar_poses: np.ndarray, state_times: np.ndarray
):
    """Pose graph with interval-propagated KISS odometry uncertainty."""

    gtsam = _gtsam()
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
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
    for index, pose in enumerate(lidar_poses):
        initial.insert(_key(index), gtsam.Pose3(pose))
    graph.add(
        gtsam.PriorFactorPose3(
            _key(0), gtsam.Pose3(lidar_poses[0]), prior_noise
        )
    )
    nominal_dt = 1.0
    for index in range(1, len(lidar_poses)):
        interval_scale = np.sqrt(
            max(
                (state_times[index] - state_times[index - 1])
                / max(nominal_dt, 1e-6),
                1e-6,
            )
        )
        odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(
            interval_scale
            * np.asarray(
                [
                    np.deg2rad(0.5),
                    np.deg2rad(0.5),
                    np.deg2rad(0.5),
                    0.10,
                    0.10,
                    0.10,
                ]
            )
        )
        relative = np.linalg.inv(lidar_poses[index - 1]) @ lidar_poses[index]
        graph.add(
            gtsam.BetweenFactorPose3(
                _key(index - 1),
                _key(index),
                gtsam.Pose3(relative),
                odometry_noise,
            )
        )
    return graph, initial


def _robust_distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    sigma = 1.4826 * max(mad, 1e-12)
    return {
        "median": median,
        "mad": mad,
        "robust_sigma": sigma,
        "p95": float(np.percentile(values, 95.0)),
        "contamination_fraction": float(
            np.mean(values > median + 3.0 * sigma)
        ),
    }


def _deduplicate_exact_measurements(
    points: list[np.ndarray], covariances: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Remove byte-equivalent replayed solution streams inside one receiver."""

    points_array = np.asarray(points, dtype=float)
    covariance_array = np.asarray(covariances, dtype=float)
    signatures = np.column_stack(
        [points_array, covariance_array.reshape(len(points_array), -1)]
    )
    _, indices = np.unique(signatures, axis=0, return_index=True)
    indices = np.sort(indices)
    return points_array[indices], covariance_array[indices]


def _canonical_alignment_reference(
    raw_points: dict[str, list[np.ndarray]],
    raw_covariances: dict[str, list[np.ndarray]],
) -> np.ndarray:
    """Return the method-independent physical-source alignment reference.

    Alignment must not change when the evaluated fusion method changes its
    grouping or covariance construction.  Exact replays are removed, each
    registered physical source contributes one arithmetic center, and a
    rotation-equivariant geometric median combines the source centers.
    """

    source_centers = []
    for source in sorted(raw_points):
        unique_points, _ = _deduplicate_exact_measurements(
            raw_points[source], raw_covariances[source]
        )
        source_centers.append(np.mean(unique_points, axis=0))
    if not source_centers:
        raise ValueError("canonical alignment requires a physical source")
    return geometric_median(np.asarray(source_centers))


def _window_decision(
    start: int,
    stop: int,
    group_names: list[str],
    group_positions: list[dict[str, np.ndarray]],
    consensus_positions: np.ndarray,
    lidar_common_positions: np.ndarray,
) -> dict[str, object]:
    """Truth-free residual-distribution decision for one GNSS epoch window."""

    lidar_window = lidar_common_positions[start:stop]
    consensus_window = consensus_positions[start:stop]
    consensus_mismatch = np.linalg.norm(
        np.diff(consensus_window, axis=0)
        - np.diff(lidar_window, axis=0),
        axis=1,
    )
    consensus_distribution = _robust_distribution(consensus_mismatch)
    receiver_distributions: dict[str, dict[str, float]] = {}
    receiver_absolute_residuals: dict[str, dict[str, float]] = {}
    receiver_matrix = np.asarray(
        [
            [positions[group] for group in group_names]
            for positions in group_positions[start:stop]
        ]
    )
    pairwise_separation = []
    for epoch_points in receiver_matrix:
        distances = np.linalg.norm(
            epoch_points[:, None, :] - epoch_points[None, :, :], axis=2
        )
        pairwise_separation.append(float(np.max(distances)))
    pairwise_distribution = _robust_distribution(
        np.asarray(pairwise_separation)
    )
    receiver_pair_distributions = {}
    for left_index, left in enumerate(group_names):
        for right in group_names[left_index + 1 :]:
            pair_name = f"{left}__{right}"
            receiver_pair_distributions[pair_name] = _robust_distribution(
                np.linalg.norm(
                    receiver_matrix[:, group_names.index(left), :]
                    - receiver_matrix[:, group_names.index(right), :],
                    axis=1,
                )
            )
    for group in group_names:
        stream = np.asarray(
            [positions[group] for positions in group_positions[start:stop]]
        )
        mismatch = np.linalg.norm(
            np.diff(stream, axis=0) - np.diff(lidar_window, axis=0),
            axis=1,
        )
        receiver_distributions[group] = _robust_distribution(mismatch)
        receiver_absolute_residuals[group] = _robust_distribution(
            np.linalg.norm(stream - lidar_window, axis=1)
        )
    motion_p95 = np.asarray(
        [receiver_distributions[group]["p95"] for group in group_names]
    )
    absolute_median = np.asarray(
        [
            receiver_absolute_residuals[group]["median"]
            for group in group_names
        ]
    )
    motion_dispersion = float(
        np.max(motion_p95) / max(float(np.min(motion_p95)), 1e-12)
    )
    absolute_dispersion = float(
        np.max(absolute_median)
        / max(float(np.min(absolute_median)), 1e-12)
    )
    heavy_tailed = (
        consensus_distribution["contamination_fraction"] > 0.10
    )
    heterogeneous = motion_dispersion > 2.0 or absolute_dispersion > 2.0
    normalized_motion = motion_p95 / max(float(np.min(motion_p95)), 1e-12)
    normalized_absolute = absolute_median / max(
        float(np.min(absolute_median)), 1e-12
    )
    scores = normalized_motion + normalized_absolute
    selected_receiver = (
        group_names[int(np.argmin(scores))]
        if heavy_tailed or heterogeneous
        else None
    )
    if heavy_tailed:
        branch = "isolated_receiver_cauchy"
        family = "cauchy"
    elif heterogeneous:
        branch = "isolated_receiver_gaussian"
        family = "direct"
    else:
        branch = "receiver_resolved_gaussian"
        family = "direct"
    return {
        "start_epoch": start,
        "stop_epoch": stop,
        "start_common_epoch": start,
        "stop_common_epoch": stop,
        "consensus_relative_mismatch": consensus_distribution,
        "receiver_relative_mismatch": receiver_distributions,
        "receiver_absolute_residual": receiver_absolute_residuals,
        "maximum_pairwise_receiver_separation": pairwise_distribution,
        "receiver_pair_separation": receiver_pair_distributions,
        "receiver_motion_p95_dispersion_ratio": motion_dispersion,
        "receiver_absolute_median_dispersion_ratio": absolute_dispersion,
        "receiver_scores": {
            group: float(score)
            for group, score in zip(group_names, scores)
        },
        "selected_branch": branch,
        "selected_receiver": selected_receiver,
        "selected_family": family,
    }


def _initial_adjudication(
    prefix_count: int,
    group_names: list[str],
    group_positions: list[dict[str, np.ndarray]],
    consensus_positions: np.ndarray,
    lidar_common_positions: np.ndarray,
    tail_adjudication_enabled: bool,
) -> dict[str, object]:
    """Infer the nominal factor topology only from the alignment prefix."""

    nominal = _window_decision(
        0,
        prefix_count,
        group_names,
        group_positions,
        consensus_positions,
        lidar_common_positions,
    )
    heavy_tailed = tail_adjudication_enabled and (
        float(
            nominal["consensus_relative_mismatch"][
                "contamination_fraction"
            ]
        )
        > 0.10
    )
    heterogeneous = (
        float(nominal["receiver_motion_p95_dispersion_ratio"]) > 2.0
    )
    selected_receiver = (
        min(
            group_names,
            key=lambda group: float(
                nominal["receiver_relative_mismatch"][group]["p95"]
            ),
        )
        if heavy_tailed or heterogeneous
        else None
    )
    if heavy_tailed:
        selected_branch = "isolated_receiver_cauchy"
        selected_family = "cauchy"
    elif heterogeneous:
        selected_branch = "isolated_receiver_gaussian"
        selected_family = "direct"
    else:
        selected_branch = "receiver_resolved_gaussian"
        selected_family = "direct"
    return {
        "prefix_epoch_count": prefix_count,
        "selected_branch": selected_branch,
        "selected_receiver": selected_receiver,
        "selected_family": selected_family,
        "consensus_relative_mismatch": nominal[
            "consensus_relative_mismatch"
        ],
        "receiver_relative_mismatch": nominal[
            "receiver_relative_mismatch"
        ],
        "receiver_absolute_residual": nominal[
            "receiver_absolute_residual"
        ],
        "receiver_motion_p95_dispersion_ratio": nominal[
            "receiver_motion_p95_dispersion_ratio"
        ],
        "receiver_absolute_median_dispersion_ratio": nominal[
            "receiver_absolute_median_dispersion_ratio"
        ],
        "maximum_pairwise_receiver_separation": nominal[
            "maximum_pairwise_receiver_separation"
        ],
        "receiver_pair_separation": nominal["receiver_pair_separation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-output", type=Path, required=True)
    parser.add_argument("--tolerance-s", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--reference-max-gap-s",
        type=float,
        default=0.0,
        help=(
            "If positive, score only states bracketed by reference samples "
            "whose interval does not exceed this value. Zero preserves the "
            "strict complete-reference requirement."
        ),
    )
    parser.add_argument(
        "--reference-nearest-tolerance-s",
        type=float,
        default=0.0,
        help=(
            "With --reference-max-gap-s, also require a state to be within "
            "this distance of a bracketing reference sample."
        ),
    )
    parser.add_argument(
        "--alignment-prefix-epochs",
        type=int,
        default=0,
        help=(
            "Use only this many initial common GNSS epochs to initialize the "
            "LiDAR-to-ENU transform. Zero preserves the legacy full-sequence "
            "alignment and is intended only for reproduction."
        ),
    )
    parser.add_argument(
        "--initial-prefix-epochs",
        type=int,
        default=0,
        help=(
            "Use this fixed initial evidence count for receiver anchoring. "
            "Zero preserves the legacy sequence-length-dependent 25 percent "
            "prefix and is intended only for reproduction."
        ),
    )
    parser.add_argument("--window-epochs", type=int, default=30)
    parser.add_argument(
        "--epoch-policy",
        choices=("all_sources", "all_streams"),
        default="all_sources",
        help=(
            "all_sources is the operational policy: every registered physical "
            "source must be represented, but a missing sibling solution stream "
            "does not suppress the source. all_streams freezes a complete-case "
            "epoch set for causal grouping ablations."
        ),
    )
    parser.add_argument(
        "--decision-stride-epochs",
        type=int,
        default=0,
        help=(
            "If positive, evaluate overlapping evidence windows of "
            "--window-epochs and apply each decision for this many epochs."
        ),
    )
    parser.add_argument("--contamination-threshold", type=float, default=0.10)
    parser.add_argument(
        "--persistent-shift-ratio", type=float, default=1.35
    )
    parser.add_argument(
        "--persistent-shift-difference-m", type=float, default=12.0
    )
    parser.add_argument(
        "--isolation-ratio-threshold", type=float, default=0.80
    )
    parser.add_argument("--pair-shift-ratio", type=float, default=2.0)
    parser.add_argument(
        "--pair-shift-difference-m", type=float, default=8.0
    )
    parser.add_argument("--pair-tail-ceiling", type=float, default=3.0)
    parser.add_argument("--quorum-tail-threshold", type=float, default=1.40)
    parser.add_argument(
        "--change-action",
        choices=(
            "receiver_resolved",
            "isolate_best",
            "hierarchical_consensus",
            "topology_adaptive",
            "quorum_conservative",
            "quorum_motion",
            "quorum_guarded",
            "quorum_chi2_guarded",
        ),
        default="topology_adaptive",
        help=(
            "After a confirmed change, topology_adaptive distinguishes an "
            "isolated nominal receiver from distributed disagreement using "
            "the physical-receiver separation triangle."
        ),
    )
    parser.add_argument(
        "--provisional-action",
        choices=(
            "hierarchical_consensus_cauchy",
            "hierarchical_consensus_direct",
            "receiver_resolved_direct",
            "quorum_aware",
        ),
        default="quorum_aware",
        help=(
            "Action for the first unconfirmed change window. The final "
            "quorum-aware rule uses receiver-resolved Gaussian factors only "
            "with at least three physical receivers and a coherent pair "
            "shift; otherwise it uses a Cauchy consensus factor."
        ),
    )
    parser.add_argument(
        "--ablation",
        choices=(
            "none",
            "no_dependency_grouping",
            "no_tail_adjudication",
            "no_change_guard",
        ),
        default="none",
    )
    args = parser.parse_args()

    input_paths = sorted(args.case_root.glob("*.estimator_input.npz"))
    datasets = [load_estimator_input(path) for path in input_paths]
    canonical_groups = [_hardware_group(path) for path in input_paths]
    if args.ablation == "no_dependency_grouping":
        groups = [
            path.name.replace(".estimator_input.npz", "")
            for path in input_paths
        ]
    else:
        groups = [_hardware_group(path) for path in input_paths]
    group_names = sorted(set(groups))
    canonical_group_names = sorted(set(canonical_groups))
    if len(group_names) < 2:
        raise ValueError("DDA-FC requires at least two physical receivers")
    start = max(data.lidar_timestamps_s[0] for data in datasets)
    end = min(data.lidar_timestamps_s[-1] for data in datasets)
    base = datasets[0]
    all_times = np.unique(
        np.concatenate([data.gnss_timestamps_s for data in datasets])
    )
    all_times = all_times[(all_times >= start) & (all_times <= end)]
    factor_construction_started = time.perf_counter()
    times = []
    group_positions: list[dict[str, np.ndarray]] = []
    group_covariances: list[dict[str, np.ndarray]] = []
    flat_positions: list[np.ndarray] = []
    flat_covariances: list[np.ndarray] = []
    alignment_positions: list[np.ndarray] = []
    for timestamp in all_times:
        raw_points: dict[str, list[np.ndarray]] = {}
        raw_covariances: dict[str, list[np.ndarray]] = {}
        alignment_raw_points: dict[str, list[np.ndarray]] = {}
        alignment_raw_covariances: dict[str, list[np.ndarray]] = {}
        epoch_flat_points = []
        epoch_flat_covariances = []
        for data, group, canonical_group in zip(
            datasets, groups, canonical_groups
        ):
            indices, differences = _nearest_indices(
                data.gnss_timestamps_s, np.asarray([timestamp])
            )
            if differences[0] > args.tolerance_s:
                continue
            index = int(indices[0])
            point = np.asarray(data.gnss_positions_enu_m[index], dtype=float)
            covariance = np.asarray(
                data.gnss_covariances_m2[index], dtype=float
            )
            raw_points.setdefault(group, []).append(point)
            raw_covariances.setdefault(group, []).append(covariance)
            alignment_raw_points.setdefault(canonical_group, []).append(
                point
            )
            alignment_raw_covariances.setdefault(
                canonical_group, []
            ).append(covariance)
            epoch_flat_points.append(point)
            epoch_flat_covariances.append(covariance)
        complete_stream_epoch = len(epoch_flat_points) == len(datasets)
        complete_source_epoch = (
            len(alignment_raw_points) == len(canonical_group_names)
        )
        if args.epoch_policy == "all_streams" and not complete_stream_epoch:
            continue
        if args.epoch_policy == "all_sources" and not complete_source_epoch:
            continue
        if (
            args.ablation == "no_dependency_grouping"
            and not complete_stream_epoch
        ):
            raise ValueError(
                "no_dependency_grouping requires --epoch-policy all_streams "
                "so every stream-defined group exists at every epoch"
            )
        positions = {}
        covariances = {}
        for group in group_names:
            unique_points, unique_covariances = _deduplicate_exact_measurements(
                raw_points[group], raw_covariances[group]
            )
            positions[group], covariances[group] = _aggregate(
                unique_points,
                unique_covariances,
                "geometric",
                "nonshrinking",
            )
        times.append(timestamp)
        group_positions.append(positions)
        group_covariances.append(covariances)
        flat_positions.append(np.asarray(epoch_flat_points))
        flat_covariances.append(np.asarray(epoch_flat_covariances))
        alignment_positions.append(
            _canonical_alignment_reference(
                alignment_raw_points, alignment_raw_covariances
            )
        )
    times = np.asarray(times)
    if len(times) < 100:
        raise ValueError(f"too few all-group epochs: {len(times)}")

    consensus_positions = []
    consensus_covariances = []
    information_positions = []
    information_covariances = []
    for positions, covariances, raw_points, raw_covariances in zip(
        group_positions,
        group_covariances,
        flat_positions,
        flat_covariances,
    ):
        position, covariance = _aggregate(
            np.asarray([positions[group] for group in group_names]),
            np.asarray([covariances[group] for group in group_names]),
            "geometric",
            "independence",
        )
        consensus_positions.append(position)
        consensus_covariances.append(covariance)
        position, covariance = _aggregate(
            raw_points,
            raw_covariances,
            "information_mean",
            "independence",
        )
        information_positions.append(position)
        information_covariances.append(covariance)
    consensus_positions = np.asarray(consensus_positions)
    consensus_covariances = np.asarray(consensus_covariances)
    information_positions = np.asarray(information_positions)
    information_covariances = np.asarray(information_covariances)
    factor_construction_runtime_s = (
        time.perf_counter() - factor_construction_started
    )

    alignment_started = time.perf_counter()
    alignment_count = (
        len(times)
        if args.alignment_prefix_epochs <= 0
        else min(len(times), args.alignment_prefix_epochs)
    )
    if alignment_count < 30:
        raise ValueError("alignment prefix must contain at least 30 epochs")
    alignment = estimate_truth_free_alignment(
        base.lidar_timestamps_s,
        base.lidar_poses_local,
        times[:alignment_count],
        np.asarray(alignment_positions)[:alignment_count],
        ransac_threshold_m=20.0,
        minimum_count=30,
        minimum_duration_s=60.0,
        random_seed=args.seed,
    )
    local_common = interpolate_poses(
        base.lidar_timestamps_s, base.lidar_poses_local, times
    )
    lidar_common = transform_poses(
        alignment.transform_enu_from_lidar, local_common
    )
    lidar_common_positions = lidar_common[:, :3, 3]
    regular_times = np.arange(times[0], times[-1] + 1e-9, 1.0)
    state_times = np.unique(np.concatenate([regular_times, times]))
    state_times = state_times[
        (state_times >= times[0]) & (state_times <= times[-1])
    ]
    measurement_indices = np.searchsorted(state_times, times)
    if not np.allclose(state_times[measurement_indices], times, atol=1e-9):
        raise RuntimeError("GNSS epochs were not preserved in the state grid")
    local_states = interpolate_poses(
        base.lidar_timestamps_s, base.lidar_poses_local, state_times
    )
    lidar = transform_poses(
        alignment.transform_enu_from_lidar, local_states
    )
    lidar_positions = lidar[:, :3, 3]
    alignment_runtime_s = time.perf_counter() - alignment_started

    if args.window_epochs < 20:
        raise ValueError("window-epochs must be at least 20")
    decision_started = time.perf_counter()
    prefix_count = (
        max(30, int(np.ceil(0.25 * len(times))))
        if args.initial_prefix_epochs <= 0
        else min(len(times), args.initial_prefix_epochs)
    )
    if prefix_count < 30:
        raise ValueError("initial prefix must contain at least 30 epochs")
    prefix_pairwise_mahalanobis = []
    for positions, covariances in zip(
        group_positions[:prefix_count],
        group_covariances[:prefix_count],
    ):
        for left_index, left in enumerate(group_names):
            for right in group_names[left_index + 1 :]:
                difference = positions[left] - positions[right]
                covariance = covariances[left] + covariances[right]
                prefix_pairwise_mahalanobis.append(
                    float(
                        difference
                        @ np.linalg.solve(covariance, difference)
                    )
                )
    prefix_pairwise_mahalanobis = np.asarray(
        prefix_pairwise_mahalanobis, dtype=float
    )
    chi2_3_95 = 7.814727903251179
    prefix_pairwise_mahalanobis_p95 = float(
        np.percentile(prefix_pairwise_mahalanobis, 95.0)
    )
    prefix_receivers_statistically_consistent = bool(
        prefix_pairwise_mahalanobis_p95 <= chi2_3_95
    )
    initial_decision = _initial_adjudication(
        prefix_count,
        group_names,
        group_positions,
        consensus_positions,
        lidar_common_positions,
        args.ablation != "no_tail_adjudication",
    )
    initial_decision["pairwise_mahalanobis_d2"] = {
        "median": float(np.median(prefix_pairwise_mahalanobis)),
        "p95": prefix_pairwise_mahalanobis_p95,
        "chi2_3_95_threshold": chi2_3_95,
        "fraction_within_chi2_3_95": float(
            np.mean(prefix_pairwise_mahalanobis <= chi2_3_95)
        ),
        "prefix_receivers_statistically_consistent": (
            prefix_receivers_statistically_consistent
        ),
    }
    windows = []
    if args.decision_stride_epochs > 0:
        if args.decision_stride_epochs > args.window_epochs:
            raise ValueError(
                "decision stride cannot exceed evidence-window length"
            )
        for segment_start in range(
            0, len(times), args.decision_stride_epochs
        ):
            segment_stop = min(
                segment_start + args.decision_stride_epochs, len(times)
            )
            evidence_stop = segment_stop
            evidence_start = max(
                0, evidence_stop - args.window_epochs
            )
            if evidence_stop - evidence_start < 20:
                evidence_stop = min(
                    len(times), evidence_start + max(20, args.window_epochs)
                )
            window = _window_decision(
                evidence_start,
                evidence_stop,
                group_names,
                group_positions,
                consensus_positions,
                lidar_common_positions,
            )
            window["evidence_start_epoch"] = evidence_start
            window["evidence_stop_epoch"] = evidence_stop
            window["start_epoch"] = segment_start
            window["stop_epoch"] = segment_stop
            window["start_common_epoch"] = segment_start
            window["stop_common_epoch"] = segment_stop
            windows.append(window)
    else:
        for window_start in range(0, len(times), args.window_epochs):
            window_stop = min(
                window_start + args.window_epochs, len(times)
            )
            if window_stop - window_start < 20 and windows:
                previous_start = int(windows[-1]["start_epoch"])
                windows[-1] = _window_decision(
                    previous_start,
                    window_stop,
                    group_names,
                    group_positions,
                    consensus_positions,
                    lidar_common_positions,
                )
                break
            windows.append(
                _window_decision(
                    window_start,
                    window_stop,
                    group_names,
                    group_positions,
                    consensus_positions,
                    lidar_common_positions,
                )
            )
    initial_receiver = initial_decision["selected_receiver"]
    initial_family = str(initial_decision["selected_family"])
    baseline_pairwise = initial_decision[
        "maximum_pairwise_receiver_separation"
    ]
    pairwise_median_limit = max(
        10.0,
        float(baseline_pairwise["median"])
        + 6.0 * float(baseline_pairwise["robust_sigma"]),
    )
    pairwise_p95_limit = max(
        30.0, float(baseline_pairwise["p95"]) + 20.0
    )
    guard_latched = False
    latched_topology_decision = None
    selected_pair_key = None
    previous_selected_pair_median = None
    pending_pair_shift_reference = None
    if initial_receiver is not None:
        initial_receiver_name = str(initial_receiver)
        alternative = min(
            (
                group
                for group in group_names
                if group != initial_receiver_name
            ),
            key=lambda group: float(
                initial_decision["receiver_relative_mismatch"][group]["p95"]
            ),
        )
        selected_pair_key = "__".join(
            sorted((initial_receiver_name, alternative))
        )
    for window in windows:
        window["unanchored_selected_branch"] = window["selected_branch"]
        window["unanchored_selected_receiver"] = window[
            "selected_receiver"
        ]
        window["unanchored_selected_family"] = window["selected_family"]
        absolute_medians = {
            group: float(
                window["receiver_absolute_residual"][group]["median"]
            )
            for group in group_names
        }
        motion_p95 = {
            group: float(
                window["receiver_relative_mismatch"][group]["p95"]
            )
            for group in group_names
        }
        normalized_score = {
            group: (
                absolute_medians[group]
                / max(min(absolute_medians.values()), 1e-12)
                + motion_p95[group]
                / max(min(motion_p95.values()), 1e-12)
            )
            for group in group_names
        }
        candidate = min(normalized_score, key=normalized_score.get)
        pairwise = window["maximum_pairwise_receiver_separation"]
        receiver_divergence = bool(
            float(pairwise["median"]) > pairwise_median_limit
            or float(pairwise["p95"]) > pairwise_p95_limit
        )
        if args.ablation == "no_change_guard":
            window["selected_receiver"] = initial_receiver
            window["selected_family"] = initial_family
            window["selected_branch"] = "ablation_static_nominal_topology"
            window["change_guard_reason"] = "ablation_no_change_guard"
            window["receiver_divergence"] = receiver_divergence
            continue
        if initial_receiver is not None:
            current = str(initial_receiver)
            selected_pair = window["receiver_pair_separation"][
                selected_pair_key
            ]
            selected_pair_median = float(selected_pair["median"])
            selected_pair_tail_ratio = float(selected_pair["p95"]) / max(
                selected_pair_median, 1e-12
            )
            pair_reference = previous_selected_pair_median
            sudden_pair_shift = bool(
                pair_reference is not None
                and selected_pair_median
                > args.pair_shift_ratio * pair_reference
                and selected_pair_median
                - pair_reference
                > args.pair_shift_difference_m
                and selected_pair_tail_ratio < args.pair_tail_ceiling
            )
            confirmed_pair_shift = bool(
                pending_pair_shift_reference is not None
                and selected_pair_median
                > args.pair_shift_ratio * pending_pair_shift_reference
                and selected_pair_median
                - pending_pair_shift_reference
                > args.pair_shift_difference_m
                and selected_pair_tail_ratio < args.pair_tail_ceiling
            )
            previous_selected_pair_median = selected_pair_median
            best_absolute = min(
                group_names, key=lambda group: absolute_medians[group]
            )
            current_absolute = absolute_medians[current]
            best_value = absolute_medians[best_absolute]
            persistent_selected_shift = bool(
                best_absolute != current
                and current_absolute
                / max(best_value, 1e-12)
                > args.persistent_shift_ratio
                and current_absolute - best_value
                > args.persistent_shift_difference_m
            )
            if (
                guard_latched
                or persistent_selected_shift
                or confirmed_pair_shift
            ):
                guard_latched = True
                pending_pair_shift_reference = None
                pair_medians = {
                    key: float(value["median"])
                    for key, value in window[
                        "receiver_pair_separation"
                    ].items()
                }
                current_incident = [
                    value
                    for key, value in pair_medians.items()
                    if current in key.split("__")
                ]
                alternative_internal = [
                    value
                    for key, value in pair_medians.items()
                    if current not in key.split("__")
                ]
                current_incident_median = float(
                    np.median(current_incident)
                )
                alternative_internal_median = (
                    float(np.median(alternative_internal))
                    if alternative_internal
                    else float("inf")
                )
                isolation_ratio = (
                    alternative_internal_median
                    / max(current_incident_median, 1e-12)
                )
                window["selected_receiver_isolation_ratio"] = (
                    isolation_ratio
                )
                window["selected_receiver_isolation_threshold"] = (
                    args.isolation_ratio_threshold
                )
                if args.change_action in {
                    "topology_adaptive",
                    "quorum_conservative",
                    "quorum_motion",
                    "quorum_guarded",
                    "quorum_chi2_guarded",
                }:
                    selected = None
                    if latched_topology_decision is None:
                        latched_topology_decision = (
                            "isolated_nominal_receiver"
                            if isolation_ratio < args.isolation_ratio_threshold
                            else "distributed_receiver_disagreement"
                        )
                    window["change_guard_topology_decision"] = (
                        latched_topology_decision
                    )
                    window[
                        "topology_decision_latched_from_first_confirmation"
                    ] = True
                    if (
                        latched_topology_decision
                        == "isolated_nominal_receiver"
                    ):
                        if args.change_action in {
                            "quorum_conservative",
                            "quorum_motion",
                        }:
                            alternatives = [
                                group
                                for group in group_names
                                if group != current
                            ]
                            selected = min(
                                alternatives,
                                key=lambda group: normalized_score[group],
                            )
                            window["selected_branch"] = (
                                "quorum_isolated_anchor_excluded_cauchy"
                            )
                            window["selected_family"] = "cauchy"
                        else:
                            window["selected_branch"] = (
                                "change_guard_isolated_receiver_resolved_gaussian"
                            )
                            window["selected_family"] = "direct"
                            window["factor_topology"] = "receiver_resolved"
                    else:
                        if args.change_action in {
                            "quorum_conservative",
                            "quorum_motion",
                            "quorum_guarded",
                            "quorum_chi2_guarded",
                        }:
                            consensus_motion = float(
                                window["consensus_relative_mismatch"][
                                    "median"
                                ]
                            )
                            anchor_motion = float(
                                window["receiver_relative_mismatch"][current][
                                    "median"
                                ]
                            )
                            use_consensus = bool(
                                (
                                    args.change_action == "quorum_motion"
                                    and consensus_motion < anchor_motion
                                )
                                or (
                                    args.change_action
                                    == "quorum_chi2_guarded"
                                    and prefix_receivers_statistically_consistent
                                )
                            )
                            if use_consensus:
                                selected = None
                                window["selected_branch"] = (
                                    "quorum_distributed_chi2_consensus_cauchy"
                                    if args.change_action
                                    == "quorum_chi2_guarded"
                                    else
                                    "quorum_distributed_motion_consensus_cauchy"
                                )
                                window["factor_topology"] = (
                                    "hierarchical_consensus"
                                )
                            else:
                                selected = current
                                window["selected_branch"] = (
                                    "quorum_distributed_anchor_retained_cauchy"
                                )
                            window["selected_family"] = "cauchy"
                        else:
                            window["selected_branch"] = (
                                "change_guard_distributed_consensus_cauchy"
                            )
                            window["selected_family"] = "cauchy"
                            window["factor_topology"] = (
                                "hierarchical_consensus"
                            )
                elif args.change_action == "isolate_best":
                    alternatives = [
                        group for group in group_names if group != current
                    ]
                    selected = min(
                        alternatives,
                        key=lambda group: normalized_score[group],
                    )
                    window["selected_branch"] = (
                        "change_guard_isolated_alternative_cauchy"
                    )
                    window["selected_family"] = "cauchy"
                elif args.change_action == "hierarchical_consensus":
                    selected = None
                    window["selected_branch"] = (
                        "change_guard_hierarchical_consensus_cauchy"
                    )
                    window["selected_family"] = "cauchy"
                    window["factor_topology"] = "hierarchical_consensus"
                else:
                    selected = None
                    window["selected_branch"] = (
                        "change_guard_receiver_resolved_gaussian"
                    )
                    window["selected_family"] = "direct"
                window["change_guard_reason"] = (
                    "latched_selected_receiver_change"
                )
            elif sudden_pair_shift:
                pending_pair_shift_reference = pair_reference
                selected = None
                if (
                    args.provisional_action
                    == "hierarchical_consensus_cauchy"
                ):
                    window["selected_branch"] = (
                        "provisional_hierarchical_consensus_cauchy"
                    )
                    window["selected_family"] = "cauchy"
                    window["factor_topology"] = "hierarchical_consensus"
                elif (
                    args.provisional_action
                    == "hierarchical_consensus_direct"
                ):
                    window["selected_branch"] = (
                        "provisional_hierarchical_consensus_direct"
                    )
                    window["selected_family"] = "direct"
                    window["factor_topology"] = "hierarchical_consensus"
                elif args.provisional_action == "quorum_aware":
                    if (
                        len(group_names) >= 3
                        and selected_pair_tail_ratio
                        < args.quorum_tail_threshold
                    ):
                        window["selected_branch"] = (
                            "provisional_quorum_receiver_resolved_gaussian"
                        )
                        window["selected_family"] = "direct"
                    else:
                        window["selected_branch"] = (
                            "provisional_quorum_consensus_cauchy"
                        )
                        window["selected_family"] = "cauchy"
                        window["factor_topology"] = (
                            "hierarchical_consensus"
                        )
                else:
                    window["selected_branch"] = (
                        "provisional_receiver_resolved_gaussian"
                    )
                    window["selected_family"] = "direct"
                window["change_guard_reason"] = (
                    "provisional_selected_receiver_change"
                )
            elif (
                args.ablation != "no_tail_adjudication"
                and
                float(
                    window["receiver_relative_mismatch"][current][
                        "contamination_fraction"
                    ]
                )
                > args.contamination_threshold
            ):
                pending_pair_shift_reference = None
                selected = current
                window["selected_branch"] = (
                    "anchored_receiver_cauchy"
                )
                window["selected_family"] = "cauchy"
                window["change_guard_reason"] = (
                    "selected_receiver_heavy_tail"
                )
            else:
                pending_pair_shift_reference = None
                selected = current
                window["selected_branch"] = (
                    "anchored_receiver_nominal"
                )
                window["selected_family"] = initial_family
                window["change_guard_reason"] = "no_change"
            window["selected_receiver"] = selected
            window["selected_pair_key"] = selected_pair_key
            window["selected_pair_tail_ratio"] = selected_pair_tail_ratio
            window["sudden_selected_pair_shift"] = sudden_pair_shift
            window["confirmed_selected_pair_shift"] = confirmed_pair_shift
            window["change_guard_latched"] = guard_latched
        elif receiver_divergence:
            window["selected_receiver"] = candidate
            window["selected_branch"] = (
                "change_guard_isolated_receiver_cauchy"
            )
            window["selected_family"] = "cauchy"
            window["change_guard_reason"] = (
                "physical_receiver_pairwise_divergence"
            )
        else:
            window["selected_receiver"] = None
            window["selected_branch"] = "anchored_receiver_resolved_gaussian"
            window["selected_family"] = "direct"
            window["change_guard_reason"] = "no_change"
        window["receiver_divergence"] = receiver_divergence
    branch_counts: dict[str, int] = {}
    receiver_counts: dict[str, int] = {}
    for window in windows:
        branch = str(window["selected_branch"])
        if window.get("factor_topology") == "receiver_subset_consensus":
            receiver = "subset:" + ",".join(
                sorted(
                    str(group)
                    for group in window.get("selected_receivers", [])
                )
            )
        else:
            receiver = (
                str(window["selected_receiver"])
                if window["selected_receiver"] is not None
                else "all_physical_receivers"
            )
        branch_counts[branch] = branch_counts.get(branch, 0) + 1
        receiver_counts[receiver] = receiver_counts.get(receiver, 0) + 1
    decision = {
        "alignment_prefix_epoch_count": alignment_count,
        "initial_prefix_policy": (
            "legacy_fraction"
            if args.initial_prefix_epochs <= 0
            else "fixed_count"
        ),
        "window_epochs": args.window_epochs,
        "decision_stride_epochs": args.decision_stride_epochs,
        "contamination_threshold": args.contamination_threshold,
        "receiver_dispersion_threshold": 2.0,
        "persistent_shift_ratio_threshold": args.persistent_shift_ratio,
        "persistent_shift_difference_threshold_m": (
            args.persistent_shift_difference_m
        ),
        "selected_receiver_isolation_ratio_threshold": (
            args.isolation_ratio_threshold
        ),
        "pairwise_median_limit_m": pairwise_median_limit,
        "pairwise_p95_limit_m": pairwise_p95_limit,
        "sudden_pair_shift_ratio_threshold": args.pair_shift_ratio,
        "sudden_pair_shift_difference_threshold_m": (
            args.pair_shift_difference_m
        ),
        "sudden_pair_tail_ratio_ceiling": args.pair_tail_ceiling,
        "quorum_pair_tail_ratio_threshold": args.quorum_tail_threshold,
        "initial_adjudication": initial_decision,
        "ablation": args.ablation,
        "change_action": args.change_action,
        "provisional_action": args.provisional_action,
        "branch_counts": branch_counts,
        "receiver_counts": receiver_counts,
        "windows": windows,
    }
    decision_runtime_s = time.perf_counter() - decision_started

    # Every trajectory is estimated before hidden truth is opened.
    trajectories: dict[str, np.ndarray] = {
        "aligned_kiss_icp": lidar_positions,
    }
    runtimes: dict[str, float] = {"aligned_kiss_icp": 0.0}
    for family in ("direct", "huber", "cauchy"):
        name = f"grouped_physical_receivers__{family}"
        trajectories[name], runtimes[name] = _optimize_grouped(
            lidar,
            state_times,
            measurement_indices,
            group_positions,
            group_covariances,
            family,
        )
        name = f"hierarchical_consensus__{family}"
        trajectories[name], runtimes[name] = _optimize_single(
            lidar,
            state_times,
            measurement_indices,
            consensus_positions,
            consensus_covariances,
            family,
        )
        name = f"flat_information_mean__{family}"
        trajectories[name], runtimes[name] = _optimize_single(
            lidar,
            state_times,
            measurement_indices,
            information_positions,
            information_covariances,
            family,
        )
    for method in ("switch_irls", "dcs_irls"):
        name = f"grouped_physical_receivers__{method}"
        trajectories[name], runtimes[name] = _optimize_grouped_irls(
            lidar,
            state_times,
            measurement_indices,
            group_positions,
            group_covariances,
            method,
        )
    for first, second in combinations(group_names, 2):
        pair_positions = []
        pair_covariances = []
        for positions, covariances in zip(
            group_positions, group_covariances
        ):
            position, covariance = _aggregate(
                np.asarray([positions[first], positions[second]]),
                np.asarray([covariances[first], covariances[second]]),
                "geometric",
                "nonshrinking",
            )
            pair_positions.append(position)
            pair_covariances.append(covariance)
        name = f"fixed_pair_consensus__{first}__{second}__cauchy"
        trajectories[name], runtimes[name] = _optimize_single(
            lidar,
            state_times,
            measurement_indices,
            np.asarray(pair_positions),
            np.asarray(pair_covariances),
            "cauchy",
        )
    for group in group_names:
        positions = np.asarray(
            [epoch[group] for epoch in group_positions]
        )
        covariances = np.asarray(
            [epoch[group] for epoch in group_covariances]
        )
        for family in ("direct", "huber", "cauchy"):
            name = f"single_physical_receiver__{group}__{family}"
            trajectories[name], runtimes[name] = _optimize_single(
                lidar,
                state_times,
                measurement_indices,
                positions,
                covariances,
                family,
            )
    trajectories["dda_fc"], runtimes["dda_fc"] = _optimize_adjudicated(
        lidar,
        state_times,
        measurement_indices,
        group_positions,
        group_covariances,
        consensus_positions,
        consensus_covariances,
        windows,
    )
    dda_source = "windowed_mixed_factor_topology"

    hidden_path = input_paths[0].with_name(
        input_paths[0].name.replace(
            ".estimator_input.npz", ".hidden_labels.npz"
        )
    )
    with np.load(hidden_path, allow_pickle=False) as hidden:
        truth_timestamps = np.asarray(
            hidden["truth_timestamps_s"], dtype=float
        )
        truth_poses = np.asarray(hidden["truth_poses_enu"], dtype=float)
    reference_validity = None
    if args.reference_max_gap_s > 0.0:
        if args.reference_nearest_tolerance_s <= 0.0:
            raise ValueError(
                "partial-reference scoring requires a positive nearest "
                "tolerance"
            )
        right = np.searchsorted(truth_timestamps, state_times, side="left")
        inside = (right > 0) & (right < len(truth_timestamps))
        left = np.clip(right - 1, 0, len(truth_timestamps) - 1)
        right_clipped = np.clip(right, 0, len(truth_timestamps) - 1)
        gaps = truth_timestamps[right_clipped] - truth_timestamps[left]
        nearest = np.minimum(
            np.abs(state_times - truth_timestamps[left]),
            np.abs(truth_timestamps[right_clipped] - state_times),
        )
        reference_mask = (
            inside
            & (gaps <= args.reference_max_gap_s)
            & (nearest <= args.reference_nearest_tolerance_s)
        )
        truth = np.full((len(state_times), 3), np.nan, dtype=float)
        truth[reference_mask] = interpolate_poses(
            truth_timestamps,
            truth_poses,
            state_times[reference_mask],
        )[:, :3, 3]
        reference_validity = {
            "policy": "bracketed_reference_mask",
            "maximum_bracketing_gap_s": args.reference_max_gap_s,
            "nearest_reference_tolerance_s": (
                args.reference_nearest_tolerance_s
            ),
            "valid_state_count_before_initialization_cut": int(
                np.count_nonzero(reference_mask)
            ),
            "invalid_state_count_before_initialization_cut": int(
                len(reference_mask) - np.count_nonzero(reference_mask)
            ),
        }
    else:
        truth = interpolate_poses(
            truth_timestamps,
            truth_poses,
            state_times,
        )[:, :3, 3]
        reference_mask = np.ones(len(state_times), dtype=bool)
    evaluation_start = int(
        np.searchsorted(state_times, times[alignment.used_count])
    )
    evaluation_mask = reference_mask.copy()
    evaluation_mask[:evaluation_start] = False
    if not np.any(evaluation_mask):
        raise ValueError("no valid post-initialization reference states")
    metrics = {
        name: {
            **_metrics(
                trajectory[evaluation_mask], truth[evaluation_mask], 0
            ),
            "runtime_s": runtimes[name],
        }
        for name, trajectory in trajectories.items()
    }
    best_single = min(
        (
            (entry["rmse_3d_m"], name)
            for name, entry in metrics.items()
            if name.startswith("single_physical_receiver__")
        ),
        key=lambda item: item[0],
    )
    best_fixed_topology = min(
        (
            metrics["grouped_physical_receivers__direct"]["rmse_3d_m"],
            "grouped_physical_receivers__direct",
        ),
        (
            metrics["hierarchical_consensus__cauchy"]["rmse_3d_m"],
            "hierarchical_consensus__cauchy",
        ),
        key=lambda item: item[0],
    )
    dda = metrics["dda_fc"]["rmse_3d_m"]
    pass_rules = {
        "within_110pct_best_fixed_topology": bool(
            dda <= 1.10 * best_fixed_topology[0]
        ),
        "within_110pct_best_single": bool(dda <= 1.10 * best_single[0]),
        "improves_kiss_by_10pct": bool(
            dda <= 0.90 * metrics["aligned_kiss_icp"]["rmse_3d_m"]
        ),
        "improves_flat_information_direct_by_5pct_or_exception": bool(
            dda
            <= 0.95 * metrics["flat_information_mean__direct"]["rmse_3d_m"]
            or metrics["flat_information_mean__direct"]["rmse_3d_m"]
            <= 1.05 * best_single[0]
        ),
    }
    payload = {
        "evidentiary_status": "FROZEN_DDA_FC_PANEL",
        "epoch_policy": args.epoch_policy,
        "ablation": args.ablation,
        "truth_access_policy": (
            "All grouping, alignment, branch adjudication, receiver selection, "
            "and trajectory optimization completed before hidden truth was read."
        ),
        "physical_receiver_groups": group_names,
        "common_epoch_count": len(times),
        "state_epoch_count": len(state_times),
        "evaluation_start_index": evaluation_start,
        "evaluation_epoch_count": int(np.count_nonzero(evaluation_mask)),
        "reference_validity": reference_validity,
        "decision": decision,
        "dda_source_trajectory": dda_source,
        "proposed_runtime_breakdown_s": {
            "factor_construction": factor_construction_runtime_s,
            "truth_free_alignment": alignment_runtime_s,
            "topology_decision": decision_runtime_s,
            "final_graph_optimization": runtimes["dda_fc"],
            "sum_excluding_file_io": (
                factor_construction_runtime_s
                + alignment_runtime_s
                + decision_runtime_s
                + runtimes["dda_fc"]
            ),
        },
        "metrics": metrics,
        "best_single_physical_receiver": {
            "name": best_single[1],
            "rmse_3d_m": best_single[0],
        },
        "best_fixed_topology": {
            "name": best_fixed_topology[1],
            "rmse_3d_m": best_fixed_topology[0],
        },
        "registered_primary_pass_rules": pass_rules,
        "registered_primary_pass": bool(all(pass_rules.values())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
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
