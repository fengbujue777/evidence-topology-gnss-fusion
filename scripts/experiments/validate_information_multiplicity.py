"""No-fault validation of GNSS solution-stream information multiplicity.

The experiment isolates cross-stream dependence from fault detection.  Every
GNSS observation is zero-mean Gaussian and has the same marginal covariance.
Only the number of solution streams, their common-source correlation, and
their physical-source grouping are changed.

The position chain uses a public UrbanNav ground-truth trajectory as motion
geometry.  Odometry and GNSS observations are generated from a fully known
linear-Gaussian model, so posterior consistency can be evaluated with
normalized estimation error squared (ANEES) and 95% marginal coverage.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from paper_pipeline.alignment import interpolate_poses


METHODS = (
    "flat_gaussian",
    "flat_huber",
    "flat_cauchy",
    "global_ci",
    "provenance_capped",
    "oracle_gls",
)


@dataclass(frozen=True)
class ChainNoise:
    prior_sigma_m: float = 0.20
    odometry_sigma_m: float = 0.20
    gnss_sigma_m: float = 2.00


def _urban_truth(
    hidden_labels: Path, state_count: int, duration_s: float
) -> np.ndarray:
    with np.load(hidden_labels, allow_pickle=False) as archive:
        timestamps = np.asarray(archive["truth_timestamps_s"], dtype=float)
        poses = np.asarray(archive["truth_poses_enu"], dtype=float)
    usable_duration = min(duration_s, float(timestamps[-1] - timestamps[0]))
    target = np.linspace(
        timestamps[0], timestamps[0] + usable_duration, state_count
    )
    return interpolate_poses(timestamps, poses, target)[:, :3, 3]


def _ldlt_solve_and_marginal(
    diagonal: np.ndarray, off_diagonal: np.ndarray, rhs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Solve an SPD tridiagonal system and return diag(A^-1)."""

    count = len(diagonal)
    d = np.empty(count, dtype=float)
    lower = np.empty(count, dtype=float)
    d[0] = diagonal[0]
    lower[0] = 0.0
    for index in range(1, count):
        lower[index] = off_diagonal[index - 1] / d[index - 1]
        d[index] = (
            diagonal[index]
            - lower[index] * off_diagonal[index - 1]
        )
        if d[index] <= 0.0:
            raise np.linalg.LinAlgError("non-positive tridiagonal pivot")

    forward = np.empty(count, dtype=float)
    forward[0] = rhs[0]
    for index in range(1, count):
        forward[index] = rhs[index] - lower[index] * forward[index - 1]
    scaled = forward / d
    solution = np.empty(count, dtype=float)
    solution[-1] = scaled[-1]
    for index in range(count - 2, -1, -1):
        solution[index] = (
            scaled[index] - lower[index + 1] * solution[index + 1]
        )

    marginal = np.empty(count, dtype=float)
    marginal[-1] = 1.0 / d[-1]
    for index in range(count - 2, -1, -1):
        marginal[index] = (
            1.0 / d[index]
            + lower[index + 1] ** 2 * marginal[index + 1]
        )
    return solution, marginal


def _base_information(
    prior: float,
    odometry: np.ndarray,
    noise: ChainNoise,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(odometry) + 1
    diagonal = np.zeros(count, dtype=float)
    off_diagonal = np.full(
        count - 1, -1.0 / noise.odometry_sigma_m**2, dtype=float
    )
    rhs = np.zeros(count, dtype=float)
    prior_information = 1.0 / noise.prior_sigma_m**2
    odometry_information = 1.0 / noise.odometry_sigma_m**2
    diagonal[0] += prior_information
    rhs[0] += prior_information * prior
    for index, displacement in enumerate(odometry, start=1):
        diagonal[index - 1] += odometry_information
        diagonal[index] += odometry_information
        rhs[index - 1] -= odometry_information * displacement
        rhs[index] += odometry_information * displacement
    return diagonal, off_diagonal, rhs


def _robust_weights(
    residual: np.ndarray, family: str
) -> np.ndarray:
    absolute = np.abs(residual)
    if family == "gaussian":
        return np.ones_like(absolute)
    if family == "huber":
        delta = 1.345
        return np.where(
            absolute <= delta, 1.0, delta / np.maximum(absolute, 1e-12)
        )
    if family == "cauchy":
        scale = 2.3849
        return 1.0 / (1.0 + (residual / scale) ** 2)
    raise ValueError(family)


def _solve_dimension(
    prior: float,
    odometry: np.ndarray,
    observations: np.ndarray,
    variances: np.ndarray,
    noise: ChainNoise,
    family: str,
    iterations: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve one coordinate of a chain with possibly robust GNSS factors.

    observations and variances have shape (factor_count, state_count).
    """

    base_diagonal, off_diagonal, base_rhs = _base_information(
        prior, odometry, noise
    )
    factor_count, state_count = observations.shape
    estimate = np.zeros(state_count, dtype=float)
    weights = np.ones((factor_count, state_count), dtype=float)
    iteration_count = 1 if family == "gaussian" else iterations
    for _ in range(iteration_count):
        information = weights / variances
        diagonal = base_diagonal + np.sum(information, axis=0)
        rhs = base_rhs + np.sum(information * observations, axis=0)
        estimate, marginal = _ldlt_solve_and_marginal(
            diagonal, off_diagonal, rhs
        )
        standardized = (
            estimate[None, :] - observations
        ) / np.sqrt(variances)
        updated = _robust_weights(standardized, family)
        if np.max(np.abs(updated - weights)) < 1e-7:
            weights = updated
            break
        weights = updated

    information = weights / variances
    diagonal = base_diagonal + np.sum(information, axis=0)
    rhs = base_rhs + np.sum(information * observations, axis=0)
    return _ldlt_solve_and_marginal(diagonal, off_diagonal, rhs)


def _deduplicate_streams(streams: np.ndarray) -> np.ndarray:
    """Drop byte-equivalent full streams while preserving first occurrence."""

    flattened = np.ascontiguousarray(streams).reshape(len(streams), -1)
    _, indices = np.unique(flattened, axis=0, return_index=True)
    return streams[np.sort(indices)]


def _method_factors(
    streams: np.ndarray,
    method: str,
    noise: ChainNoise,
    rho: float,
    physical_group_count: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Construct scalar-coordinate factors for one method.

    streams shape: (physical_group_count, streams_per_group, states, 3).
    Returned observations and variances have shape (factor_count, states, 3).
    """

    variance = noise.gnss_sigma_m**2
    group_count, streams_per_group, state_count, dimensions = streams.shape
    if group_count != physical_group_count:
        raise ValueError("physical group count mismatch")

    if method.startswith("flat_"):
        observations = streams.reshape(
            group_count * streams_per_group, state_count, dimensions
        )
        variances = np.full_like(observations, variance)
        family = method.removeprefix("flat_")
        return observations, variances, family

    if method == "global_ci":
        unique = _deduplicate_streams(
            streams.reshape(
                group_count * streams_per_group, state_count, dimensions
            )
        )
        # Equal-weight covariance intersection for equal marginal
        # covariances: the weights sum to one, so covariance cannot shrink.
        center = np.mean(unique, axis=0)
        return (
            center[None, :, :],
            np.full_like(center[None, :, :], variance),
            "gaussian",
        )

    centers = []
    group_variances = []
    for group_streams in streams:
        unique = _deduplicate_streams(group_streams)
        center = np.mean(unique, axis=0)
        centers.append(center)
        if method == "oracle_gls":
            multiplicity = len(unique)
            aggregate_variance = variance * (
                rho + (1.0 - rho) / multiplicity
            )
            group_variances.append(
                np.full_like(center, aggregate_variance)
            )
        elif method == "provenance_capped":
            if len(unique) == 1:
                scatter = np.zeros_like(center)
            else:
                scatter = np.var(unique, axis=0, ddof=1)
            group_variances.append(variance + scatter)
        else:
            raise ValueError(method)
    return (
        np.asarray(centers),
        np.asarray(group_variances),
        "gaussian",
    )


def _solve_method(
    truth: np.ndarray,
    prior: np.ndarray,
    odometry: np.ndarray,
    streams: np.ndarray,
    method: str,
    noise: ChainNoise,
    rho: float,
) -> tuple[np.ndarray, np.ndarray]:
    observations, variances, family = _method_factors(
        streams,
        method,
        noise,
        rho,
        physical_group_count=len(streams),
    )
    estimate = np.empty_like(truth)
    marginal = np.empty_like(truth)
    for dimension in range(3):
        estimate[:, dimension], marginal[:, dimension] = _solve_dimension(
            prior[dimension],
            odometry[:, dimension],
            observations[:, :, dimension],
            variances[:, :, dimension],
            noise,
            family,
        )
    return estimate, marginal


def _dependent_streams(
    truth: np.ndarray,
    common: np.ndarray,
    independent: np.ndarray,
    rho: float,
    multiplicity: int,
    noise: ChainNoise,
) -> np.ndarray:
    errors = noise.gnss_sigma_m * (
        np.sqrt(rho) * common[None, :, :]
        + np.sqrt(1.0 - rho) * independent[:multiplicity]
    )
    return (truth[None, :, :] + errors)[None, :, :, :]


def _independent_groups(
    truth: np.ndarray,
    independent: np.ndarray,
    group_count: int,
    noise: ChainNoise,
) -> np.ndarray:
    errors = noise.gnss_sigma_m * independent[:group_count]
    return (truth[None, :, :] + errors)[:, None, :, :]


def _accumulate(
    accumulator: dict[tuple[str, str, float, int], dict[str, list[float]]],
    scenario: str,
    method: str,
    rho: float,
    multiplicity: int,
    truth: np.ndarray,
    estimate: np.ndarray,
    marginal: np.ndarray,
    evaluation_start: int,
) -> None:
    error = estimate[evaluation_start:] - truth[evaluation_start:]
    variance = marginal[evaluation_start:]
    squared_norm = np.sum(error**2, axis=1)
    mahalanobis = np.sum(error**2 / variance, axis=1)
    key = (scenario, method, rho, multiplicity)
    bucket = accumulator.setdefault(
        key,
        {
            "squared_error": [],
            "anees": [],
            "coverage": [],
            "posterior_sigma": [],
        },
    )
    bucket["squared_error"].extend(squared_norm.tolist())
    bucket["anees"].extend((mahalanobis / 3.0).tolist())
    bucket["coverage"].extend((mahalanobis <= 7.814727903251179).tolist())
    bucket["posterior_sigma"].extend(
        np.sqrt(np.mean(variance, axis=1)).tolist()
    )


def _summarize(
    accumulator: dict[tuple[str, str, float, int], dict[str, list[float]]]
) -> list[dict[str, object]]:
    rows = []
    for key in sorted(accumulator):
        scenario, method, rho, multiplicity = key
        values = accumulator[key]
        rows.append(
            {
                "scenario": scenario,
                "method": method,
                "rho": rho,
                "multiplicity": multiplicity,
                "rmse_3d_m": float(
                    np.sqrt(np.mean(values["squared_error"]))
                ),
                "normalized_anees": float(np.mean(values["anees"])),
                "coverage_95": float(np.mean(values["coverage"])),
                "mean_posterior_sigma_m": float(
                    np.mean(values["posterior_sigma"])
                ),
                "sample_count": len(values["anees"]),
            }
        )
    return rows


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-labels", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=120)
    parser.add_argument("--states", type=int, default=90)
    parser.add_argument("--duration-s", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()

    truth = _urban_truth(
        args.hidden_labels, args.states, args.duration_s
    )
    noise = ChainNoise()
    rhos = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
    multiplicities = (1, 2, 4, 8)
    evaluation_start = 5
    accumulator: dict[
        tuple[str, str, float, int], dict[str, list[float]]
    ] = {}
    paired_trajectories: dict[
        tuple[int, str, float, int], np.ndarray
    ] = {}

    for seed_offset in range(args.seeds):
        rng = np.random.default_rng(args.seed + seed_offset)
        prior = truth[0] + rng.normal(
            0.0, noise.prior_sigma_m, size=3
        )
        odometry = np.diff(truth, axis=0) + rng.normal(
            0.0,
            noise.odometry_sigma_m,
            size=(len(truth) - 1, 3),
        )
        common = rng.normal(size=truth.shape)
        independent = rng.normal(size=(8, *truth.shape))

        for rho in rhos:
            for multiplicity in multiplicities:
                streams = _dependent_streams(
                    truth,
                    common,
                    independent,
                    rho,
                    multiplicity,
                    noise,
                )
                for method in METHODS:
                    estimate, marginal = _solve_method(
                        truth,
                        prior,
                        odometry,
                        streams,
                        method,
                        noise,
                        rho,
                    )
                    _accumulate(
                        accumulator,
                        "one_dependent_physical_source",
                        method,
                        rho,
                        multiplicity,
                        truth,
                        estimate,
                        marginal,
                        evaluation_start,
                    )
                    if rho == 1.0 and multiplicity in (1, 8):
                        paired_trajectories[
                            (seed_offset, method, rho, multiplicity)
                        ] = estimate

        for group_count in multiplicities:
            streams = _independent_groups(
                truth, independent, group_count, noise
            )
            for method in METHODS:
                estimate, marginal = _solve_method(
                    truth,
                    prior,
                    odometry,
                    streams,
                    method,
                    noise,
                    rho=0.0,
                )
                _accumulate(
                    accumulator,
                    "independent_physical_sources",
                    method,
                    0.0,
                    group_count,
                    truth,
                    estimate,
                    marginal,
                    evaluation_start,
                )

    rows = _summarize(accumulator)
    output = args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "no_fault_multiplicity_results.csv", rows)

    invariance_rows = []
    for method in METHODS:
        changes = []
        for seed_offset in range(args.seeds):
            one = paired_trajectories[(seed_offset, method, 1.0, 1)]
            eight = paired_trajectories[(seed_offset, method, 1.0, 8)]
            changes.append(
                float(np.max(np.linalg.norm(eight - one, axis=1)))
            )
        invariance_rows.append(
            {
                "method": method,
                "mean_max_trajectory_change_m": float(np.mean(changes)),
                "p95_max_trajectory_change_m": float(
                    np.percentile(changes, 95.0)
                ),
                "maximum_trajectory_change_m": float(np.max(changes)),
            }
        )
    _write_csv(output / "rho1_replication_invariance.csv", invariance_rows)

    def row_for(
        scenario: str, method: str, rho: float, multiplicity: int
    ) -> dict[str, object]:
        return next(
            row
            for row in rows
            if row["scenario"] == scenario
            and row["method"] == method
            and row["rho"] == rho
            and row["multiplicity"] == multiplicity
        )

    dependent_high = {
        method: row_for(
            "one_dependent_physical_source", method, 0.9, 8
        )
        for method in METHODS
    }
    independent_one = row_for(
        "independent_physical_sources",
        "provenance_capped",
        0.0,
        1,
    )
    independent_eight = row_for(
        "independent_physical_sources",
        "provenance_capped",
        0.0,
        8,
    )
    independent_global_ci_eight = row_for(
        "independent_physical_sources",
        "global_ci",
        0.0,
        8,
    )
    capped_invariance = next(
        row
        for row in invariance_rows
        if row["method"] == "provenance_capped"
    )
    checks = {
        "capped_exact_replication_invariant_1e9_m": bool(
            capped_invariance["maximum_trajectory_change_m"] <= 1e-9
        ),
        "flat_gaussian_overconfident_at_rho09_m8": bool(
            dependent_high["flat_gaussian"]["normalized_anees"] > 1.5
            and dependent_high["flat_gaussian"]["coverage_95"] < 0.90
        ),
        "flat_huber_does_not_remove_inlier_multiplicity": bool(
            dependent_high["flat_huber"]["normalized_anees"] > 1.3
            and dependent_high["flat_huber"]["coverage_95"] < 0.92
        ),
        "flat_cauchy_does_not_remove_inlier_multiplicity": bool(
            dependent_high["flat_cauchy"]["normalized_anees"] > 1.2
            and dependent_high["flat_cauchy"]["coverage_95"] < 0.93
        ),
        "capped_is_not_overconfident_at_rho09_m8": bool(
            dependent_high["provenance_capped"]["normalized_anees"] <= 1.1
            and dependent_high["provenance_capped"]["coverage_95"] >= 0.94
        ),
        "capped_rmse_better_than_flat_at_rho09_m8": bool(
            dependent_high["provenance_capped"]["rmse_3d_m"]
            < dependent_high["flat_gaussian"]["rmse_3d_m"]
        ),
        "independent_sources_are_not_collapsed": bool(
            independent_eight["rmse_3d_m"]
            < independent_one["rmse_3d_m"]
            and independent_eight["mean_posterior_sigma_m"]
            < independent_one["mean_posterior_sigma_m"]
        ),
        "provenance_structure_uses_independence_better_than_global_ci": bool(
            independent_eight["rmse_3d_m"]
            < independent_global_ci_eight["rmse_3d_m"]
            and independent_eight["mean_posterior_sigma_m"]
            < independent_global_ci_eight["mean_posterior_sigma_m"]
        ),
        "oracle_is_calibrated_at_rho09_m8": bool(
            0.8 <= dependent_high["oracle_gls"]["normalized_anees"] <= 1.2
            and dependent_high["oracle_gls"]["coverage_95"] >= 0.93
        ),
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "experiment_type": (
            "NO_FAULT_CONTROLLED_INFORMATION_MULTIPLICITY_VALIDATION"
        ),
        "fault_injection": False,
        "truth_geometry_source": str(args.hidden_labels),
        "seeds": args.seeds,
        "state_count": args.states,
        "evaluation_start": evaluation_start,
        "noise": {
            "prior_sigma_m": noise.prior_sigma_m,
            "odometry_sigma_m": noise.odometry_sigma_m,
            "gnss_sigma_m": noise.gnss_sigma_m,
        },
        "rho_values": list(rhos),
        "multiplicities": list(multiplicities),
        "checks": checks,
        "dependent_rho09_m8": dependent_high,
        "independent_source_control": {
            "one_source": independent_one,
            "eight_sources": independent_eight,
            "global_ci_eight_sources": independent_global_ci_eight,
        },
        "replication_invariance": invariance_rows,
        "claim_boundary": (
            "The experiment validates information-count control when physical "
            "source provenance is known. It does not claim automatic discovery "
            "of arbitrary unknown cross-sensor dependence."
        ),
    }
    (output / "NO_FAULT_MULTIPLICITY_CERTIFICATE.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
