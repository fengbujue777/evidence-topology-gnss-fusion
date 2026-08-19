"""Truth-free contrast identification for correlated position solution streams.

The unknown platform position cancels from same-epoch stream contrasts.  Given
receiver-reported marginal covariance blocks, the covariance of those
contrasts identifies the symmetric cross blocks of a joint covariance model.
This module turns that model into one generalized-least-squares position factor.

The implementation deliberately keeps the statistical mechanism independent
from GTSAM so that replication invariance and covariance reconstruction can be
unit tested without the nonlinear backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContrastModel:
    """Frozen prefix model used to construct later position factors."""

    relative_biases: np.ndarray
    joint_covariance: np.ndarray
    shrinkage: float
    validation_nll: dict[float, float]
    effective_rank: int


def regularize_covariance(
    covariance: np.ndarray, floor: float = 1e-6
) -> np.ndarray:
    covariance = 0.5 * (
        np.asarray(covariance, dtype=float)
        + np.asarray(covariance, dtype=float).T
    )
    values, vectors = np.linalg.eigh(covariance)
    return (vectors * np.maximum(values, floor)) @ vectors.T


def _geometric_median(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    center = np.median(points, axis=0)
    for _ in range(100):
        distances = np.linalg.norm(points - center, axis=1)
        if np.min(distances) < 1e-12:
            return points[int(np.argmin(distances))].copy()
        weights = 1.0 / np.maximum(distances, 1e-12)
        updated = np.sum(weights[:, None] * points, axis=0) / np.sum(weights)
        if np.linalg.norm(updated - center) < 1e-10:
            return updated
        center = updated
    return center


def estimate_relative_biases(points: np.ndarray) -> np.ndarray:
    """Estimate only stream-relative offsets; the common offset remains free."""

    points = np.asarray(points, dtype=float)
    epoch_centers = np.asarray([_geometric_median(epoch) for epoch in points])
    offsets = points - epoch_centers[:, None, :]
    biases = np.median(offsets, axis=0)
    # Fix the otherwise arbitrary common translation gauge.
    return biases - np.mean(biases, axis=0, keepdims=True)


def robust_covariance(samples: np.ndarray) -> np.ndarray:
    """Median-centered, radially Winsorized covariance.

    The radial cap is data-derived (median plus three scaled MADs), so it does
    not use trajectory truth or a dataset-specific distance threshold.
    """

    samples = np.asarray(samples, dtype=float)
    if len(samples) < 4:
        return np.eye(samples.shape[1])
    center = np.median(samples, axis=0)
    deviations = samples - center
    radii = np.linalg.norm(deviations, axis=1)
    radial_median = float(np.median(radii))
    radial_mad = float(np.median(np.abs(radii - radial_median)))
    cap = radial_median + 3.0 * 1.4826 * max(radial_mad, 1e-9)
    scales = np.minimum(1.0, cap / np.maximum(radii, 1e-12))
    bounded = deviations * scales[:, None]
    covariance = bounded.T @ bounded / max(len(bounded) - 1, 1)
    return regularize_covariance(covariance)


def _raw_joint_covariance(
    points: np.ndarray, reported_covariances: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray]]:
    points = np.asarray(points, dtype=float)
    reported_covariances = np.asarray(reported_covariances, dtype=float)
    stream_count = points.shape[1]
    blocks = [
        regularize_covariance(
            np.median(reported_covariances[:, stream], axis=0), floor=1.0
        )
        for stream in range(stream_count)
    ]
    joint = np.zeros((3 * stream_count, 3 * stream_count), dtype=float)
    for stream, block in enumerate(blocks):
        slc = slice(3 * stream, 3 * (stream + 1))
        joint[slc, slc] = block
    for left in range(stream_count):
        left_slice = slice(3 * left, 3 * (left + 1))
        for right in range(left + 1, stream_count):
            right_slice = slice(3 * right, 3 * (right + 1))
            difference_covariance = robust_covariance(
                points[:, left] - points[:, right]
            )
            cross = 0.5 * (
                blocks[left] + blocks[right] - difference_covariance
            )
            cross = 0.5 * (cross + cross.T)
            joint[left_slice, right_slice] = cross
            joint[right_slice, left_slice] = cross.T
    return joint, blocks


def _project_psd_with_fixed_blocks(
    matrix: np.ndarray,
    diagonal_blocks: list[np.ndarray],
    iterations: int = 100,
) -> np.ndarray:
    """Alternating projection onto PSD matrices and fixed diagonal blocks."""

    stream_count = len(diagonal_blocks)
    estimate = 0.5 * (matrix + matrix.T)
    correction = np.zeros_like(estimate)
    for _ in range(iterations):
        candidate = estimate - correction
        values, vectors = np.linalg.eigh(0.5 * (candidate + candidate.T))
        psd = (vectors * np.maximum(values, 0.0)) @ vectors.T
        correction = psd - candidate
        updated = psd
        for stream in range(stream_count):
            slc = slice(3 * stream, 3 * (stream + 1))
            updated[slc, slc] = diagonal_blocks[stream]
        updated = 0.5 * (updated + updated.T)
        if np.linalg.norm(updated - estimate, ord="fro") < 1e-9:
            estimate = updated
            break
        estimate = updated
    # A final PSD projection may move diagonal blocks by roundoff only.  It is
    # preferable to returning a slightly indefinite noise model.
    values, vectors = np.linalg.eigh(0.5 * (estimate + estimate.T))
    return (vectors * np.maximum(values, 0.0)) @ vectors.T


def _shrunken_joint(
    raw: np.ndarray,
    diagonal_blocks: list[np.ndarray],
    shrinkage: float,
) -> np.ndarray:
    independent = np.zeros_like(raw)
    for stream, block in enumerate(diagonal_blocks):
        slc = slice(3 * stream, 3 * (stream + 1))
        independent[slc, slc] = block
    candidate = (
        float(shrinkage) * raw
        + (1.0 - float(shrinkage)) * independent
    )
    return _project_psd_with_fixed_blocks(candidate, diagonal_blocks)


def _contrast_nll(points: np.ndarray, joint: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    stream_count = points.shape[1]
    terms = []
    for left in range(stream_count):
        left_slice = slice(3 * left, 3 * (left + 1))
        for right in range(left + 1, stream_count):
            right_slice = slice(3 * right, 3 * (right + 1))
            predicted = (
                joint[left_slice, left_slice]
                + joint[right_slice, right_slice]
                - joint[left_slice, right_slice]
                - joint[right_slice, left_slice]
            )
            predicted = regularize_covariance(predicted, floor=1e-6)
            inverse = np.linalg.inv(predicted)
            logdet = float(np.linalg.slogdet(predicted)[1])
            differences = points[:, left] - points[:, right]
            differences -= np.median(differences, axis=0)
            quadratic = np.einsum(
                "ni,ij,nj->n", differences, inverse, differences
            )
            terms.extend((quadratic + logdet).tolist())
    return float(np.mean(terms)) if terms else 0.0


def fit_contrast_model(
    calibration_points: np.ndarray,
    calibration_covariances: np.ndarray,
    shrinkages: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> ContrastModel:
    """Fit with a prefix-only train/validation split and no position truth."""

    points = np.asarray(calibration_points, dtype=float)
    covariances = np.asarray(calibration_covariances, dtype=float)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("points must have shape [epochs, streams, 3]")
    if covariances.shape != (*points.shape[:2], 3, 3):
        raise ValueError("covariances must have shape [epochs, streams, 3, 3]")
    if len(points) < 20:
        raise ValueError("at least 20 calibration epochs are required")

    biases = estimate_relative_biases(points)
    corrected = points - biases[None, :, :]
    split = max(10, int(np.floor(0.7 * len(points))))
    split = min(split, len(points) - 5)
    raw_train, diagonal_blocks = _raw_joint_covariance(
        corrected[:split], covariances[:split]
    )
    validation_nll = {}
    for shrinkage in shrinkages:
        candidate = _shrunken_joint(
            raw_train, diagonal_blocks, shrinkage
        )
        validation_nll[float(shrinkage)] = _contrast_nll(
            corrected[split:], candidate
        )
    selected = min(validation_nll, key=validation_nll.get)

    raw_full, diagonal_blocks = _raw_joint_covariance(
        corrected, covariances
    )
    joint = _shrunken_joint(raw_full, diagonal_blocks, selected)
    values = np.linalg.eigvalsh(joint)
    threshold = max(float(np.max(values)) * 1e-9, 1e-9)
    return ContrastModel(
        relative_biases=biases,
        joint_covariance=joint,
        shrinkage=float(selected),
        validation_nll=validation_nll,
        effective_rank=int(np.count_nonzero(values > threshold)),
    )


def condensed_gls_factor(
    points: np.ndarray,
    model: ContrastModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one position and covariance with singular-safe GLS."""

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [streams, 3]")
    stream_count = len(points)
    if model.relative_biases.shape != points.shape:
        raise ValueError("stream count does not match the fitted model")
    measurement = (
        points - model.relative_biases
    ).reshape(3 * stream_count)
    observation = np.kron(np.ones((stream_count, 1)), np.eye(3))
    inverse = np.linalg.pinv(
        model.joint_covariance, rcond=1e-9, hermitian=True
    )
    information = observation.T @ inverse @ observation
    covariance = np.linalg.pinv(information, rcond=1e-9, hermitian=True)
    position = covariance @ observation.T @ inverse @ measurement
    return position, regularize_covariance(covariance, floor=1e-6)
