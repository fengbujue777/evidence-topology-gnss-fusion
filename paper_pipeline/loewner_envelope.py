"""Provenance-group Loewner-envelope position factor construction."""

from __future__ import annotations

import numpy as np


def _regularize(covariance: np.ndarray, floor_m: float = 1.0) -> np.ndarray:
    covariance = 0.5 * (
        np.asarray(covariance, dtype=float)
        + np.asarray(covariance, dtype=float).T
    )
    values, vectors = np.linalg.eigh(covariance)
    return (vectors * np.maximum(values, floor_m**2)) @ vectors.T


def geometric_median(points: np.ndarray) -> np.ndarray:
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


def _positive_part(matrix: np.ndarray) -> np.ndarray:
    """Return the positive-semidefinite part of a symmetric matrix."""

    matrix = 0.5 * (
        np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T
    )
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.maximum(values, 0.0)) @ vectors.T


def provenance_loewner_envelope(
    points: np.ndarray,
    covariances: np.ndarray,
    floor_m: float = 1.0,
    center_mode: str = "geometric",
    envelope_mode: str = "positive_part",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return a group factor no more informative than any member covariance.

    Let B be the arithmetic marginal covariance plus the undivided group
    scatter.  The released method forms factors from a singleton or a pair;
    this matches the covariance center in the manuscript exactly.
    The default anisotropic loading

        Delta = sum_i (R_i - B)_+

    guarantees B + Delta >= R_i in the Loewner order for every stream i,
    because (R_i-B)_+ >= R_i-B.  ``isotropic`` instead uses the scalar
    loading gamma=max_i max(0, lambda_max(R_i-B)).  Exact replays must be
    removed before calling this function.

    ``arithmetic`` is the nominal, fixed convex center for which an arbitrary-
    dependence covariance upper-bound proof is available.  ``geometric`` is
    retained only as a robustness ablation; its data-dependent weights are not
    covered by that nominal proof.
    """

    points = np.asarray(points, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [streams, 3]")
    if covariances.shape != (len(points), 3, 3):
        raise ValueError("covariance shape mismatch")
    if center_mode == "arithmetic":
        center = np.mean(points, axis=0)
    elif center_mode == "geometric":
        center = geometric_median(points)
    else:
        raise ValueError(f"unsupported center mode: {center_mode}")
    deviations = points - center
    scatter = (
        deviations.T @ deviations / max(len(points) - 1, 1)
        if len(points) > 1
        else np.zeros((3, 3))
    )
    marginal_center = _regularize(
        np.mean(covariances, axis=0), floor_m=floor_m
    )
    base = _regularize(
        marginal_center + scatter, floor_m=floor_m
    )
    regularized_covariances = [
        _regularize(covariance, floor_m=floor_m)
        for covariance in covariances
    ]
    if envelope_mode == "positive_part":
        loading_matrix = sum(
            (
                _positive_part(covariance - base)
                for covariance in regularized_covariances
            ),
            start=np.zeros((3, 3), dtype=float),
        )
    elif envelope_mode == "isotropic":
        loading = max(
            0.0,
            max(
                float(np.max(np.linalg.eigvalsh(covariance - base)))
                for covariance in regularized_covariances
            ),
        )
        loading_matrix = loading * np.eye(3)
    else:
        raise ValueError(f"unsupported envelope mode: {envelope_mode}")
    envelope = _regularize(base + loading_matrix, floor_m=floor_m)
    return center, envelope, float(np.trace(loading_matrix))
