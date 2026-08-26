"""Covariance-union construction for a singleton or receiver pair.

The returned pair ``(u, U)`` satisfies the covariance-union feasibility
constraints

    U >= P_i + (u - x_i)(u - x_i)^T

for every input estimate.  For the two-receiver case used by the paper, the
union center is searched on the segment joining the two means and the trace
of a feasible common upper bound is minimized.  The construction is a
deterministic CU baseline; it does not use ground truth or odometry.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


def _positive_part(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(_symmetrize(matrix))
    return (vectors * np.maximum(values, 0.0)) @ vectors.T


def _regularize(covariance: np.ndarray, floor_m: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(_symmetrize(covariance))
    return (vectors * np.maximum(values, floor_m**2)) @ vectors.T


def _trace_minimum_common_upper_bound(
    left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    """Return a trace-minimum PSD upper bound of two symmetric matrices."""

    left = _symmetrize(left)
    right = _symmetrize(right)
    return _symmetrize(left + _positive_part(right - left))


def pair_covariance_union(
    points: np.ndarray,
    covariances: np.ndarray,
    floor_m: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Construct a deterministic trace-optimized covariance union.

    The released method only proposes singleton or pair factors.  Supporting
    exactly that cardinality keeps the comparator explicit and prevents an
    order-dependent sequential union from being mistaken for a multi-input
    optimum.  The third return value is the optimized segment coordinate.
    """

    center, union, diagnostics = pair_covariance_union_diagnostics(
        points, covariances, floor_m=floor_m
    )
    return center, union, float(diagnostics["alpha"])


def pair_covariance_union_diagnostics(
    points: np.ndarray,
    covariances: np.ndarray,
    floor_m: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | bool | str]]:
    """Construct the pair CU and expose reproducibility diagnostics.

    The scalar parameter ``alpha`` is globally bounded to ``[0, 1]`` and
    solved with SciPy's bounded golden-section/parabolic search.  Returning
    solver status and feasibility makes the paper comparator auditable while
    preserving :func:`pair_covariance_union`'s original public interface.
    """

    points = np.asarray(points, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [receivers, 3]")
    if covariances.shape != (len(points), 3, 3):
        raise ValueError("covariance shape mismatch")
    if len(points) not in (1, 2):
        raise ValueError("covariance-union factor requires one or two receivers")
    if floor_m <= 0.0:
        raise ValueError("covariance floor must be positive")

    regularized = np.asarray(
        [_regularize(covariance, floor_m) for covariance in covariances]
    )
    if len(points) == 1:
        diagnostics: dict[str, float | int | bool | str] = {
            "alpha": 0.0,
            "success": True,
            "used_fallback": False,
            "function_evaluations": 0,
            "objective_trace_m2": float(np.trace(regularized[0])),
            "minimum_feasibility_eigenvalue_m2": 0.0,
            "search_lower": 0.0,
            "search_upper": 1.0,
            "absolute_alpha_tolerance": 1e-8,
            "maximum_iterations": 200,
            "objective": "trace",
            "solver": "scipy.optimize.minimize_scalar(method=bounded)",
        }
        return points[0].copy(), regularized[0], diagnostics

    direction = points[1] - points[0]

    def candidate(alpha: float) -> tuple[np.ndarray, np.ndarray]:
        center = points[0] + float(alpha) * direction
        shifted = []
        for point, covariance in zip(points, regularized):
            offset = center - point
            shifted.append(covariance + np.outer(offset, offset))
        union = _trace_minimum_common_upper_bound(shifted[0], shifted[1])
        return center, _regularize(union, floor_m)

    result = minimize_scalar(
        lambda alpha: float(np.trace(candidate(alpha)[1])),
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 200},
    )
    alpha = float(result.x) if result.success else 0.5
    center, union = candidate(alpha)
    minimum_feasibility = float(
        np.min(
            union_feasibility_eigenvalues(
                center, union, points, regularized
            )
        )
    )
    diagnostics = {
        "alpha": alpha,
        "success": bool(result.success),
        "used_fallback": bool(not result.success),
        "function_evaluations": int(getattr(result, "nfev", 0)),
        "objective_trace_m2": float(np.trace(union)),
        "minimum_feasibility_eigenvalue_m2": minimum_feasibility,
        "search_lower": 0.0,
        "search_upper": 1.0,
        "absolute_alpha_tolerance": 1e-8,
        "maximum_iterations": 200,
        "objective": "trace",
        "solver": "scipy.optimize.minimize_scalar(method=bounded)",
    }
    return center, union, diagnostics


def union_feasibility_eigenvalues(
    center: np.ndarray,
    union: np.ndarray,
    points: np.ndarray,
    covariances: np.ndarray,
) -> np.ndarray:
    """Return minimum CU-constraint eigenvalue for every member."""

    values = []
    for point, covariance in zip(points, covariances):
        offset = np.asarray(center) - np.asarray(point)
        difference = (
            np.asarray(union)
            - np.asarray(covariance)
            - np.outer(offset, offset)
        )
        values.append(float(np.min(np.linalg.eigvalsh(_symmetrize(difference)))))
    return np.asarray(values)
