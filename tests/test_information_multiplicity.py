from __future__ import annotations

import numpy as np

from validate_information_multiplicity import (
    ChainNoise,
    _ldlt_solve_and_marginal,
    _method_factors,
)
from run_ci_baseline import _covariance_intersection


def test_ldlt_solution_and_marginal_match_dense_inverse() -> None:
    diagonal = np.asarray([4.0, 5.0, 6.0, 5.5])
    off_diagonal = np.asarray([-1.0, -0.8, -1.2])
    rhs = np.asarray([1.0, 2.0, -1.0, 0.5])
    dense = np.diag(diagonal)
    dense += np.diag(off_diagonal, 1)
    dense += np.diag(off_diagonal, -1)
    solution, marginal = _ldlt_solve_and_marginal(
        diagonal, off_diagonal, rhs
    )
    np.testing.assert_allclose(solution, np.linalg.solve(dense, rhs))
    np.testing.assert_allclose(marginal, np.diag(np.linalg.inv(dense)))


def test_provenance_factor_is_exact_replay_invariant() -> None:
    rng = np.random.default_rng(12)
    stream = rng.normal(size=(1, 20, 3))
    one = stream[None, :, :, :]
    eight = np.repeat(one, 8, axis=1)
    noise = ChainNoise()
    first_position, first_covariance, _ = _method_factors(
        one, "provenance_capped", noise, rho=1.0, physical_group_count=1
    )
    repeated_position, repeated_covariance, _ = _method_factors(
        eight,
        "provenance_capped",
        noise,
        rho=1.0,
        physical_group_count=1,
    )
    np.testing.assert_array_equal(first_position, repeated_position)
    np.testing.assert_array_equal(first_covariance, repeated_covariance)


def test_covariance_intersection_is_exact_replay_invariant() -> None:
    point = np.asarray([[1.0, -2.0, 0.5]])
    covariance = np.asarray([np.diag([2.0, 3.0, 4.0])])
    first_position, first_covariance, _ = _covariance_intersection(
        point, covariance
    )
    repeated_position, repeated_covariance, _ = _covariance_intersection(
        np.repeat(point, 8, axis=0),
        np.repeat(covariance, 8, axis=0),
    )
    np.testing.assert_array_equal(first_position, repeated_position)
    np.testing.assert_array_equal(first_covariance, repeated_covariance)


def test_covariance_intersection_weights_form_simplex() -> None:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 2.0, -1.0], [-1.0, 0.5, 2.0]]
    )
    covariances = np.asarray(
        [
            np.diag([4.0, 2.0, 3.0]),
            np.diag([2.0, 5.0, 3.0]),
            np.diag([3.0, 3.0, 2.0]),
        ]
    )
    _, covariance, weights = _covariance_intersection(
        points, covariances
    )
    assert np.isclose(np.sum(weights), 1.0)
    assert np.all(weights >= 0.0)
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)
