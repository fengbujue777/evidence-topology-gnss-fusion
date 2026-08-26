import numpy as np

from paper_pipeline.covariance_union import (
    pair_covariance_union,
    pair_covariance_union_diagnostics,
    union_feasibility_eigenvalues,
)


def test_pair_union_satisfies_member_constraints():
    points = np.asarray([[0.0, 1.0, 2.0], [4.0, -1.0, 2.5]])
    covariances = np.asarray(
        [
            [[2.0, 0.3, 0.0], [0.3, 1.0, 0.1], [0.0, 0.1, 3.0]],
            [[1.0, -0.2, 0.0], [-0.2, 4.0, 0.2], [0.0, 0.2, 2.0]],
        ]
    )
    center, union, alpha = pair_covariance_union(
        points, covariances, floor_m=0.1
    )
    assert 0.0 <= alpha <= 1.0
    assert np.all(
        union_feasibility_eigenvalues(
            center, union, points, covariances
        )
        >= -1e-8
    )


def test_singleton_union_is_replay_invariant():
    point = np.asarray([[1.0, 2.0, 3.0]])
    covariance = np.asarray([np.diag([2.0, 3.0, 4.0])])
    center, union, alpha = pair_covariance_union(point, covariance)
    np.testing.assert_allclose(center, point[0])
    np.testing.assert_allclose(union, covariance[0])
    assert alpha == 0.0


def test_identical_pair_does_not_shrink_covariance():
    points = np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    covariance = np.diag([2.0, 3.0, 4.0])
    center, union, _ = pair_covariance_union(
        points, np.asarray([covariance, covariance])
    )
    np.testing.assert_allclose(center, points[0], atol=1e-8)
    np.testing.assert_allclose(union, covariance, atol=1e-8)


def test_pair_union_exposes_reproducible_solver_diagnostics():
    points = np.asarray([[0.0, 0.0, 0.0], [3.0, -1.0, 0.5]])
    covariances = np.asarray([np.eye(3), np.diag([2.0, 3.0, 4.0])])
    center, union, diagnostics = pair_covariance_union_diagnostics(
        points, covariances
    )
    assert diagnostics["success"] is True
    assert diagnostics["used_fallback"] is False
    assert diagnostics["function_evaluations"] > 0
    assert 0.0 <= diagnostics["alpha"] <= 1.0
    assert diagnostics["minimum_feasibility_eigenvalue_m2"] >= -1e-8
    assert np.all(
        union_feasibility_eigenvalues(
            center, union, points, covariances
        )
        >= -1e-8
    )
