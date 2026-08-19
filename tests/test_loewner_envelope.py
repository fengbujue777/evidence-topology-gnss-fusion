from __future__ import annotations

import numpy as np

from paper_pipeline.loewner_envelope import provenance_loewner_envelope


def test_envelope_dominates_every_input_covariance() -> None:
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, -0.2, 0.3], [-0.4, 0.5, 0.1]]
    )
    covariances = np.array(
        [
            np.diag([1.0, 2.0, 3.0]),
            np.diag([5.0, 1.0, 2.0]),
            np.array(
                [[2.0, 0.7, 0.0], [0.7, 4.0, 0.2], [0.0, 0.2, 1.5]]
            ),
        ]
    )
    _, envelope, _ = provenance_loewner_envelope(
        points, covariances, center_mode="arithmetic"
    )
    for covariance in covariances:
        assert np.min(np.linalg.eigvalsh(envelope - covariance)) >= -1e-8


def test_exact_replay_after_deduplication_is_invariant() -> None:
    point = np.array([[1.0, 2.0, 3.0]])
    covariance = np.array([np.diag([2.0, 3.0, 4.0])])
    center, envelope, _ = provenance_loewner_envelope(point, covariance)
    replay_center, replay_envelope, _ = provenance_loewner_envelope(
        point.copy(), covariance.copy()
    )
    assert np.array_equal(center, replay_center)
    assert np.array_equal(envelope, replay_envelope)


def test_arithmetic_center_is_fixed_convex_combination() -> None:
    points = np.asarray([[1.0, 2.0, 3.0], [5.0, 4.0, 3.0]])
    covariances = np.asarray([np.eye(3), 2.0 * np.eye(3)])
    center, _, _ = provenance_loewner_envelope(
        points, covariances, center_mode="arithmetic"
    )
    np.testing.assert_allclose(center, [3.0, 3.0, 3.0])


def test_pair_envelope_bounds_mean_under_unknown_cross_correlation() -> None:
    """The pair factor remains conservative for any valid cross-covariance."""

    rng = np.random.default_rng(20260807)
    for _ in range(100):
        left = rng.normal(size=(3, 3))
        right = rng.normal(size=(3, 3))
        covariance_left = left @ left.T + np.eye(3)
        covariance_right = right @ right.T + np.eye(3)

        # C = L1 K L2^T yields a positive-semidefinite joint covariance when
        # the spectral norm of K is at most one.
        contraction = rng.normal(size=(3, 3))
        contraction /= max(np.linalg.svd(contraction, compute_uv=False)[0], 1.0)
        chol_left = np.linalg.cholesky(covariance_left)
        chol_right = np.linalg.cholesky(covariance_right)
        cross = chol_left @ contraction @ chol_right.T
        mean_covariance = (
            covariance_left + covariance_right + cross + cross.T
        ) / 4.0

        _, envelope, _ = provenance_loewner_envelope(
            np.zeros((2, 3)),
            np.asarray([covariance_left, covariance_right]),
            center_mode="arithmetic",
        )
        assert np.min(np.linalg.eigvalsh(envelope - mean_covariance)) >= -1e-8
