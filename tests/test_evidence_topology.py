from __future__ import annotations

import numpy as np

import topology_factor_graph as topology
from run_evidence_topology import _candidate_subset


def _window(
    motion: dict[str, float], pairs: dict[str, float]
) -> dict[str, object]:
    return {
        "receiver_relative_mismatch": {
            name: {"median": value} for name, value in motion.items()
        },
        "receiver_pair_separation": {
            name: {"median": value} for name, value in pairs.items()
        },
    }


def test_clear_kiss_motion_selects_one_receiver() -> None:
    subset, reason, _ = _candidate_subset(
        _window(
            {"a": 1.0, "b": 2.0, "c": 3.0},
            {"a__b": 4.0, "a__c": 4.5, "b__c": 5.0},
        ),
        ["a", "b", "c"],
        1.5,
        1.5,
    )
    assert subset == ["a"]
    assert reason == "kiss_motion_clear_single"


def test_ambiguous_motion_creates_two_receiver_subset() -> None:
    subset, reason, _ = _candidate_subset(
        _window(
            {"a": 1.0, "b": 1.2, "c": 3.0},
            {"a__b": 4.0, "a__c": 5.0, "b__c": 5.5},
        ),
        ["a", "b", "c"],
        1.5,
        1.5,
    )
    assert subset == ["a", "b"]
    assert reason == "kiss_motion_ambiguous_pair"


def test_separation_triangle_can_override_motion_ranking() -> None:
    subset, reason, _ = _candidate_subset(
        _window(
            {"a": 1.0, "b": 2.0, "c": 2.1},
            {"a__b": 20.0, "a__c": 18.0, "b__c": 2.0},
        ),
        ["a", "b", "c"],
        1.5,
        1.5,
    )
    assert subset == ["b", "c"]
    assert reason == "separation_triangle_coherent_pair"


def test_motion_first_ablation_reverses_priority_when_both_margins_are_clear() -> None:
    window = _window(
        {"a": 1.0, "b": 2.0, "c": 2.1},
        {"a__b": 20.0, "a__c": 18.0, "b__c": 2.0},
    )
    full_subset, full_reason, _ = _candidate_subset(
        window,
        ["a", "b", "c"],
        1.5,
        1.5,
        "full",
    )
    motion_first_subset, motion_first_reason, _ = _candidate_subset(
        window,
        ["a", "b", "c"],
        1.5,
        1.5,
        "motion_first",
    )
    assert full_subset == ["b", "c"]
    assert full_reason == "separation_triangle_coherent_pair"
    assert motion_first_subset == ["a"]
    assert motion_first_reason == "ablation_motion_first_clear_single"


def test_subset_topology_inserts_one_factor_per_epoch(monkeypatch) -> None:
    class Graph:
        def __init__(self) -> None:
            self.factors = []

        def add(self, factor) -> None:
            self.factors.append(factor)

    class Gtsam:
        @staticmethod
        def GPSFactor(key, position, noise):
            return key, np.asarray(position), noise

    graph = Graph()
    monkeypatch.setattr(
        topology, "_base_graph", lambda lidar, times: (graph, object())
    )
    monkeypatch.setattr(topology, "_gtsam", lambda: Gtsam())
    monkeypatch.setattr(topology, "_key", lambda index: ("x", index))
    monkeypatch.setattr(
        topology, "_gps_noise", lambda covariance, family: family
    )
    monkeypatch.setattr(topology, "_optimize", lambda graph, initial: object())
    monkeypatch.setattr(
        topology,
        "_trajectory",
        lambda result, count: np.zeros((count, 3)),
    )
    monkeypatch.setattr(
        topology,
        "_aggregate",
        lambda points, covariances, center, mode: (
            np.mean(points, axis=0),
            np.eye(3),
        ),
    )

    positions = [
        {"a": np.zeros(3), "b": np.ones(3), "c": 2.0 * np.ones(3)}
        for _ in range(2)
    ]
    covariances = [
        {"a": np.eye(3), "b": np.eye(3), "c": np.eye(3)}
        for _ in range(2)
    ]
    topology._optimize_adjudicated(
        np.repeat(np.eye(4)[None, :, :], 2, axis=0),
        np.array([0.0, 1.0]),
        np.array([0, 1]),
        positions,
        covariances,
        np.zeros((2, 3)),
        np.repeat(np.eye(3)[None, :, :], 2, axis=0),
        [
            {
                "start_epoch": 0,
                "stop_epoch": 2,
                "selected_receiver": None,
                "selected_receivers": ["a", "b"],
                "selected_family": "cauchy",
                "factor_topology": "receiver_subset_consensus",
            }
        ],
    )
    assert len(graph.factors) == 2


def test_independent_pair_ablation_inserts_two_factors_per_epoch(
    monkeypatch,
) -> None:
    class Graph:
        def __init__(self) -> None:
            self.factors = []

        def add(self, factor) -> None:
            self.factors.append(factor)

    class Gtsam:
        @staticmethod
        def GPSFactor(key, position, noise):
            return key, np.asarray(position), noise

    graph = Graph()
    monkeypatch.setattr(
        topology, "_base_graph", lambda lidar, times: (graph, object())
    )
    monkeypatch.setattr(topology, "_gtsam", lambda: Gtsam())
    monkeypatch.setattr(topology, "_key", lambda index: ("x", index))
    monkeypatch.setattr(
        topology, "_gps_noise", lambda covariance, family: family
    )
    monkeypatch.setattr(topology, "_optimize", lambda graph, initial: object())
    monkeypatch.setattr(
        topology,
        "_trajectory",
        lambda result, count: np.zeros((count, 3)),
    )

    positions = [
        {"a": np.zeros(3), "b": np.ones(3), "c": 2.0 * np.ones(3)}
        for _ in range(2)
    ]
    covariances = [
        {"a": np.eye(3), "b": np.eye(3), "c": np.eye(3)}
        for _ in range(2)
    ]
    topology._optimize_adjudicated(
        np.repeat(np.eye(4)[None, :, :], 2, axis=0),
        np.array([0.0, 1.0]),
        np.array([0, 1]),
        positions,
        covariances,
        np.zeros((2, 3)),
        np.repeat(np.eye(3)[None, :, :], 2, axis=0),
        [
            {
                "start_epoch": 0,
                "stop_epoch": 2,
                "selected_receiver": None,
                "selected_receivers": ["a", "b"],
                "selected_family": "cauchy",
                "factor_topology": "receiver_subset_independent",
            }
        ],
    )
    assert len(graph.factors) == 4
