from __future__ import annotations

from pathlib import Path

import pytest

from paper_pipeline.provenance import ProvenanceRegistry


def test_explicit_streams_share_one_physical_receiver() -> None:
    registry = ProvenanceRegistry(
        {
            "m8t_gc": "receiver_m8t",
            "m8t_gej": "receiver_m8t",
        }
    )
    assert (
        registry.physical_source(
            Path("route__m8t_gc__real_spp.estimator_input.npz")
        )
        == "receiver_m8t"
    )
    assert (
        registry.physical_source(
            Path("route__m8t_gej__real_spp.estimator_input.npz")
        )
        == "receiver_m8t"
    )


def test_unknown_stream_is_rejected_instead_of_guessed() -> None:
    registry = ProvenanceRegistry({"known": "receiver"})
    with pytest.raises(ValueError, match="absent from provenance registry"):
        registry.physical_source(
            Path("route__unknown__real_spp.estimator_input.npz")
        )
