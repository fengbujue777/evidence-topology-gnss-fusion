"""Dataset-free sanity check for the released evidence-topology components."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from paper_pipeline.loewner_envelope import provenance_loewner_envelope
from paper_pipeline.provenance import ProvenanceRegistry
from run_evidence_topology import _candidate_subset


def main() -> None:
    registry = ProvenanceRegistry.load(REPOSITORY / "PROVENANCE_REGISTRY.json")
    m8t_fix = registry.physical_source(
        Path("demo__locsp_m8t__natural.estimator_input.npz")
    )
    m8t_navpvt = registry.physical_source(
        Path("demo__locsp_m8t_navpvt__natural.estimator_input.npz")
    )
    assert m8t_fix == m8t_navpvt == "locsp_m8t"

    window = {
        "receiver_relative_mismatch": {
            "novatel": {"median": 1.0},
            "ublox_f9p": {"median": 1.1},
            "ublox_m8t": {"median": 7.5},
        },
        "receiver_pair_separation": {
            "novatel__ublox_f9p": {"median": 0.8},
            "novatel__ublox_m8t": {"median": 6.4},
            "ublox_f9p__ublox_m8t": {"median": 6.1},
        },
    }
    subset, reason, diagnostics = _candidate_subset(
        window,
        ["novatel", "ublox_f9p", "ublox_m8t"],
        motion_margin=2.0,
        separation_margin=2.0,
    )
    assert subset == ["novatel", "ublox_f9p"]
    assert reason == "separation_triangle_coherent_pair"

    points = np.array([[0.0, 0.0, 0.0], [0.4, -0.2, 0.1]])
    covariances = np.array(
        [np.diag([1.0, 2.0, 1.5]), np.diag([2.0, 1.0, 1.2])]
    )
    center, envelope, loading_trace = provenance_loewner_envelope(
        points,
        covariances,
        floor_m=1.0,
        center_mode="arithmetic",
    )
    for covariance in covariances:
        assert np.min(np.linalg.eigvalsh(envelope - covariance)) >= -1e-10

    print("PASS: physical-source quotient maps both M8T streams to one source")
    print(f"PASS: selected subset={subset}, reason={reason}")
    print(f"PASS: center={center.tolist()}, loading_trace={loading_trace:.6f}")
    print(f"diagnostics={diagnostics}")


if __name__ == "__main__":
    main()
