"""QM-TAFC with a provable within-provenance Loewner covariance envelope."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from evidence_topology import quorum_median_topology as quorum
from evidence_topology import topology_factor_graph as base

from paper_pipeline.loewner_envelope import provenance_loewner_envelope
from paper_pipeline.provenance import ProvenanceRegistry


_original_aggregate = base._aggregate
_diagnostics: list[dict[str, object]] = []
_center_mode = "arithmetic"
_envelope_mode = "positive_part"
_floor_m = 1.0


def _loewner_aggregate(points, covariances, center_mode, covariance_mode):
    if center_mode == "geometric" and covariance_mode == "nonshrinking":
        center, covariance, loading_trace = provenance_loewner_envelope(
            points,
            covariances,
            floor_m=_floor_m,
            center_mode=_center_mode,
            envelope_mode=_envelope_mode,
        )
        _diagnostics.append(
            {
                "stream_count": int(len(points)),
                "loading_trace_m2": float(loading_trace),
            }
        )
        return center, covariance
    return _original_aggregate(
        points, covariances, center_mode, covariance_mode
    )


if __name__ == "__main__":
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument(
        "--loewner-center",
        choices=("arithmetic", "geometric"),
        default="arithmetic",
    )
    custom.add_argument(
        "--loewner-envelope",
        choices=("positive_part", "isotropic"),
        default="positive_part",
    )
    custom.add_argument(
        "--provenance-registry",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "PROVENANCE_REGISTRY.json",
    )
    custom_args, remaining = custom.parse_known_args()
    _center_mode = custom_args.loewner_center
    _envelope_mode = custom_args.loewner_envelope
    provenance_registry = ProvenanceRegistry.load(
        custom_args.provenance_registry
    )
    sys.argv = [sys.argv[0], *remaining]
    base._aggregate = _loewner_aggregate
    base._hardware_group = provenance_registry.physical_source
    base._initial_adjudication = quorum._quorum_median_initial_adjudication
    base.main()
    output_index = sys.argv.index("--output") + 1
    output = Path(sys.argv[output_index])
    result_payload = json.loads(output.read_text(encoding="utf-8"))
    registry_bytes = custom_args.provenance_registry.read_bytes()
    result_payload["method_metadata"] = {
        "method_name": "provenance_loewner_quorum_guarded_factor",
        "center_mode": _center_mode,
        "envelope_mode": _envelope_mode,
        "provenance_registry": str(custom_args.provenance_registry),
        "provenance_registry_sha256": hashlib.sha256(
            registry_bytes
        ).hexdigest(),
        "physical_source_identity_is_explicit_metadata": True,
    }
    output.write_text(
        json.dumps(result_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    loadings = [float(item["loading_trace_m2"]) for item in _diagnostics]
    diagnostic_output = output.with_name(
        f"{output.stem}.loewner_diagnostics.json"
    )
    diagnostic_output.write_text(
        json.dumps(
            {
                "center_mode": _center_mode,
                "envelope_mode": _envelope_mode,
                "aggregate_call_count": len(_diagnostics),
                "nonzero_loading_count": int(
                    sum(value > 1e-12 for value in loadings)
                ),
                "loading_trace_m2": {
                    "median": (
                        float(np.median(loadings)) if loadings else 0.0
                    ),
                    "p95": (
                        float(np.percentile(loadings, 95.0))
                        if loadings
                        else 0.0
                    ),
                    "maximum": max(loadings, default=0.0),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
