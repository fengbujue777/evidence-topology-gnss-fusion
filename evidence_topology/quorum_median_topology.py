"""Quorum-median receiver adjudication layered on the frozen TA-FC panel.

This candidate exists because the sealed Harsh holdout falsified the original
tail-priority receiver-identity decision.  Harsh is therefore development data
for this candidate.  The original frozen source is imported without editing so
that its failed blind result and SHA-256 remain reproducible.

The change is deliberately narrow:

* with three or more physical receivers, receiver identity is chosen using the
  median KISS-versus-receiver relative-motion mismatch;
* with only two physical receivers, the original p95 decision is retained
  because no third-receiver quorum exists;
* tail evidence still selects the robust loss family, but cannot by itself
  choose receiver identity.
"""

from __future__ import annotations

from evidence_topology import topology_factor_graph as base


def _quorum_median_initial_adjudication(
    prefix_count: int,
    group_names: list[str],
    group_positions: list[dict[str, object]],
    consensus_positions: object,
    lidar_common_positions: object,
    tail_adjudication_enabled: bool,
) -> dict[str, object]:
    nominal = base._window_decision(
        0,
        prefix_count,
        group_names,
        group_positions,
        consensus_positions,
        lidar_common_positions,
    )
    heavy_tailed = tail_adjudication_enabled and (
        float(
            nominal["consensus_relative_mismatch"][
                "contamination_fraction"
            ]
        )
        > 0.10
    )
    heterogeneous = (
        float(nominal["receiver_motion_p95_dispersion_ratio"]) > 2.0
    )
    statistic = "median" if len(group_names) >= 3 else "p95"
    selected_receiver = (
        min(
            group_names,
            key=lambda group: float(
                nominal["receiver_relative_mismatch"][group][statistic]
            ),
        )
        if heavy_tailed or heterogeneous
        else None
    )
    if heavy_tailed:
        selected_branch = "quorum_median_receiver_cauchy"
        selected_family = "cauchy"
    elif heterogeneous:
        selected_branch = "quorum_median_receiver_gaussian"
        selected_family = "direct"
    else:
        selected_branch = "receiver_resolved_gaussian"
        selected_family = "direct"
    return {
        "prefix_epoch_count": prefix_count,
        "selected_branch": selected_branch,
        "selected_receiver": selected_receiver,
        "selected_family": selected_family,
        "receiver_identity_statistic": statistic,
        "consensus_relative_mismatch": nominal[
            "consensus_relative_mismatch"
        ],
        "receiver_relative_mismatch": nominal[
            "receiver_relative_mismatch"
        ],
        "receiver_absolute_residual": nominal[
            "receiver_absolute_residual"
        ],
        "receiver_motion_p95_dispersion_ratio": nominal[
            "receiver_motion_p95_dispersion_ratio"
        ],
        "receiver_absolute_median_dispersion_ratio": nominal[
            "receiver_absolute_median_dispersion_ratio"
        ],
        "maximum_pairwise_receiver_separation": nominal[
            "maximum_pairwise_receiver_separation"
        ],
        "receiver_pair_separation": nominal["receiver_pair_separation"],
    }


if __name__ == "__main__":
    base._initial_adjudication = _quorum_median_initial_adjudication
    base.main()
