"""KISS-conditioned discrete receiver-subset topology audit.

This candidate removes three brittle choices from the earlier implementation:

* it never commits to a receiver solely because it wins by a tiny amount;
* an ambiguous decision produces one conservative subset-consensus factor,
  not multiple independently counted GNSS factors;
* the robust loss family is fixed, so tail statistics cannot silently change
  both receiver identity and the objective.

The evidence window is evaluated as a fixed-lag batch.  Hidden reference truth
is not opened until all candidate and baseline trajectories have been built by
the imported evaluator.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import loewner_topology as loewner
import topology_factor_graph as base
from paper_pipeline.provenance import ProvenanceRegistry


def _candidate_subset(
    window: dict[str, object],
    group_names: list[str],
    motion_margin: float,
    separation_margin: float,
    subset_policy: str = "full",
) -> tuple[list[str], str, dict[str, float]]:
    motion = {
        group: float(
            window["receiver_relative_mismatch"][group]["median"]
        )
        for group in group_names
    }
    ordered_motion = sorted(group_names, key=motion.get)
    best = ordered_motion[0]
    runner_up = ordered_motion[1]
    motion_ratio = motion[runner_up] / max(motion[best], 1e-12)

    pairwise = {
        str(key): float(value["median"])
        for key, value in window["receiver_pair_separation"].items()
    }
    ordered_pairs = sorted(pairwise, key=pairwise.get)
    coherent_pair = ordered_pairs[0].split("__")
    pair_ratio = (
        pairwise[ordered_pairs[1]]
        / max(pairwise[ordered_pairs[0]], 1e-12)
        if len(ordered_pairs) >= 2
        else 1.0
    )
    diagnostics = {
        "motion_margin_ratio": motion_ratio,
        "separation_margin_ratio": pair_ratio,
    }

    if subset_policy == "motion_single":
        return [best], "ablation_motion_winner_single", diagnostics
    if subset_policy == "separation_pair":
        return (
            sorted(coherent_pair),
            "ablation_closest_separation_pair",
            diagnostics,
        )
    if subset_policy != "full":
        raise ValueError(f"unsupported subset policy: {subset_policy}")

    if pair_ratio >= separation_margin:
        return (
            sorted(coherent_pair),
            "separation_triangle_coherent_pair",
            diagnostics,
        )
    if motion_ratio >= motion_margin:
        return (
            [best],
            "kiss_motion_clear_single",
            diagnostics,
        )
    return (
        sorted((best, runner_up)),
        "kiss_motion_ambiguous_pair",
        diagnostics,
    )


def main() -> None:
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument("--motion-margin", type=float, default=2.0)
    custom.add_argument("--separation-margin", type=float, default=2.0)
    custom.add_argument(
        "--subset-policy",
        choices=("full", "motion_single", "separation_pair"),
        default="full",
        help="Frozen full method or one-component ablation.",
    )
    custom.add_argument(
        "--pair-factor-mode",
        choices=("consensus", "independent"),
        default="consensus",
        help=(
            "Construct one conservative factor for a selected pair or "
            "ablate it with two separately counted receiver factors."
        ),
    )
    custom.add_argument(
        "--loewner-floor-m",
        type=float,
        default=1.0,
        help=(
            "Effective eigenvalue standard-deviation floor for every GNSS "
            "factor, including the receiver-subset envelope."
        ),
    )
    custom.add_argument(
        "--provenance-registry",
        type=Path,
        default=Path(__file__).with_name("PROVENANCE_REGISTRY.json"),
    )
    custom_args, remaining = custom.parse_known_args()
    if custom_args.motion_margin <= 1.0:
        raise ValueError("motion margin must exceed one")
    if custom_args.separation_margin <= 1.0:
        raise ValueError("separation margin must exceed one")
    if custom_args.loewner_floor_m <= 0.0:
        raise ValueError("Loewner floor must be positive")

    registry = ProvenanceRegistry.load(custom_args.provenance_registry)
    loewner._center_mode = "arithmetic"
    loewner._envelope_mode = "positive_part"
    loewner._floor_m = custom_args.loewner_floor_m
    base._aggregate = loewner._loewner_aggregate
    base._hardware_group = registry.physical_source
    original_gps_noise = base._gps_noise

    def gps_noise_with_effective_floor(covariance, family):
        return original_gps_noise(
            covariance,
            family,
            floor_m=custom_args.loewner_floor_m,
        )

    # Keep the covariance-floor sensitivity honest: the selected floor must
    # survive the final GTSAM noise construction for both single-receiver and
    # receiver-subset factors.  The frozen nominal value remains 1.0 m.
    base._gps_noise = gps_noise_with_effective_floor
    original_optimize = base._optimize_adjudicated
    candidate_summary: list[dict[str, object]] = []

    def optimize_subset(
        lidar,
        state_times,
        measurement_indices,
        group_positions,
        group_covariances,
        consensus_positions,
        consensus_covariances,
        windows,
    ):
        group_names = sorted(group_positions[0])
        candidate_windows = copy.deepcopy(windows)
        candidate_summary.clear()
        for window in candidate_windows:
            subset, reason, diagnostics = _candidate_subset(
                window,
                group_names,
                custom_args.motion_margin,
                custom_args.separation_margin,
                custom_args.subset_policy,
            )
            window["selected_family"] = "cauchy"
            window["selection_reason"] = reason
            window.update(diagnostics)
            if len(subset) == 1:
                window["selected_receiver"] = subset[0]
                window["selected_receivers"] = subset
                window["factor_topology"] = "single_receiver"
                window["selected_branch"] = f"mms__{reason}"
            else:
                window["selected_receiver"] = None
                window["selected_receivers"] = subset
                window["factor_topology"] = (
                    "receiver_subset_consensus"
                    if custom_args.pair_factor_mode == "consensus"
                    else "receiver_subset_independent"
                )
                window["selected_branch"] = f"mms__{reason}"
            candidate_summary.append(
                {
                    "start_epoch": int(window["start_epoch"]),
                    "stop_epoch": int(window["stop_epoch"]),
                    "selected_receivers": subset,
                    "reason": reason,
                    **diagnostics,
                }
            )
        return original_optimize(
            lidar,
            state_times,
            measurement_indices,
            group_positions,
            group_covariances,
            consensus_positions,
            consensus_covariances,
            candidate_windows,
        )

    base._optimize_adjudicated = optimize_subset
    sys.argv = [sys.argv[0], *remaining]
    base.main()

    output = Path(sys.argv[sys.argv.index("--output") + 1])
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["evidentiary_status"] = "MOTION_MARGIN_SUBSET_CANDIDATE"
    payload["method_metadata"] = {
        "method_name": "motion_margin_subset_topology",
        "subset_policy": custom_args.subset_policy,
        "pair_factor_mode": custom_args.pair_factor_mode,
        "loewner_floor_m": custom_args.loewner_floor_m,
        "effective_gnss_covariance_floor_m": custom_args.loewner_floor_m,
        "motion_margin": custom_args.motion_margin,
        "separation_margin": custom_args.separation_margin,
        "loss_family": "cauchy_fixed",
        "multi_receiver_factor": (
            "single conservative Loewner-envelope subset consensus"
        ),
        "decision_semantics": "same-window fixed-lag batch",
    }
    payload["candidate_subset_windows"] = candidate_summary
    branch_counts: dict[str, int] = {}
    subset_counts: dict[str, int] = {}
    for window in candidate_summary:
        reason = str(window["reason"])
        subset = ",".join(str(item) for item in window["selected_receivers"])
        branch_counts[reason] = branch_counts.get(reason, 0) + 1
        subset_counts[subset] = subset_counts.get(subset, 0) + 1
    payload["candidate_branch_counts"] = branch_counts
    payload["candidate_subset_counts"] = subset_counts
    payload["candidate_source_trajectory"] = "trajectory__dda_fc"
    payload.pop("dda_source_trajectory", None)
    payload.pop("registered_primary_pass_rules", None)
    payload.pop("registered_primary_pass", None)
    payload["metrics"]["motion_margin_subset"] = payload["metrics"].pop(
        "dda_fc"
    )
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
