"""Compare regenerated headline metrics with the frozen paper artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parent
NATURAL = ("hk", "deep", "harsh", "phone")
FAULTS = (
    "novatel_step",
    "novatel_ramp",
    "ublox_f9p_step",
    "ublox_f9p_ramp",
    "ublox_m8t_step",
    "ublox_m8t_ramp",
    "distributed_novatel_step_f9p_ramp",
)
HEADLINE_METHODS = (
    "motion_margin_subset",
    "grouped_physical_receivers__cauchy",
    "grouped_physical_receivers__switch_irls",
    "grouped_physical_receivers__dcs_irls",
    "hierarchical_consensus__cauchy",
)


def _compare_case(
    scope: str,
    name: str,
    actual_path: Path,
    expected_path: Path,
    proposed_tolerance_m: float,
    baseline_tolerance_m: float,
) -> dict[str, object]:
    actual = json.loads(actual_path.read_text(encoding="utf-8"))
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    rows = []
    for method in HEADLINE_METHODS:
        if method not in expected["metrics"]:
            continue
        actual_value = float(actual["metrics"][method]["rmse_3d_m"])
        expected_value = float(expected["metrics"][method]["rmse_3d_m"])
        difference = abs(actual_value - expected_value)
        tolerance_m = (
            proposed_tolerance_m
            if method == "motion_margin_subset"
            else baseline_tolerance_m
        )
        rows.append(
            {
                "method": method,
                "actual_rmse_3d_m": actual_value,
                "expected_rmse_3d_m": expected_value,
                "absolute_difference_m": difference,
                "tolerance_m": tolerance_m,
                "pass": difference <= tolerance_m,
            }
        )
    counts_match = (
        int(actual["evaluation_epoch_count"])
        == int(expected["evaluation_epoch_count"])
        and int(actual["common_epoch_count"])
        == int(expected["common_epoch_count"])
    )
    return {
        "case": f"{scope}__{name}",
        "counts_match": counts_match,
        "metric_rows": rows,
        "pass": counts_match and all(row["pass"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual-root", type=Path, required=True)
    parser.add_argument(
        "--expected-root",
        type=Path,
        default=REPOSITORY / "results" / "frozen_json",
    )
    parser.add_argument(
        "--proposed-tolerance-m",
        type=float,
        default=0.03,
        help=(
            "Absolute RMSE tolerance for the proposed method. The default "
            "is 3 cm to accommodate cross-platform nonlinear-solver drift."
        ),
    )
    parser.add_argument(
        "--baseline-tolerance-m",
        type=float,
        default=0.15,
        help=(
            "Absolute RMSE tolerance for iterative robust baselines. Their "
            "platform-sensitive stopping paths require a wider 15 cm bound."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reproduction_verification.json"),
    )
    args = parser.parse_args()
    if args.proposed_tolerance_m <= 0.0 or args.baseline_tolerance_m <= 0.0:
        raise ValueError("tolerances must be positive")

    cases = []
    for scope, names, frozen_name in (
        ("natural", NATURAL, "motion_margin_final_natural"),
        ("fault", FAULTS, "motion_margin_final_fault"),
    ):
        for name in names:
            cases.append(
                _compare_case(
                    scope,
                    name,
                    args.actual_root / scope / f"{name}.json",
                    args.expected_root / frozen_name / f"{name}.json",
                    args.proposed_tolerance_m,
                    args.baseline_tolerance_m,
                )
            )
    payload = {
        "proposed_tolerance_m": args.proposed_tolerance_m,
        "baseline_tolerance_m": args.baseline_tolerance_m,
        "tolerance_rationale": (
            "Counts must match exactly. RMSE is compared within explicit "
            "absolute bounds because GTSAM and iterative robust baselines can "
            "take slightly different floating-point stopping paths across "
            "operating systems and CPU libraries."
        ),
        "cases": cases,
        "pass": all(case["pass"] for case in cases),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
