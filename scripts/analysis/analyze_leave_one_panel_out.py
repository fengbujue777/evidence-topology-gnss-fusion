"""Leave-one-panel-out audit over the already completed sensitivity grid."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PANELS = ("hk", "deep", "harsh", "phone")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    configurations: list[dict[str, object]] = []
    for directory in sorted(args.grid_root.glob("margin*_window*")):
        if not directory.is_dir():
            continue
        panel_rows: dict[str, dict[str, float]] = {}
        for panel in PANELS:
            payload = json.loads(
                (directory / f"{panel}.json").read_text(encoding="utf-8")
            )
            metrics = payload["metrics"]
            panel_rows[panel] = {
                "candidate_rmse_m": float(
                    metrics["motion_margin_subset"]["rmse_3d_m"]
                ),
                "all_cauchy_rmse_m": float(
                    metrics["grouped_physical_receivers__cauchy"][
                        "rmse_3d_m"
                    ]
                ),
            }
        configurations.append(
            {"configuration": directory.name, "panels": panel_rows}
        )

    rows: list[dict[str, object]] = []
    for held_out in PANELS:
        training = [panel for panel in PANELS if panel != held_out]

        def training_score(configuration: dict[str, object]) -> float:
            panels = configuration["panels"]
            return sum(
                panels[panel]["candidate_rmse_m"]
                / panels[panel]["all_cauchy_rmse_m"]
                for panel in training
            ) / len(training)

        selected = min(
            configurations,
            key=lambda configuration: (
                training_score(configuration),
                str(configuration["configuration"]),
            ),
        )
        held = selected["panels"][held_out]
        rows.append(
            {
                "held_out_panel": held_out,
                "selected_configuration": selected["configuration"],
                "training_mean_candidate_over_all_cauchy": training_score(
                    selected
                ),
                "held_out_candidate_rmse_m": held["candidate_rmse_m"],
                "held_out_all_cauchy_rmse_m": held["all_cauchy_rmse_m"],
                "held_out_relative_improvement_percent": 100.0
                * (
                    held["all_cauchy_rmse_m"]
                    - held["candidate_rmse_m"]
                )
                / held["all_cauchy_rmse_m"],
            }
        )

    result = {
        "design": "leave-one-panel-out selection on normalized RMSE",
        "candidate_configurations": len(configurations),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with args.output.with_suffix(".csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
