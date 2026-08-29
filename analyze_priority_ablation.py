"""Paired moving-block bootstrap for separation-first versus motion-first.

Both policies are run on the same prepared cases with identical numerical
settings.  The only changed operation is the ordering of the two topology
tests when both receiver-separation and KISS-motion margins are decisive.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from analyze_crossrun_statistics import FAULT, NATURAL, _row
from analyze_paired_statistics import _apply_holm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--motion-first-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--block", type=int, default=30)
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for scope, cases in (("natural", NATURAL), ("fault", FAULT)):
        for case in cases:
            rows.append(
                _row(
                    scope,
                    case,
                    args.full_root / scope,
                    args.motion_first_root / scope,
                    "motion_first",
                    args.block,
                    args.trials,
                    args.seed + len(rows),
                )
            )
    _apply_holm(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "paired separation-first versus motion-first ablation",
        "trials": args.trials,
        "block_epochs": args.block,
        "rows": rows,
    }
    (args.output_root / "report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (args.output_root / "report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
