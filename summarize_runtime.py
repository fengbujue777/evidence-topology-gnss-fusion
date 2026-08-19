"""Summarize five frozen fusion-layer runtime repeats."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np
import psutil


def _mean_sd(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.repeat_root.glob("repeat_*.json"))
    if len(paths) != 5:
        raise ValueError(f"expected five repeats, found {len(paths)}")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]

    factor = np.asarray(
        [p["proposed_runtime_breakdown_s"]["factor_construction"] for p in payloads]
    )
    alignment = np.asarray(
        [p["proposed_runtime_breakdown_s"]["truth_free_alignment"] for p in payloads]
    )
    topology = np.asarray(
        [p["proposed_runtime_breakdown_s"]["topology_decision"] for p in payloads]
    )
    shared = factor + alignment
    graph_keys = {
        "all_receiver_cauchy": "grouped_physical_receivers__cauchy",
        "switch_irls": "grouped_physical_receivers__switch_irls",
        "dcs_irls": "grouped_physical_receivers__dcs_irls",
    }
    totals = {
        name: shared
        + np.asarray([p["metrics"][key]["runtime_s"] for p in payloads])
        for name, key in graph_keys.items()
    }
    totals["proposed"] = np.asarray(
        [
            p["proposed_runtime_breakdown_s"]["sum_excluding_file_io"]
            for p in payloads
        ]
    )
    rows = [
        {"method": name, **_mean_sd(values)}
        for name, values in totals.items()
    ]
    common_epochs = int(payloads[0]["common_epoch_count"])
    result = {
        "repeats": 5,
        "state_epochs": int(payloads[0]["state_epoch_count"]),
        "common_gnss_epochs": common_epochs,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_processors": psutil.cpu_count(logical=True),
            "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
            "gpu_used": False,
        },
        "method_total_excluding_file_io_s": {
            row["method"]: {
                "mean": row["mean"],
                "sample_sd": row["sample_sd"],
            }
            for row in rows
        },
        "topology_decision_s": {
            **_mean_sd(topology),
            "mean_ms_per_common_gnss_epoch": float(
                1000.0 * topology.mean() / common_epochs
            ),
        },
        "timing_scope": (
            "CPU process time for fusion; file I/O and shared KISS-ICP "
            "registration are excluded."
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "runtime_benchmark.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (args.output_root / "runtime_benchmark.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
