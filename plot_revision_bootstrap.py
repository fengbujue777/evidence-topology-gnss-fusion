"""Render the revised paired-bootstrap figure from an archived report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ORDER = (
    "natural__hk",
    "natural__deep",
    "natural__harsh",
    "natural__phone",
    "fault__novatel_step",
    "fault__novatel_ramp",
    "fault__ublox_f9p_step",
    "fault__ublox_f9p_ramp",
    "fault__ublox_m8t_step",
    "fault__ublox_m8t_ramp",
    "fault__distributed_novatel_step_f9p_ramp",
)
LABELS = (
    "Medium",
    "Deep",
    "Harsh",
    "Phones",
    "NovAtel step",
    "NovAtel ramp",
    "F9P step",
    "F9P ramp",
    "M8T step",
    "M8T ramp",
    "Distributed double fault",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.report.read_text(encoding="utf-8"))
    rows = {
        row["case"]: row
        for row in payload["rows"]
        if row["baseline"] == "grouped_physical_receivers__cauchy"
    }
    selected = [rows[name] for name in ORDER]
    centers = np.asarray(
        [row["candidate_minus_baseline_rmse_m"] for row in selected]
    )
    low = np.asarray([row["ci95_low_m"] for row in selected])
    high = np.asarray([row["ci95_high_m"] for row in selected])
    significant = np.asarray(
        [row["holm_significant_0p05"] for row in selected]
    )
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
        }
    )
    gold, gray, dark, blue = "#F5B83D", "#8A96A3", "#26313D", "#2D718E"
    y = np.arange(len(LABELS))[::-1]
    fig, ax = plt.subplots(figsize=(6.8, 3.45))
    for yi, center, lower, upper, is_significant in zip(
        y, centers, low, high, significant
    ):
        color = gold if is_significant else gray
        ax.errorbar(
            center,
            yi,
            xerr=[[center - lower], [upper - center]],
            fmt="o",
            markersize=4.2,
            color=color,
            ecolor=color,
            elinewidth=1.7,
            capsize=2.8,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )
    ax.axvline(0.0, color=dark, linewidth=0.9, linestyle="--")
    ax.axhline(6.5, color="#D9DEE4", linewidth=0.8)
    ax.set_yticks(y, LABELS)
    ax.set_xlabel(
        r"RMSE difference: $\Delta$RMSE = RMSE(proposed) "
        r"$-$ RMSE(all-receiver Cauchy) (m)"
    )
    ax.text(
        -0.08,
        6.62,
        r"Proposed better $\leftarrow$",
        ha="right",
        va="bottom",
        color=gold,
        fontsize=7.2,
        fontweight="bold",
    )
    ax.text(
        0.08,
        6.62,
        r"$\rightarrow$ Baseline better",
        ha="left",
        va="bottom",
        color=blue,
        fontsize=7.2,
        fontweight="bold",
    )
    ax.set_title(
        "Paired moving-block bootstrap (10,000 resamples, 30-epoch blocks)",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="x", color="#E8EBEF", linewidth=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for suffix, options in (
        ("pdf", {}),
        ("png", {"dpi": 400}),
    ):
        fig.savefig(
            args.output_root / f"figure3_bootstrap_intervals.{suffix}",
            bbox_inches="tight",
            **options,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
