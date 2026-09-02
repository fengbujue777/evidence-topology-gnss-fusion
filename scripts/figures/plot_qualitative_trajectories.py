"""Render the three representative trajectory panels used in the manuscript.

The script reads the same frozen natural and midpoint-injection fault runs used by
the result tables.  RMSE labels are recomputed from the plotted arrays rather than
copied from manuscript text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TRUTH = "#20262E"
PROPOSED = "#F4B63A"
CAUCHY = "#2A718E"
SWITCH = "#765A9E"
GRID = "#E6EAEF"

METHODS = [
    ("trajectory__dda_fc", "Proposed", PROPOSED),
    (
        "trajectory__grouped_physical_receivers__cauchy",
        "All-receiver Cauchy",
        CAUCHY,
    ),
    (
        "trajectory__grouped_physical_receivers__switch_irls",
        "Switch-IRLS",
        SWITCH,
    ),
]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "axes.edgecolor": "#AEB7C1",
            "axes.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "#FBFCFD",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )


def load_case(directory: Path, stem: str) -> tuple[dict[str, np.ndarray], int]:
    npz_path = directory / f"{stem}.npz"
    json_path = directory / f"{stem}.json"
    with np.load(npz_path) as archive:
        required = ["truth_positions_enu_m", *(key for key, _, _ in METHODS)]
        data = {key: archive[key] for key in required}
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    return data, int(metadata["evaluation_start_index"])


def rmse(estimate: np.ndarray, truth: np.ndarray, start: int) -> float:
    error = np.linalg.norm(estimate[start:] - truth[start:], axis=1)
    return float(np.sqrt(np.mean(error**2)))


def plot_route(
    ax: plt.Axes,
    data: dict[str, np.ndarray],
    start: int,
    title: str,
) -> dict[str, float]:
    truth = data["truth_positions_enu_m"]
    ax.plot(
        truth[start:, 0],
        truth[start:, 1],
        color=TRUTH,
        linewidth=1.65,
        label="Reference",
        zorder=5,
    )
    for key, label, color in METHODS:
        ax.plot(
            data[key][start:, 0],
            data[key][start:, 1],
            color=color,
            linewidth=1.15,
            alpha=0.96,
            label=label,
            zorder=3,
        )
    ax.scatter(
        truth[start, 0],
        truth[start, 1],
        s=22,
        marker="o",
        facecolor="white",
        edgecolor=TRUTH,
        linewidth=1.1,
        zorder=7,
    )
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(color=GRID, linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(0.04)
    return {label: rmse(data[key], truth, start) for key, label, _ in METHODS}


def plot_local_detail(
    ax: plt.Axes,
    data: dict[str, np.ndarray],
    start: int,
    scores: dict[str, float],
) -> None:
    truth = data["truth_positions_enu_m"]
    proposed = data[METHODS[0][0]]
    comparator = data[METHODS[1][0]]
    separation = np.linalg.norm(proposed[start:, :2] - comparator[start:, :2], axis=1)
    center = start + int(np.argmax(separation))
    lo = max(start, center - 25)
    hi = min(len(truth), center + 26)
    points = np.vstack([truth[lo:hi, :2], *(data[key][lo:hi, :2] for key, _, _ in METHODS)])
    span = np.ptp(points, axis=0)
    pad = max(6.0, 0.18 * float(max(span)))

    ax.plot(truth[lo:hi, 0], truth[lo:hi, 1], color=TRUTH, linewidth=1.45, zorder=5)
    for key, _, color in METHODS:
        ax.plot(
            data[key][lo:hi, 0],
            data[key][lo:hi, 1],
            color=color,
            linewidth=1.05,
            zorder=3,
        )
    ax.set_xlim(points[:, 0].min() - pad, points[:, 0].max() + pad)
    ax.set_ylim(points[:, 1].min() - pad, points[:, 1].max() + pad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_facecolor("white")
    ax.grid(color=GRID, linewidth=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#99A4AF")
        spine.set_linewidth(0.75)
    ax.set_title(
        "Local detail\n"
        f"RMSE P/A/S: {scores['Proposed']:.2f} / "
        f"{scores['All-receiver Cauchy']:.2f} / {scores['Switch-IRLS']:.2f} m",
        loc="left",
        fontsize=6.7,
        color="#46515C",
        pad=3.0,
    )


def build_figure(natural_root: Path, fault_root: Path, output_stem: Path) -> None:
    cases = [
        (*load_case(natural_root, "hk"), "(a) HK Medium: natural"),
        (*load_case(natural_root, "harsh"), "(b) HK Harsh: natural"),
        (*load_case(fault_root, "novatel_ramp"), "(c) NovAtel ramp fault"),
    ]

    fig = plt.figure(figsize=(7.25, 3.65))
    grid = fig.add_gridspec(2, 3, height_ratios=[2.0, 0.85], hspace=0.46, wspace=0.32)
    route_axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    detail_axes = [fig.add_subplot(grid[1, index]) for index in range(3)]
    audit: dict[str, dict[str, float]] = {}
    for route_ax, detail_ax, (data, start, title) in zip(route_axes, detail_axes, cases):
        scores = plot_route(route_ax, data, start, title)
        plot_local_detail(detail_ax, data, start, scores)
        audit[title] = scores

    handles, labels = route_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.25,
    )
    fig.subplots_adjust(top=0.94, bottom=0.14, left=0.075, right=0.99)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(audit, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--natural-root",
        type=Path,
        default=ROOT / "outputs" / "repro_test_results" / "natural",
    )
    parser.add_argument(
        "--fault-root",
        type=Path,
        default=ROOT / "outputs" / "revision_proposed_fault_v2",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=ROOT / "outputs" / "paper_figures" / "figure2_qualitative_trajectories",
    )
    args = parser.parse_args()
    configure_style()
    build_figure(args.natural_root, args.fault_root, args.output_stem)


if __name__ == "__main__":
    main()
