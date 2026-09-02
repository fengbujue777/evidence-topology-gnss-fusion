"""Build an honest point-cloud-map comparison from the LOCSP CR2 rosbag.

The fusion layer estimates position only.  Therefore both panels use exactly
the same Hesai scans and KISS-ICP orientations; only the antenna-reference
translation is replaced by the source-quotient or naive-stream trajectory.
This avoids presenting an unrelated or synthetically generated point cloud.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation

from paper_pipeline.pointcloud2 import xyz_and_relative_time


TOPIC = "/hesai/pandar"
PROPOSED = "#F6B73C"
NAIVE = "#38A3C7"
REFERENCE = "#F7FAFC"
BACKGROUND = "#102A33"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--source-aware", required=True, type=Path)
    parser.add_argument("--naive", required=True, type=Path)
    parser.add_argument("--tf-json", required=True, type=Path)
    parser.add_argument("--paper-en", required=True, type=Path)
    parser.add_argument("--paper-cn", required=True, type=Path)
    parser.add_argument("--scan-stride", type=int, default=12)
    parser.add_argument("--points-per-scan", type=int, default=2400)
    parser.add_argument("--evaluation-start-index", type=int, default=181)
    return parser.parse_args()


def transform(translation: list[float], quat_xyzw: list[float]) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    result[:3, 3] = np.asarray(translation, dtype=float)
    return result


def lidar_to_state(path: Path) -> np.ndarray:
    records = json.loads(path.read_text(encoding="utf-8"))
    transforms = {
        str(row["child"]): transform(row["translation"], row["rotation"])
        for row in records
        if row.get("parent") == "base_link"
    }
    return np.linalg.inv(transforms["hesai_frame"]) @ transforms["antenne_frame"]


def interpolate_positions(
    source_times: np.ndarray, positions: np.ndarray, query_times: np.ndarray
) -> np.ndarray:
    return np.column_stack(
        [np.interp(query_times, source_times, positions[:, axis]) for axis in range(3)]
    )


def estimate_yaw_alignment(local_xy: np.ndarray, global_xy: np.ndarray) -> np.ndarray:
    source = local_xy - local_xy.mean(axis=0)
    target = global_xy - global_xy.mean(axis=0)
    u, _, vt = np.linalg.svd(source.T @ target)
    rotation_row = u @ vt
    if np.linalg.det(rotation_row) < 0.0:
        u[:, -1] *= -1.0
        rotation_row = u @ vt
    rotation_column = rotation_row.T
    result = np.eye(3)
    result[:2, :2] = rotation_column
    return result


def read_selected_scans(
    bag: Path, stride: int, points_per_scan: int
) -> tuple[np.ndarray, list[np.ndarray]]:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    times: list[float] = []
    scans: list[np.ndarray] = []
    rng = np.random.default_rng(20260816)
    with AnyReader([bag], default_typestore=typestore) as reader:
        connections = [c for c in reader.connections if c.topic == TOPIC]
        if len(connections) != 1:
            raise RuntimeError(f"Expected one {TOPIC} connection; found {len(connections)}")
        for index, (connection, timestamp_ns, rawdata) in enumerate(
            reader.messages(connections=connections)
        ):
            if index % stride:
                continue
            message = reader.deserialize(rawdata, connection.msgtype)
            xyz, _ = xyz_and_relative_time(message)
            xyz = np.asarray(xyz[:, :3], dtype=np.float64)
            radius = np.linalg.norm(xyz[:, :2], axis=1)
            keep = (
                np.all(np.isfinite(xyz), axis=1)
                & (radius >= 3.0)
                & (radius <= 65.0)
                & (xyz[:, 2] >= -3.0)
                & (xyz[:, 2] <= 7.0)
            )
            xyz = xyz[keep]
            if len(xyz) > points_per_scan:
                selection = rng.choice(len(xyz), points_per_scan, replace=False)
                xyz = xyz[selection]
            times.append(timestamp_ns * 1e-9)
            scans.append(xyz)
    if len(scans) < 100:
        raise RuntimeError(f"Only {len(scans)} scans were retained")
    return np.asarray(times, dtype=float), scans


def build_map(
    scans: list[np.ndarray],
    scan_indices: np.ndarray,
    local_state_poses: np.ndarray,
    rotation_enu_from_local: np.ndarray,
    state_from_lidar: np.ndarray,
    fused_state_positions: np.ndarray,
) -> np.ndarray:
    lidar_from_state = np.linalg.inv(state_from_lidar)
    blocks: list[np.ndarray] = []
    for points, index, state_position in zip(
        scans, scan_indices, fused_state_positions, strict=True
    ):
        local_state = local_state_poses[index]
        local_lidar = local_state @ lidar_from_state
        state_orientation = rotation_enu_from_local @ local_state[:3, :3]
        orientation = rotation_enu_from_local @ local_lidar[:3, :3]
        state_to_lidar_translation = lidar_from_state[:3, 3]
        lidar_position = state_position + state_orientation @ state_to_lidar_translation
        blocks.append(points @ orientation.T + lidar_position)
    return np.vstack(blocks)


def scale_bar(ax: plt.Axes, length: float, label: str) -> None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x0 = xmin + 0.06 * (xmax - xmin)
    y0 = ymin + 0.07 * (ymax - ymin)
    ax.plot([x0, x0 + length], [y0, y0], color="white", linewidth=2.3)
    ax.text(
        x0 + length / 2,
        y0 + 0.018 * (ymax - ymin),
        label,
        color="white",
        fontsize=7.0,
        ha="center",
        va="bottom",
    )


def draw_map(
    ax: plt.Axes,
    points: np.ndarray,
    trajectory: np.ndarray,
    truth: np.ndarray,
    title: str,
    color: str,
    limits: tuple[float, float, float, float] | None,
    scale_length: float,
) -> None:
    if limits is None:
        lo = np.percentile(points[:, :2], 0.3, axis=0)
        hi = np.percentile(points[:, :2], 99.7, axis=0)
        pad = 0.025 * float(max(hi - lo))
        limits = (lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad)
    xmin, xmax, ymin, ymax = limits
    mask = (
        (points[:, 0] >= xmin)
        & (points[:, 0] <= xmax)
        & (points[:, 1] >= ymin)
        & (points[:, 1] <= ymax)
    )
    visible = points[mask]
    height = visible[:, 2]
    low, high = np.percentile(height, [5, 95]) if len(height) else (-2.0, 4.0)
    ax.scatter(
        visible[:, 0],
        visible[:, 1],
        c=np.clip(height, low, high),
        cmap="viridis",
        s=0.28,
        alpha=0.62,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    ax.plot(truth[:, 0], truth[:, 1], color=REFERENCE, linewidth=1.5, zorder=4)
    ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=1.25, zorder=5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor(BACKGROUND)
    ax.set_title(title, loc="left", fontsize=9.0, fontweight="bold", color="#27323D")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#8FA1AC")
        spine.set_linewidth(0.7)
    scale_bar(ax, scale_length, f"{scale_length:g} m")


def main() -> None:
    args = parse_args()
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )
    cache = np.load(args.cache)
    source = np.load(args.source_aware)
    naive = np.load(args.naive)

    cache_times = cache["lidar_timestamps_s"]
    local_state_poses = cache["lidar_poses_local"]
    final_times = source["timestamps_s"]
    final_local_positions = interpolate_positions(
        cache_times, local_state_poses[:, :3, 3], final_times
    )
    rotation_enu_from_local = estimate_yaw_alignment(
        final_local_positions[args.evaluation_start_index :, :2],
        source["trajectory__aligned_kiss_icp"][args.evaluation_start_index :, :2],
    )

    scan_times, scans = read_selected_scans(
        args.bag, args.scan_stride, args.points_per_scan
    )
    scan_indices = np.searchsorted(cache_times, scan_times)
    scan_indices = np.clip(scan_indices, 1, len(cache_times) - 1)
    left = scan_indices - 1
    choose_left = np.abs(cache_times[left] - scan_times) < np.abs(
        cache_times[scan_indices] - scan_times
    )
    scan_indices[choose_left] = left[choose_left]
    time_error = np.abs(cache_times[scan_indices] - scan_times)
    keep = time_error < 0.02
    if int(np.sum(keep)) < 100:
        raise RuntimeError(
            f"Only {int(np.sum(keep))} selected scans match the frozen cache within 20 ms"
        )
    scan_times = scan_times[keep]
    scan_indices = scan_indices[keep]
    scans = [scan for scan, accepted in zip(scans, keep, strict=True) if accepted]

    source_positions = interpolate_positions(
        final_times,
        source["trajectory__grouped_physical_receivers__cauchy"],
        scan_times,
    )
    naive_positions = interpolate_positions(
        final_times,
        naive["trajectory__grouped_physical_receivers__cauchy"],
        scan_times,
    )
    state_from_lidar = lidar_to_state(args.tf_json)
    source_map = build_map(
        scans,
        scan_indices,
        local_state_poses,
        rotation_enu_from_local,
        state_from_lidar,
        source_positions,
    )
    naive_map = build_map(
        scans,
        scan_indices,
        local_state_poses,
        rotation_enu_from_local,
        state_from_lidar,
        naive_positions,
    )

    start = args.evaluation_start_index
    source_traj = source["trajectory__grouped_physical_receivers__cauchy"][start:]
    naive_traj = naive["trajectory__grouped_physical_receivers__cauchy"][start:]
    truth = source["truth_positions_enu_m"][start:]
    difference = np.linalg.norm(source_traj[:, :2] - naive_traj[:, :2], axis=1)
    source_xy_error = np.linalg.norm(source_traj[:, :2] - truth[:, :2], axis=1)
    naive_xy_error = np.linalg.norm(naive_traj[:, :2] - truth[:, :2], axis=1)
    horizontal_gain = naive_xy_error - source_xy_error
    detail_index = int(np.argmax(horizontal_gain))
    center = 0.5 * (
        source_traj[detail_index, :2]
        + naive_traj[detail_index, :2]
    )
    radius = 55.0
    crop = (
        center[0] - radius,
        center[0] + radius,
        center[1] - radius,
        center[1] + radius,
    )

    all_points = np.vstack([source_map, naive_map])
    lo = np.percentile(all_points[:, :2], 0.3, axis=0)
    hi = np.percentile(all_points[:, :2], 99.7, axis=0)
    pad = 0.025 * float(max(hi - lo))
    full = (lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad)

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.0))
    draw_map(
        axes[0, 0],
        source_map,
        source_traj,
        truth,
        "(a) Source quotient: full map (6.05 m)",
        PROPOSED,
        full,
        100.0,
    )
    draw_map(
        axes[0, 1],
        naive_map,
        naive_traj,
        truth,
        "(b) Naive stream count: full map (7.71 m)",
        NAIVE,
        full,
        100.0,
    )
    draw_map(
        axes[1, 0],
        source_map,
        source_traj,
        truth,
        "(c) Source quotient: local detail",
        PROPOSED,
        crop,
        20.0,
    )
    draw_map(
        axes[1, 1],
        naive_map,
        naive_traj,
        truth,
        "(d) Naive stream count: same detail",
        NAIVE,
        crop,
        20.0,
    )
    fig.legend(
        [
            plt.Line2D([0], [0], color=REFERENCE, linewidth=2.2),
            plt.Line2D([0], [0], color=PROPOSED, linewidth=2.2),
            plt.Line2D([0], [0], color=NAIVE, linewidth=2.2),
        ],
        ["NovAtel reference", "Source quotient", "Naive stream count"],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=True,
        facecolor=BACKGROUND,
        edgecolor="#82949F",
        labelcolor="white",
    )
    fig.subplots_adjust(top=0.91, hspace=0.20, wspace=0.10)

    for paper in (args.paper_en, args.paper_cn):
        output = paper / "figures"
        output.mkdir(parents=True, exist_ok=True)
        fig.savefig(output / "figure4_locsp_pointcloud_maps.pdf", dpi=350, bbox_inches="tight")
        fig.savefig(output / "figure4_locsp_pointcloud_maps.png", dpi=450, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "bag": str(args.bag),
        "topic": TOPIC,
        "selected_scans": len(scans),
        "scan_stride": args.scan_stride,
        "points_per_scan_cap": args.points_per_scan,
        "source_map_points": int(len(source_map)),
        "naive_map_points": int(len(naive_map)),
        "maximum_source_naive_xy_separation_m": float(np.max(difference)),
        "detail_selection": "post-evaluation epoch with largest naive-minus-source horizontal error",
        "detail_source_xy_error_m": float(source_xy_error[detail_index]),
        "detail_naive_xy_error_m": float(naive_xy_error[detail_index]),
        "detail_horizontal_error_reduction_m": float(horizontal_gain[detail_index]),
        "map_construction": (
            "Identical Hesai scans and KISS-ICP orientations; method-specific "
            "antenna-reference translations only."
        ),
    }
    for paper in (args.paper_en, args.paper_cn):
        (paper / "supplement" / "pointcloud_map_manifest.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
