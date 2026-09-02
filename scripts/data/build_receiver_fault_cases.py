"""Create controlled receiver-fault copies without exposing truth to the estimator.

The generated cases preserve timestamps, LiDAR odometry, covariances, and hidden
ground truth.  Only the selected physical receiver's SPP positions are changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from evidence_topology.receiver_consensus import _hardware_group, _nearest_indices


def _inject(
    positions: np.ndarray,
    timestamps_s: np.ndarray,
    mode: str,
    start_fraction: float,
    seed: int,
    step_offset_m: np.ndarray,
    ramp_end_offset_m: np.ndarray,
    start_timestamp_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    result = np.asarray(positions, dtype=float).copy()
    count = len(result)
    if start_timestamp_s is None:
        start = int(np.floor(start_fraction * count))
    else:
        start = int(np.searchsorted(timestamps_s, start_timestamp_s, side="left"))
    start = min(max(start, 1), count - 1)
    affected = np.arange(count) >= start
    if mode == "step":
        result[affected] += np.asarray(step_offset_m, dtype=float)
    elif mode == "ramp":
        progress = np.zeros(count)
        denominator = max(count - start - 1, 1)
        progress[affected] = np.arange(np.count_nonzero(affected)) / denominator
        result += progress[:, None] * np.asarray(
            ramp_end_offset_m, dtype=float
        )
    elif mode == "sparse_outlier":
        rng = np.random.default_rng(seed)
        candidates = np.flatnonzero(affected)
        selected_count = max(1, int(np.ceil(0.05 * len(candidates))))
        selected = np.sort(
            rng.choice(candidates, selected_count, replace=False)
        )
        headings = rng.uniform(0.0, 2.0 * np.pi, len(selected))
        result[selected, 0] += 100.0 * np.cos(headings)
        result[selected, 1] += 100.0 * np.sin(headings)
        affected = np.zeros(count, dtype=bool)
        affected[selected] = True
    else:
        raise ValueError(f"unsupported fault mode: {mode}")
    return result, affected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receiver-token", required=True)
    parser.add_argument(
        "--mode",
        choices=("step", "ramp", "sparse_outlier"),
        required=True,
    )
    parser.add_argument("--start-fraction", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--step-offset-m",
        type=float,
        nargs=3,
        default=(20.0, -10.0, 0.0),
    )
    parser.add_argument(
        "--ramp-end-offset-m",
        type=float,
        nargs=3,
        default=(50.0, 25.0, 0.0),
    )
    args = parser.parse_args()

    if not 0.0 < args.start_fraction < 1.0:
        raise ValueError("start-fraction must lie strictly between zero and one")
    args.output_root.mkdir(parents=True, exist_ok=True)
    sources = []
    for estimator_path in sorted(
        args.source_root.glob("*.estimator_input.npz")
    ):
        with np.load(estimator_path, allow_pickle=False) as source:
            timestamps = np.asarray(source["gnss_timestamps_s"], dtype=float)
            lidar_timestamps = np.asarray(
                source["lidar_timestamps_s"], dtype=float
            )
        sources.append(
            {
                "path": estimator_path,
                "group": _hardware_group(estimator_path),
                "gnss_timestamps_s": timestamps,
                "lidar_start_s": float(lidar_timestamps[0]),
                "lidar_end_s": float(lidar_timestamps[-1]),
            }
        )
    if not sources:
        raise ValueError(f"no estimator inputs in {args.source_root}")
    common_start_s = max(float(source["lidar_start_s"]) for source in sources)
    common_end_s = min(float(source["lidar_end_s"]) for source in sources)
    if common_end_s <= common_start_s:
        raise ValueError("receiver streams have no common LiDAR time interval")
    groups = sorted({str(source["group"]) for source in sources})
    all_times = np.unique(
        np.concatenate(
            [
                np.asarray(source["gnss_timestamps_s"], dtype=float)
                for source in sources
            ]
        )
    )
    all_times = all_times[
        (all_times >= common_start_s) & (all_times <= common_end_s)
    ]
    common_epochs = []
    for timestamp in all_times:
        complete = True
        for group in groups:
            group_complete = False
            for source in sources:
                if source["group"] != group:
                    continue
                _, difference = _nearest_indices(
                    np.asarray(source["gnss_timestamps_s"], dtype=float),
                    np.asarray([timestamp]),
                )
                if float(difference[0]) <= 0.2:
                    group_complete = True
                    break
            if not group_complete:
                complete = False
                break
        if complete:
            common_epochs.append(float(timestamp))
    if len(common_epochs) < 2:
        raise ValueError("fewer than two synchronized physical-source epochs")
    start_epoch = min(
        max(int(np.floor(args.start_fraction * len(common_epochs))), 1),
        len(common_epochs) - 1,
    )
    start_timestamp_s = common_epochs[start_epoch]
    records = []
    for estimator_path in sorted(
        args.source_root.glob("*.estimator_input.npz")
    ):
        hidden_path = estimator_path.with_name(
            estimator_path.name.replace(
                ".estimator_input.npz", ".hidden_labels.npz"
            )
        )
        output_estimator = args.output_root / estimator_path.name
        output_hidden = args.output_root / hidden_path.name
        injected = args.receiver_token.lower() in estimator_path.name.lower()
        if injected:
            with np.load(estimator_path, allow_pickle=False) as source:
                arrays = {
                    name: np.asarray(source[name]) for name in source.files
                }
            changed, affected = _inject(
                arrays["gnss_positions_enu_m"],
                arrays["gnss_timestamps_s"],
                args.mode,
                args.start_fraction,
                args.seed,
                np.asarray(args.step_offset_m, dtype=float),
                np.asarray(args.ramp_end_offset_m, dtype=float),
                start_timestamp_s=start_timestamp_s,
            )
            arrays["gnss_positions_enu_m"] = changed
            np.savez_compressed(output_estimator, **arrays)
            affected_count = int(np.count_nonzero(affected))
        else:
            shutil.copy2(estimator_path, output_estimator)
            affected_count = 0
        shutil.copy2(hidden_path, output_hidden)
        records.append(
            {
                "input": estimator_path.name,
                "receiver_selected": injected,
                "affected_epoch_count": affected_count,
            }
        )
    if not any(record["receiver_selected"] for record in records):
        raise ValueError(
            f"receiver token {args.receiver_token!r} matched no input file"
        )
    manifest = {
        "source_root": str(args.source_root),
        "receiver_token": args.receiver_token,
        "mode": args.mode,
        "start_fraction": args.start_fraction,
        "common_time_interval_s": [common_start_s, common_end_s],
        "common_source_epoch_count": len(common_epochs),
        "start_common_source_epoch": start_epoch,
        "start_timestamp_s": start_timestamp_s,
        "step_offset_m": list(args.step_offset_m),
        "ramp_end_offset_m": list(args.ramp_end_offset_m),
        "seed": args.seed,
        "truth_modified": False,
        "records": records,
    }
    (args.output_root / "fault_injection_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
