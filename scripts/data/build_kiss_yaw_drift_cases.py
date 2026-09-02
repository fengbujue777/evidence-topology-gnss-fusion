"""Copy a receiver panel and inject accumulated yaw error into KISS odometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rate-degps", type=float, required=True)
    parser.add_argument("--start-fraction", type=float, default=0.5)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for estimator_path in sorted(
        args.source_root.glob("*.estimator_input.npz")
    ):
        hidden_path = estimator_path.with_name(
            estimator_path.name.replace(
                ".estimator_input.npz", ".hidden_labels.npz"
            )
        )
        with np.load(estimator_path, allow_pickle=False) as source:
            arrays = {
                name: np.asarray(source[name]) for name in source.files
            }
        times = np.asarray(arrays["lidar_timestamps_s"], dtype=float)
        original = np.asarray(arrays["lidar_poses_local"], dtype=float)
        poses = original.copy()
        onset = times[0] + args.start_fraction * (times[-1] - times[0])
        elapsed = np.maximum(times - onset, 0.0)
        angles = np.deg2rad(args.rate_degps * elapsed)
        start = int(np.searchsorted(times, onset))
        for index in range(max(start, 1), len(poses)):
            delta = original[index, :3, 3] - original[index - 1, :3, 3]
            rotation = Rotation.from_euler("z", angles[index]).as_matrix()
            poses[index, :3, 3] = poses[index - 1, :3, 3] + rotation @ delta
            poses[index, :3, :3] = rotation @ original[index, :3, :3]
        arrays["lidar_poses_local"] = poses
        np.savez_compressed(args.output_root / estimator_path.name, **arrays)
        shutil.copy2(hidden_path, args.output_root / hidden_path.name)
        records.append(
            {
                "input": estimator_path.name,
                "onset_s": float(onset),
                "maximum_yaw_error_deg": float(np.max(args.rate_degps * elapsed)),
            }
        )
    (args.output_root / "manifest.json").write_text(
        json.dumps(
            {
                "source_root": str(args.source_root),
                "truth_modified": False,
                "gnss_modified": False,
                "rate_degps": args.rate_degps,
                "start_fraction": args.start_fraction,
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
