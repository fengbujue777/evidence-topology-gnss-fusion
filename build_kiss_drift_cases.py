"""Copy a receiver panel and inject deterministic KISS translation drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rate-x-mps", type=float, required=True)
    parser.add_argument("--start-fraction", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 < args.start_fraction < 1.0:
        raise ValueError("start fraction must lie in (0, 1)")

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
        poses = np.asarray(arrays["lidar_poses_local"], dtype=float).copy()
        onset = times[0] + args.start_fraction * (times[-1] - times[0])
        elapsed = np.maximum(times - onset, 0.0)
        poses[:, 0, 3] += args.rate_x_mps * elapsed
        poses[:, 1, 3] += 0.5 * args.rate_x_mps * elapsed
        arrays["lidar_poses_local"] = poses
        output_estimator = args.output_root / estimator_path.name
        np.savez_compressed(output_estimator, **arrays)
        shutil.copy2(hidden_path, args.output_root / hidden_path.name)
        records.append(
            {
                "input": estimator_path.name,
                "onset_s": float(onset),
                "maximum_injected_translation_m": float(
                    np.max(
                        np.linalg.norm(
                            np.column_stack(
                                [
                                    args.rate_x_mps * elapsed,
                                    0.5 * args.rate_x_mps * elapsed,
                                ]
                            ),
                            axis=1,
                        )
                    )
                ),
            }
        )
    manifest = {
        "source_root": str(args.source_root),
        "truth_modified": False,
        "gnss_modified": False,
        "rate_x_mps": args.rate_x_mps,
        "rate_y_mps": 0.5 * args.rate_x_mps,
        "start_fraction": args.start_fraction,
        "records": records,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
