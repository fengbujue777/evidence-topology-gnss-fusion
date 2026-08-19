"""Prepare one UrbanNav receiver panel from public raw files.

The command optionally extracts KISS-ICP and truth caches from a ROS bag,
runs the frozen RTKLIB SPP configuration for every registered stream, applies
the documented phone clock normalization when requested, and writes the
truth-isolated estimator/label case pair consumed by the paper experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parent


def _run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPOSITORY, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--rinex-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--solution-root", type=Path, required=True)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--bag", nargs="+", type=Path)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--extrinsics-json", type=Path)
    parser.add_argument("--deskew", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    panel_path = args.panel.resolve()
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    dataset_id = str(panel.get("sensor_dataset_id", panel["dataset_id"]))
    sensor_cache = args.cache_root / f"{dataset_id}.estimator_sensor_cache.npz"
    truth_cache = args.cache_root / f"{dataset_id}.evaluation_truth_cache.npz"

    if args.bag:
        if args.extrinsics_json is None:
            raise ValueError("--extrinsics-json is required with --bag")
        if args.overwrite or not (sensor_cache.exists() and truth_cache.exists()):
            command = [
                sys.executable,
                str(REPOSITORY / "extract_urbannav_cache.py"),
                "--dataset-id", dataset_id,
                "--bag", *[str(path) for path in args.bag],
                "--output-root", str(args.cache_root),
                "--extrinsics-json", str(args.extrinsics_json),
            ]
            if args.truth_csv is not None:
                command.extend(("--truth-csv", str(args.truth_csv)))
            if args.deskew:
                command.append("--deskew")
            _run(command)
    if not sensor_cache.is_file() or not truth_cache.is_file():
        raise FileNotFoundError(
            "missing KISS/truth cache; provide --bag or point --cache-root "
            "to a completed extraction"
        )

    raw_solutions = (
        args.solution_root / "raw"
        if panel.get("requires_timestamp_normalization", False)
        else args.solution_root
    )
    spp_manifest = raw_solutions / "spp_manifest.json"
    if args.overwrite or not spp_manifest.exists():
        command = [
            sys.executable,
            str(REPOSITORY / "run_receiver_spp.py"),
            "--panel", str(panel_path),
            "--rinex-root", str(args.rinex_root),
            "--output-root", str(raw_solutions),
        ]
        if args.overwrite:
            command.append("--overwrite")
        _run(command)

    case_solutions = raw_solutions
    if panel.get("requires_timestamp_normalization", False):
        case_solutions = args.solution_root / "normalized"
        normalization = case_solutions / "timestamp_normalization_manifest.json"
        if args.overwrite or not normalization.exists():
            command = [
                sys.executable,
                str(REPOSITORY / "normalize_phone_solution_timestamps.py"),
                "--source-root", str(raw_solutions),
                "--output-root", str(case_solutions),
            ]
            if args.overwrite:
                command.append("--overwrite")
            _run(command)

    args.case_root.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            str(REPOSITORY / "build_receiver_panel_cases.py"),
            "--panel", str(panel_path),
            "--sensor-cache", str(sensor_cache),
            "--truth-cache", str(truth_cache),
            "--solution-root", str(case_solutions),
            "--output-directory", str(args.case_root),
            "--case-index", str(args.case_root.parent / "case_index.json"),
        ]
    )


if __name__ == "__main__":
    main()
