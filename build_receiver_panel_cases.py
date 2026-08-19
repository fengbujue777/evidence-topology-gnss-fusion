"""Build truth-isolated real-SPP cases for the frozen receiver panel."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from paper_pipeline.cases import build_real_gnss_case


def portable_path(value: str) -> Path:
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", value)
    if os.name != "nt" and match:
        return Path(
            f"/mnt/{match.group(1).lower()}/"
            f"{match.group(2).replace(chr(92), '/')}"
        )
    return Path(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--sensor-cache", type=Path, required=True)
    parser.add_argument("--truth-cache", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--case-index", type=Path, required=True)
    parser.add_argument("--use-full-gnss-covariance", action="store_true")
    parser.add_argument("--minimum-gnss-interval-s", type=float, default=0.0)
    args = parser.parse_args()

    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    streams = [panel["primary_stream"], *panel["secondary_streams"]]
    manifests = []
    for stream in streams:
        case_dataset_id = f"{panel['dataset_id']}__{stream['id']}"
        manifest = build_real_gnss_case(
            case_dataset_id,
            args.sensor_cache,
            args.truth_cache,
            portable_path(stream["solution"]),
            args.output_directory,
            use_full_gnss_covariance=args.use_full_gnss_covariance,
            minimum_gnss_interval_s=args.minimum_gnss_interval_s,
        )
        manifest["receiver_id"] = stream["id"]
        manifest["receiver_role"] = stream["role"]
        manifests.append(manifest)

    args.case_index.parent.mkdir(parents=True, exist_ok=True)
    args.case_index.write_text(
        json.dumps(manifests, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifests, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
