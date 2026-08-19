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


def resolve_solution(
    value: str,
    panel_path: Path,
    solution_root: Path | None,
) -> Path:
    """Resolve a receiver solution without binding the panel to one machine."""

    candidate = portable_path(value)
    if candidate.is_absolute():
        return candidate
    base = solution_root if solution_root is not None else panel_path.parent
    return (base / candidate).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--sensor-cache", type=Path, required=True)
    parser.add_argument("--truth-cache", type=Path, required=True)
    parser.add_argument(
        "--solution-root",
        type=Path,
        help=(
            "Base directory for relative solution paths in the panel. "
            "Defaults to the panel file's directory."
        ),
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--case-index", type=Path, required=True)
    parser.add_argument("--use-full-gnss-covariance", action="store_true")
    parser.add_argument("--minimum-gnss-interval-s", type=float, default=0.0)
    args = parser.parse_args()

    panel_path = args.panel.resolve()
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    if "streams" in panel:
        streams = [
            stream
            for stream in panel["streams"]
            if stream.get("include_in_panel", True)
        ]
    else:
        streams = [panel["primary_stream"], *panel["secondary_streams"]]
    if not streams:
        raise ValueError("receiver panel must contain at least one stream")
    manifests = []
    for stream in streams:
        if not {"id", "solution"} <= set(stream):
            raise ValueError("every receiver stream needs id and solution")
        case_dataset_id = f"{panel['dataset_id']}__{stream['id']}"
        manifest = build_real_gnss_case(
            case_dataset_id,
            args.sensor_cache,
            args.truth_cache,
            resolve_solution(
                str(stream["solution"]), panel_path, args.solution_root
            ),
            args.output_directory,
            use_full_gnss_covariance=args.use_full_gnss_covariance,
            minimum_gnss_interval_s=args.minimum_gnss_interval_s,
        )
        manifest["receiver_id"] = stream["id"]
        manifest["receiver_role"] = stream.get("role", "physical_receiver")
        manifests.append(manifest)

    args.case_index.parent.mkdir(parents=True, exist_ok=True)
    args.case_index.write_text(
        json.dumps(manifests, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifests, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
