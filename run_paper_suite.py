"""Portable multi-panel launcher for the frozen proposed estimator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parent


def _resolve(path: str, base: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    panels = payload.get("panels", {})
    if not panels:
        raise ValueError("config must define at least one panel")

    output_root = _resolve(
        str(payload.get("output_root", "outputs/paper_panels")), REPOSITORY
    )
    output_root.mkdir(parents=True, exist_ok=True)

    for name, configured_path in panels.items():
        case_root = _resolve(str(configured_path), REPOSITORY)
        if not case_root.is_dir():
            raise FileNotFoundError(f"missing case directory for {name}: {case_root}")
        json_output = output_root / f"{name}.json"
        trajectory_output = output_root / f"{name}.npz"
        if json_output.exists() and not args.overwrite:
            print(f"SKIP {name}: {json_output} already exists")
            continue
        command = [
            sys.executable,
            str(REPOSITORY / "run_evidence_topology.py"),
            "--motion-margin", "2.0",
            "--separation-margin", "2.0",
            "--window-epochs", "30",
            "--epoch-policy", "all_sources",
            "--alignment-prefix-epochs", "120",
            "--initial-prefix-epochs", "60",
            "--case-root", str(case_root),
            "--output", str(json_output),
            "--trajectory-output", str(trajectory_output),
        ]
        print(f"RUN {name}: {case_root}")
        subprocess.run(command, cwd=REPOSITORY, check=True)


if __name__ == "__main__":
    main()
