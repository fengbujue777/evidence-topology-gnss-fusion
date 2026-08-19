"""Generate the frozen receiver-level SPP solution panel with RTKLIB.

The panel JSON contains only dataset-relative globs and output names.  Users
select the downloaded RINEX root and a separate processed-output directory on
the command line, so no author-machine path is embedded in the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from paper_pipeline.rtklib import RTKLIB_COMMIT, run_spp


REPOSITORY = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fallback_glob(root: Path, patterns: list[str], label: str) -> list[Path]:
    for pattern in patterns:
        matches = sorted(path for path in root.rglob(pattern) if path.is_file())
        if matches:
            return matches
    raise FileNotFoundError(
        f"no {label} below {root}; tried patterns {patterns}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--rinex-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--configuration",
        type=Path,
        default=REPOSITORY / "configs" / "rtklib_spp_2.4.3_b34.conf",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    panel_path = args.panel.resolve()
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    streams = list(panel.get("streams", []))
    if not streams:
        raise ValueError("panel must define a non-empty streams list")
    navigation = _fallback_glob(
        args.rinex_root.resolve(),
        list(panel["navigation_globs"]),
        "RINEX navigation file",
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for stream in streams:
        observations = _fallback_glob(
            args.rinex_root.resolve(),
            list(stream["observation_globs"]),
            f"observation file for {stream['id']}",
        )
        output = args.output_root / str(stream["solution"])
        if output.exists() and not args.overwrite:
            raise FileExistsError(
                f"{output} already exists; use --overwrite intentionally"
            )
        if output.exists():
            output.unlink()
        run_spp(observations, navigation, args.configuration, output)
        records.append(
            {
                "stream_id": stream["id"],
                "physical_source": stream.get("physical_source", stream["id"]),
                "observations": [
                    {"path": str(path), "sha256": _sha256(path)}
                    for path in observations
                ],
                "navigation": [
                    {"path": str(path), "sha256": _sha256(path)}
                    for path in navigation
                ],
                "solution": str(output),
                "solution_sha256": _sha256(output),
            }
        )

    payload = {
        "dataset_id": panel["dataset_id"],
        "rtklib_commit": RTKLIB_COMMIT,
        "configuration": str(args.configuration.resolve()),
        "configuration_sha256": _sha256(args.configuration),
        "streams": records,
    }
    manifest = args.output_root / "spp_manifest.json"
    manifest.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
