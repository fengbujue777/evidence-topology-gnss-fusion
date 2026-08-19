"""Canonicalize preregistered phone SPP timestamps to integer UTC seconds."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize(source: Path, destination: Path) -> dict[str, object]:
    output = []
    shifts = []
    epochs = 0
    for line in source.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("%", "#")):
            output.append(line)
            continue
        fields = stripped.split()
        instant = datetime.strptime(
            f"{fields[0]} {fields[1]}", "%Y/%m/%d %H:%M:%S.%f"
        ).replace(tzinfo=timezone.utc)
        normalized = (instant + timedelta(seconds=0.5)).replace(
            microsecond=0
        )
        shift = (normalized - instant).total_seconds()
        if abs(shift) > 0.5000001:
            raise AssertionError("clock phase correction exceeds 0.5 s")
        fields[0] = normalized.strftime("%Y/%m/%d")
        fields[1] = normalized.strftime("%H:%M:%S.000")
        output.append(" ".join(fields))
        shifts.append(shift)
        epochs += 1
    destination.write_text(
        "\n".join(output) + "\n",
        encoding="utf-8",
    )
    return {
        "source": str(source),
        "source_sha256": _sha256(source),
        "destination": str(destination),
        "destination_sha256": _sha256(destination),
        "epochs": epochs,
        "minimum_shift_s": min(shifts),
        "maximum_shift_s": max(shifts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_root.exists() and not args.overwrite:
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = {}
    for source in sorted(args.source_root.glob("phone.*.pos")):
        destination = args.output_root / source.name
        if destination.exists() and not args.overwrite:
            raise FileExistsError(destination)
        records[source.stem] = _normalize(source, destination)
    if len(records) != 4:
        raise AssertionError(f"expected four phone solutions, got {len(records)}")
    (args.output_root / "timestamp_normalization_manifest.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(records, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
