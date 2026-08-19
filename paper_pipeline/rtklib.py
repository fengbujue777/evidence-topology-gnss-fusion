"""Frozen RTKLIB SPP execution and solution parsing."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .geodesy import geodetic_to_local_enu


RTKLIB_COMMIT = "180043ee24b6d2b168f98b64be15f69d50046b1a"
DEFAULT_BINARY = Path(
    os.environ.get("RTKLIB_RNX2RTKP", shutil.which("rnx2rtkp") or "rnx2rtkp")
)


def run_spp(
    observation_files: list[Path],
    navigation_files: list[Path],
    configuration: Path,
    output_path: Path,
    binary: Path = DEFAULT_BINARY,
) -> None:
    if not observation_files:
        raise ValueError("no RINEX observation file supplied")
    if not navigation_files:
        raise ValueError("no RINEX navigation file supplied")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(binary),
        "-k",
        str(configuration),
        "-o",
        str(output_path),
        *[str(path) for path in observation_files],
        *[str(path) for path in navigation_files],
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, check=False
    )
    log_path = output_path.with_suffix(output_path.suffix + ".run.json")
    log_path.write_text(
        json.dumps(
            {
                "command": command,
                "rtklib_commit": RTKLIB_COMMIT,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    invalid_configuration = "invalid option value" in result.stderr.lower()
    if (
        result.returncode != 0
        or not output_path.exists()
        or invalid_configuration
    ):
        raise RuntimeError(f"rnx2rtkp failed; see {log_path}")


def parse_solution(
    path: Path,
    origin_lat_lon_alt: tuple[float, float, float] | None = None,
    use_full_covariance: bool = False,
) -> dict[str, np.ndarray | tuple[float, float, float]]:
    timestamps = []
    latitude = []
    longitude = []
    altitude = []
    covariance = []
    quality = []
    satellites = []
    for raw_line in path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%") or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 10:
            continue
        instant = datetime.strptime(
            f"{fields[0]} {fields[1]}", "%Y/%m/%d %H:%M:%S.%f"
        ).replace(tzinfo=timezone.utc)
        timestamps.append(instant.timestamp())
        latitude.append(float(fields[2]))
        longitude.append(float(fields[3]))
        altitude.append(float(fields[4]))
        quality.append(int(fields[5]))
        satellites.append(int(fields[6]))
        standard_north = max(float(fields[7]), 0.25)
        standard_east = max(float(fields[8]), 0.25)
        standard_up = max(float(fields[9]), 0.50)
        covariance_enu = np.diag(
            [
                standard_east**2,
                standard_north**2,
                standard_up**2,
            ]
        )
        if use_full_covariance and len(fields) >= 13:
            # RTKLIB prints signed square roots of the N-E, E-U and U-N
            # covariance terms.  Reconstruct the signed covariance and
            # reorder N/E/U to the project's E/N/U convention.
            standard_ne = float(fields[10])
            standard_eu = float(fields[11])
            standard_un = float(fields[12])

            def signed_square(value: float) -> float:
                return float(np.sign(value) * value**2)

            covariance_neu = np.array(
                [
                    [
                        standard_north**2,
                        signed_square(standard_ne),
                        signed_square(standard_un),
                    ],
                    [
                        signed_square(standard_ne),
                        standard_east**2,
                        signed_square(standard_eu),
                    ],
                    [
                        signed_square(standard_un),
                        signed_square(standard_eu),
                        standard_up**2,
                    ],
                ]
            )
            covariance_enu = covariance_neu[
                np.ix_([1, 0, 2], [1, 0, 2])
            ]
            values, vectors = np.linalg.eigh(covariance_enu)
            covariance_enu = (
                vectors * np.maximum(values, 1e-6)
            ) @ vectors.T
        covariance.append(covariance_enu)
    if not timestamps:
        raise ValueError(f"no RTKLIB solution epochs in {path}")
    positions, origin = geodetic_to_local_enu(
        np.asarray(latitude),
        np.asarray(longitude),
        np.asarray(altitude),
        origin_lat_lon_alt,
    )
    timestamps_array = np.asarray(timestamps, dtype=float)
    keep = np.concatenate(
        [[True], np.diff(timestamps_array) > 1e-6]
    )
    return {
        "timestamps_s": timestamps_array[keep],
        "positions_enu_m": positions[keep],
        "covariances_m2": np.asarray(covariance)[keep],
        "quality": np.asarray(quality, dtype=int)[keep],
        "satellites": np.asarray(satellites, dtype=int)[keep],
        "enu_origin_lat_lon_alt": origin,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", nargs="+", type=Path, required=True)
    parser.add_argument("--navigation", nargs="+", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_spp(
        args.observation,
        args.navigation,
        args.configuration,
        args.output,
    )
    parsed = parse_solution(args.output)
    print(
        json.dumps(
            {
                "epochs": len(parsed["timestamps_s"]),
                "start_s": float(parsed["timestamps_s"][0]),
                "end_s": float(parsed["timestamps_s"][-1]),
                "origin": parsed["enu_origin_lat_lon_alt"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
