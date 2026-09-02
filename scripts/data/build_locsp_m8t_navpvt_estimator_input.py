"""Build a LOCSP M8T NavPVT estimator stream after a source-equivalence audit.

NavPVT and NavSatFix are emitted by the same ublox driver at the same physical
measurement epoch.  The NavSatFix timestamps and already transformed ENU
positions are reused only after direct bag inspection establishes equality of
all geodetic positions.  Covariances are reconstructed from NavPVT hAcc/vAcc.
NovAtel truth is neither requested nor read.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("fix_input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS1_NOETIC)
    fix_records = []
    navpvt_records = []
    topics = {"/m8t/ublox/fix", "/m8t/ublox/navpvt"}
    with AnyReader([args.bag], default_typestore=typestore) as reader:
        connections = [c for c in reader.connections if c.topic in topics]
        for connection, _, rawdata in reader.messages(connections=connections):
            message = reader.deserialize(rawdata, connection.msgtype)
            if connection.topic.endswith("/fix"):
                fix_records.append(
                    np.asarray(
                        [message.latitude, message.longitude, message.altitude],
                        dtype=float,
                    )
                )
            else:
                position = np.asarray(
                    [
                        float(message.lat) * 1e-7,
                        float(message.lon) * 1e-7,
                        float(message.height) * 1e-3,
                    ],
                    dtype=float,
                )
                h_sigma = float(message.hAcc) * 1e-3
                v_sigma = float(message.vAcc) * 1e-3
                navpvt_records.append(
                    (position, np.diag([h_sigma**2, h_sigma**2, v_sigma**2]))
                )

    if len(fix_records) != len(navpvt_records):
        raise ValueError(
            f"stream count mismatch: fix={len(fix_records)}, "
            f"navpvt={len(navpvt_records)}"
        )
    fix_geodetic = np.asarray(fix_records)
    navpvt_geodetic = np.asarray([item[0] for item in navpvt_records])
    if not np.array_equal(fix_geodetic, navpvt_geodetic):
        maximum = float(np.max(np.abs(fix_geodetic - navpvt_geodetic)))
        raise ValueError(f"NavPVT positions are not exact Fix replays: {maximum}")

    source = np.load(args.fix_input, allow_pickle=False)
    if len(source["gnss_timestamps_s"]) != len(navpvt_records):
        raise ValueError("estimator input and bag stream count mismatch")
    navpvt_covariances = np.asarray([item[1] for item in navpvt_records])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        lidar_timestamps_s=source["lidar_timestamps_s"],
        lidar_poses_local=source["lidar_poses_local"],
        gnss_timestamps_s=source["gnss_timestamps_s"],
        gnss_positions_enu_m=source["gnss_positions_enu_m"],
        gnss_covariances_m2=navpvt_covariances,
    )
    print(
        f"wrote {args.output} with {len(navpvt_records)} source-equivalent "
        "NavPVT epochs; truth was not opened"
    )


if __name__ == "__main__":
    main()
