"""Audit whether LOCSP M8T NavPVT and NavSatFix encode one solution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS1_NOETIC)
    topics = {"/m8t/ublox/fix", "/m8t/ublox/navpvt"}
    records: dict[str, list[tuple[float, np.ndarray, np.ndarray]]] = {
        topic: [] for topic in topics
    }
    with AnyReader([args.bag], default_typestore=typestore) as reader:
        connections = [
            connection for connection in reader.connections
            if connection.topic in topics
        ]
        if {connection.topic for connection in connections} != topics:
            raise RuntimeError("LOCSP bag does not contain both M8T streams")
        for connection, timestamp_ns, rawdata in reader.messages(
            connections=connections
        ):
            message = reader.deserialize(rawdata, connection.msgtype)
            bag_time = timestamp_ns * 1e-9
            if connection.topic.endswith("/fix"):
                position = np.asarray(
                    [message.latitude, message.longitude, message.altitude],
                    dtype=float,
                )
                covariance = np.asarray(
                    message.position_covariance, dtype=float
                ).reshape(3, 3)
            else:
                position = np.asarray(
                    [
                        float(message.lat) * 1e-7,
                        float(message.lon) * 1e-7,
                        float(message.height) * 1e-3,
                    ],
                    dtype=float,
                )
                horizontal_sigma = float(message.hAcc) * 1e-3
                vertical_sigma = float(message.vAcc) * 1e-3
                covariance = np.diag(
                    [horizontal_sigma**2, horizontal_sigma**2, vertical_sigma**2]
                )
            records[connection.topic].append((bag_time, position, covariance))

    fix = records["/m8t/ublox/fix"]
    navpvt = records["/m8t/ublox/navpvt"]
    if not fix or not navpvt:
        raise RuntimeError("one or both M8T streams are empty")
    nav_times = np.asarray([item[0] for item in navpvt])
    pairs = []
    for fix_time, fix_position, fix_covariance in fix:
        nav_index = int(np.argmin(np.abs(nav_times - fix_time)))
        nav_time, nav_position, nav_covariance = navpvt[nav_index]
        pairs.append(
            (
                nav_time - fix_time,
                nav_position - fix_position,
                nav_covariance - fix_covariance,
            )
        )

    time_offsets = np.asarray([item[0] for item in pairs])
    position_differences = np.asarray([item[1] for item in pairs])
    covariance_differences = np.asarray([item[2] for item in pairs])
    maximum_position = float(np.max(np.abs(position_differences)))
    maximum_covariance = float(np.max(np.abs(covariance_differences)))
    payload = {
        "bag": str(args.bag),
        "truth_topic_opened": False,
        "fix_count": len(fix),
        "navpvt_count": len(navpvt),
        "paired_count": len(pairs),
        "maximum_absolute_pairing_offset_s": float(np.max(np.abs(time_offsets))),
        "maximum_absolute_geodetic_component_difference": maximum_position,
        "maximum_absolute_covariance_component_difference_m2": maximum_covariance,
        "same_position_solution": bool(maximum_position == 0.0),
        "covariance_equivalent_at_1e_12_m2": bool(maximum_covariance <= 1e-12),
        "timestamp_policy_for_factor_test": (
            "Use NavSatFix measurement epochs for both encodings after "
            "equivalence is established; driver publication latency must not "
            "create a second physical measurement epoch."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
