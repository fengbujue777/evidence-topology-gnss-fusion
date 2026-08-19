"""Streaming UrbanNav ROS1 reader and official KISS-ICP trajectory cache."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .geodesy import geodetic_to_local_enu
from .pointcloud2 import xyz_and_relative_time


POINTCLOUD_TOPICS = ("/velodyne_points",)
PACKET_TOPICS = ("/velodyne_packets",)
LIDAR_TOPICS = POINTCLOUD_TOPICS + PACKET_TOPICS
GT_TOPICS = ("/novatel_data/inspvax",)
GNSS_TOPICS = ("/ublox_node/fix", "/ublox/fix")
GPS_EPOCH_UNIX_S = 315964800.0
GPS_UTC_LEAP_SECONDS = 18.0


@dataclass
class ExtractionReport:
    dataset_id: str
    source_bags: list[str]
    lidar_frames: int
    ground_truth_samples: int
    receiver_gnss_samples: int
    lidar_duration_s: float
    kiss_runtime_s: float
    point_fields: list[str]
    topics: dict[str, str]
    state_frame: str
    extrinsics_source: str | None
    truth_source: str | None


def ros_timestamp_s(message, fallback_ns: int) -> float:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return fallback_ns * 1e-9
    seconds = getattr(stamp, "sec", getattr(stamp, "secs", None))
    nanoseconds = getattr(stamp, "nanosec", getattr(stamp, "nsecs", 0))
    if seconds is None:
        return fallback_ns * 1e-9
    return float(seconds) + float(nanoseconds) * 1e-9


def _first_attribute(message, names: tuple[str, ...]) -> float:
    for name in names:
        if hasattr(message, name):
            return float(getattr(message, name))
    raise AttributeError(f"none of {names} exists in {message.__class__.__name__}")


def _ins_record(message, timestamp_ns: int) -> tuple[float, ...]:
    return (
        ros_timestamp_s(message, timestamp_ns),
        _first_attribute(message, ("latitude", "lat")),
        _first_attribute(message, ("longitude", "lon")),
        _first_attribute(message, ("height", "altitude", "alt")),
        _first_attribute(message, ("roll",)),
        _first_attribute(message, ("pitch",)),
        _first_attribute(message, ("azimuth", "heading")),
    )


def _gnss_record(message, timestamp_ns: int) -> tuple[float, ...]:
    covariance = np.asarray(
        getattr(message, "position_covariance", np.eye(3).reshape(-1)),
        dtype=float,
    ).reshape(3, 3)
    return (
        ros_timestamp_s(message, timestamp_ns),
        float(message.latitude),
        float(message.longitude),
        float(message.altitude),
        *covariance.reshape(-1),
    )


def _poses_from_ins(
    records: np.ndarray,
    origin: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    timestamps = records[:, 0]
    positions, origin = geodetic_to_local_enu(
        records[:, 1], records[:, 2], records[:, 3], origin
    )
    roll = np.radians(records[:, 4])
    pitch = np.radians(records[:, 5])
    yaw_enu = np.radians(90.0 - records[:, 6])
    rotations = Rotation.from_euler(
        "xyz", np.column_stack([roll, pitch, yaw_enu])
    ).as_matrix()
    poses = np.repeat(np.eye(4)[None, :, :], len(records), axis=0)
    poses[:, :3, :3] = rotations
    poses[:, :3, 3] = positions
    return timestamps, poses, origin


def gps_week_tow_to_unix_seconds(
    gps_week: np.ndarray, tow_s: np.ndarray
) -> np.ndarray:
    return (
        GPS_EPOCH_UNIX_S
        + np.asarray(gps_week, dtype=float) * 604800.0
        + np.asarray(tow_s, dtype=float)
        - GPS_UTC_LEAP_SECONDS
    )


def load_reference_csv(
    path: Path,
    origin: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Load an official UrbanNav Tokyo CSV or Hong Kong raw reference.

    Tokyo columns are GPS TOW, GPS week, latitude, longitude, ellipsoid
    height, ECEF XYZ, roll, pitch and heading.  The Hong Kong raw files use
    Unix UTC, GPS week/TOW, DMS latitude/longitude, height, velocity,
    acceleration, roll, pitch, heading and quality.  In both formats heading
    is clockwise from north; the graph uses ENU yaw.
    """

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        first_line = handle.readline()
    if "UTCTime" in first_line:
        records = np.genfromtxt(path, skip_header=2)
        if records.ndim != 2 or records.shape[1] < 19:
            raise ValueError(
                f"unexpected UrbanNav HK reference format: {path}"
            )

        def dms_to_degrees(values: np.ndarray) -> np.ndarray:
            degrees = values[:, 0]
            sign = np.where(degrees < 0.0, -1.0, 1.0)
            return degrees + sign * (
                values[:, 1] / 60.0 + values[:, 2] / 3600.0
            )

        latitude = dms_to_degrees(records[:, 3:6])
        longitude = dms_to_degrees(records[:, 6:9])
        ins_like = np.column_stack(
            [
                records[:, 0],
                latitude,
                longitude,
                records[:, 9],
                records[:, 16],
                records[:, 17],
                records[:, 18],
            ]
        )
        return _poses_from_ins(ins_like, origin)

    records = np.genfromtxt(path, delimiter=",", skip_header=1)
    if records.ndim != 2 or records.shape[1] < 11:
        raise ValueError(f"unexpected UrbanNav reference format: {path}")
    timestamps = gps_week_tow_to_unix_seconds(records[:, 1], records[:, 0])
    ins_like = np.column_stack(
        [
            timestamps,
            records[:, 2],
            records[:, 3],
            records[:, 4],
            records[:, 8],
            records[:, 9],
            records[:, 10],
        ]
    )
    return _poses_from_ins(ins_like, origin)


def extract_urbannav(
    dataset_id: str,
    bag_paths: list[Path],
    output_root: Path,
    max_frames: int | None = None,
    truth_csv: Path | None = None,
    deskew: bool = False,
    extrinsics_json: Path | None = None,
) -> ExtractionReport:
    try:
        from kiss_icp.config import KISSConfig
        from kiss_icp.kiss_icp import KissICP
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as error:
        raise RuntimeError(
            "Install the packages in requirements.txt before extraction"
        ) from error

    lidar_to_state = np.eye(4)
    truth_to_state = np.eye(4)
    state_frame = "native_sensor_origins"
    if extrinsics_json is not None:
        extrinsics = json.loads(
            extrinsics_json.read_text(encoding="utf-8")
        )
        lidar_to_state = np.asarray(
            extrinsics["lidar_to_state"], dtype=float
        )
        truth_to_state = np.asarray(
            extrinsics["truth_to_state"], dtype=float
        )
        if lidar_to_state.shape != (4, 4) or truth_to_state.shape != (4, 4):
            raise ValueError("extrinsic transforms must be 4x4")
        state_frame = str(extrinsics["state_frame"])

    typestore = get_typestore(Stores.ROS1_NOETIC)
    config = KISSConfig(
        data={"max_range": 100.0, "min_range": 3.0, "deskew": deskew},
        mapping={"voxel_size": 1.0, "max_points_per_voxel": 20},
        registration={
            "max_num_iterations": 100,
            "convergence_criterion": 1e-4,
            "max_num_threads": 4,
        },
        adaptive_threshold={"initial_threshold": 2.0, "min_motion_th": 0.1},
    )
    odometry = KissICP(config)
    packet_decoder = None
    lidar_timestamps: list[float] = []
    lidar_poses: list[np.ndarray] = []
    ins_records: list[tuple[float, ...]] = []
    gnss_records: list[tuple[float, ...]] = []
    first_fields: list[str] = []
    topic_types: dict[str, str] = {}
    start = time.perf_counter()

    with AnyReader(bag_paths, default_typestore=typestore) as reader:
        selected_topics = set(LIDAR_TOPICS + GT_TOPICS + GNSS_TOPICS)
        connections = [
            connection
            for connection in reader.connections
            if connection.topic in selected_topics
        ]
        topic_types = {
            connection.topic: connection.msgtype for connection in connections
        }
        if not any(connection.topic in LIDAR_TOPICS for connection in connections):
            raise RuntimeError(f"no LiDAR topic in {bag_paths}; found {topic_types}")
        def register_lidar(
            timestamp_s: float, xyz: np.ndarray, relative_time: np.ndarray
        ) -> None:
            if max_frames is not None and len(lidar_timestamps) >= max_frames:
                return
            odometry.register_frame(xyz, relative_time)
            lidar_timestamps.append(float(timestamp_s))
            lidar_poses.append(odometry.last_pose.copy())

        for connection, timestamp_ns, rawdata in reader.messages(connections=connections):
            message = reader.deserialize(rawdata, connection.msgtype)
            if connection.topic in POINTCLOUD_TOPICS:
                if not first_fields:
                    first_fields = [str(field.name) for field in message.fields]
                xyz, relative_time = xyz_and_relative_time(message)
                register_lidar(
                    ros_timestamp_s(message, timestamp_ns), xyz, relative_time
                )
            elif connection.topic in PACKET_TOPICS:
                if packet_decoder is None:
                    try:
                        import velodyne_decoder
                    except ImportError as error:
                        raise RuntimeError(
                            "velodyne-decoder 3.1.0 is required for packet bags"
                        ) from error
                    packet_decoder = velodyne_decoder.StreamDecoder()
                    first_fields = [
                        "x",
                        "y",
                        "z",
                        "intensity",
                        "time",
                        "column",
                        "ring",
                        "return_type",
                    ]
                for packet in message.packets:
                    stamp_s = (
                        float(packet.stamp.sec)
                        + float(packet.stamp.nanosec) * 1e-9
                    )
                    decoded = packet_decoder.decode(stamp_s, packet.data)
                    if decoded is None:
                        continue
                    stamp_pair, points = decoded
                    relative_time = np.asarray(points[:, 4], dtype=float)
                    span = np.ptp(relative_time)
                    if span > 0:
                        relative_time = (
                            relative_time - np.min(relative_time)
                        ) / span
                    register_lidar(
                        float(stamp_pair.host),
                        np.asarray(points[:, :3], dtype=float),
                        relative_time,
                    )
                    if (
                        max_frames is not None
                        and len(lidar_timestamps) >= max_frames
                    ):
                        break
            elif connection.topic in GT_TOPICS:
                ins_records.append(_ins_record(message, timestamp_ns))
            elif connection.topic in GNSS_TOPICS:
                gnss_records.append(_gnss_record(message, timestamp_ns))
            if max_frames is not None and len(lidar_timestamps) >= max_frames:
                break

    if not lidar_timestamps:
        raise RuntimeError("no LiDAR frames were extracted")
    output_root.mkdir(parents=True, exist_ok=True)
    common_origin: tuple[float, float, float] | None = None
    if gnss_records:
        first_gnss = gnss_records[0]
        common_origin = (
            float(first_gnss[1]),
            float(first_gnss[2]),
            float(first_gnss[3]),
        )
    sensor_path = output_root / f"{dataset_id}.estimator_sensor_cache.npz"
    np.savez_compressed(
        sensor_path,
        lidar_timestamps_s=np.asarray(lidar_timestamps),
        lidar_poses_local=np.asarray(lidar_poses) @ lidar_to_state,
        lidar_poses_local_native=np.asarray(lidar_poses),
        receiver_gnss_records=np.asarray(gnss_records, dtype=float),
        enu_origin_lat_lon_alt=np.asarray(
            common_origin if common_origin is not None else [], dtype=float
        ),
        state_frame=np.asarray(state_frame),
        extrinsics_source=np.asarray(
            str(extrinsics_json) if extrinsics_json is not None else ""
        ),
    )
    if truth_csv is not None:
        truth_timestamps, truth_poses, common_origin = load_reference_csv(
            truth_csv, common_origin
        )
        np.savez_compressed(
            output_root / f"{dataset_id}.evaluation_truth_cache.npz",
            truth_timestamps_s=truth_timestamps,
            truth_poses_enu=truth_poses @ truth_to_state,
            truth_poses_enu_native=truth_poses,
            enu_origin_lat_lon_alt=np.asarray(common_origin),
            state_frame=np.asarray(state_frame),
        )
    elif ins_records:
        truth_timestamps, truth_poses, common_origin = _poses_from_ins(
            np.asarray(ins_records, dtype=float), common_origin
        )
        np.savez_compressed(
            output_root / f"{dataset_id}.evaluation_truth_cache.npz",
            truth_timestamps_s=truth_timestamps,
            truth_poses_enu=truth_poses @ truth_to_state,
            truth_poses_enu_native=truth_poses,
            enu_origin_lat_lon_alt=np.asarray(common_origin),
            state_frame=np.asarray(state_frame),
        )

    report = ExtractionReport(
        dataset_id=dataset_id,
        source_bags=[str(path) for path in bag_paths],
        lidar_frames=len(lidar_timestamps),
        ground_truth_samples=len(ins_records),
        receiver_gnss_samples=len(gnss_records),
        lidar_duration_s=float(lidar_timestamps[-1] - lidar_timestamps[0]),
        kiss_runtime_s=float(time.perf_counter() - start),
        point_fields=first_fields,
        topics=topic_types,
        state_frame=state_frame,
        extrinsics_source=(
            str(extrinsics_json) if extrinsics_json is not None else None
        ),
        truth_source=(
            str(truth_csv)
            if truth_csv is not None
            else (
                "embedded:/novatel_data/inspvax"
                if ins_records
                else None
            )
        ),
    )
    (output_root / f"{dataset_id}.extraction_report.json").write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--bag", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument(
        "--deskew",
        action="store_true",
        help="Use per-point packet times for KISS-ICP motion compensation.",
    )
    parser.add_argument(
        "--extrinsics-json",
        type=Path,
        help=(
            "JSON with lidar_to_state and truth_to_state 4x4 transforms. "
            "Use it to evaluate all trajectories at the GNSS antenna."
        ),
    )
    args = parser.parse_args()
    report = extract_urbannav(
        args.dataset_id,
        args.bag,
        args.output_root,
        args.max_frames,
        args.truth_csv,
        args.deskew,
        args.extrinsics_json,
    )
    print(json.dumps(asdict(report), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
