"""Truth-isolated LOCSP adapter for receiver-topology external validation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .geodesy import geodetic_to_local_enu
from .pointcloud2 import xyz_and_relative_time
from .urbannav_bag import ros_timestamp_s


LIDAR_TOPICS = {
    "/velodyne_points": "velodyne_frame",
    "/hesai/pandar": "hesai_frame",
}
M8T_TOPIC = "/m8t/ublox/fix"
SEPTENTRIO_TOPIC = "/sept/fix"
TRUTH_TOPIC = "/novatel/oem7/inspva"


@dataclass
class LocspExtractionReport:
    dataset_id: str
    source_bag: str
    source_bag_bytes: int
    lidar_frames: int
    lidar_duration_s: float
    m8t_samples: int
    septentrio_samples: int
    truth_samples: int
    kiss_runtime_s: float
    state_frame: str
    lidar_topic: str
    receiver_topics: list[str]
    truth_topic: str
    truth_excluded_from_estimator: bool
    tf_json: str
    point_fields: list[str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _transform(translation, quaternion_xyzw) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    result[:3, 3] = np.asarray(translation, dtype=float)
    return result


def _load_extrinsics(
    path: Path, lidar_frame: str
) -> tuple[np.ndarray, np.ndarray]:
    records = json.loads(path.read_text(encoding="utf-8"))
    transforms = {
        str(record["child"]): _transform(
            record["translation"], record["rotation"]
        )
        for record in records
        if record.get("parent") == "base_link"
    }
    required = {lidar_frame, "novatel_frame", "antenne_frame"}
    missing = sorted(required - transforms.keys())
    if missing:
        raise ValueError(f"missing LOCSP static transforms: {missing}")
    base_from_lidar = transforms[lidar_frame]
    base_from_novatel = transforms["novatel_frame"]
    base_from_state = transforms["antenne_frame"]
    lidar_to_state = np.linalg.inv(base_from_lidar) @ base_from_state
    novatel_to_state = np.linalg.inv(base_from_novatel) @ base_from_state
    return lidar_to_state, novatel_to_state


def _navsat_record(message, timestamp_ns: int) -> tuple[float, ...] | None:
    status = int(getattr(getattr(message, "status", None), "status", -1))
    values = np.asarray(
        [message.latitude, message.longitude, message.altitude], dtype=float
    )
    if status < 0 or not np.all(np.isfinite(values)):
        return None
    covariance = np.asarray(message.position_covariance, dtype=float).reshape(3, 3)
    covariance = 0.5 * (covariance + covariance.T)
    if not np.all(np.isfinite(covariance)) or np.any(np.diag(covariance) <= 0.0):
        covariance = np.eye(3) * 25.0
    return (
        ros_timestamp_s(message, timestamp_ns),
        *values,
        *covariance.reshape(-1),
    )


def _truth_record(message, timestamp_ns: int) -> tuple[float, ...] | None:
    status = int(getattr(getattr(message, "status", None), "status", -1))
    values = np.asarray(
        [
            message.latitude,
            message.longitude,
            message.height,
            message.roll,
            message.pitch,
            message.azimuth,
        ],
        dtype=float,
    )
    if status != 3 or not np.all(np.isfinite(values)):
        return None
    return (ros_timestamp_s(message, timestamp_ns), *values)


def _truth_poses(
    records: np.ndarray,
    novatel_to_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    origin = (float(records[0, 1]), float(records[0, 2]), float(records[0, 3]))
    positions, _ = geodetic_to_local_enu(
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
    return records[:, 0], poses @ novatel_to_state, origin


def _receiver_arrays(
    records: np.ndarray,
    origin: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions, _ = geodetic_to_local_enu(
        records[:, 1], records[:, 2], records[:, 3], origin
    )
    covariances = records[:, 4:].reshape(-1, 3, 3)
    return records[:, 0], positions, covariances


def extract_locsp(
    dataset_id: str,
    bag_path: Path,
    tf_json: Path,
    case_root: Path,
    cache_root: Path,
    lidar_topic: str = "/velodyne_points",
    max_frames: int | None = None,
    compute_source_sha256: bool = True,
) -> LocspExtractionReport:
    from kiss_icp.config import KISSConfig
    from kiss_icp.kiss_icp import KissICP
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    if lidar_topic not in LIDAR_TOPICS:
        raise ValueError(f"unsupported LOCSP LiDAR topic: {lidar_topic}")
    lidar_to_state, novatel_to_state = _load_extrinsics(
        tf_json, LIDAR_TOPICS[lidar_topic]
    )
    typestore = get_typestore(Stores.ROS1_NOETIC)
    config = KISSConfig(
        data={"max_range": 100.0, "min_range": 3.0, "deskew": False},
        mapping={"voxel_size": 1.0, "max_points_per_voxel": 20},
        registration={
            "max_num_iterations": 100,
            "convergence_criterion": 1e-4,
            "max_num_threads": 4,
        },
        adaptive_threshold={"initial_threshold": 2.0, "min_motion_th": 0.1},
    )
    odometry = KissICP(config)
    lidar_times: list[float] = []
    lidar_poses: list[np.ndarray] = []
    m8t_records: list[tuple[float, ...]] = []
    sept_records: list[tuple[float, ...]] = []
    truth_records: list[tuple[float, ...]] = []
    point_fields: list[str] = []
    started = time.perf_counter()

    selected = {lidar_topic, M8T_TOPIC, SEPTENTRIO_TOPIC, TRUTH_TOPIC}
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        connections = [
            connection
            for connection in reader.connections
            if connection.topic in selected
        ]
        found = {connection.topic for connection in connections}
        if found != selected:
            raise RuntimeError(f"missing LOCSP topics: {sorted(selected - found)}")
        for connection, timestamp_ns, rawdata in reader.messages(
            connections=connections
        ):
            message = reader.deserialize(rawdata, connection.msgtype)
            if connection.topic == lidar_topic:
                if max_frames is not None and len(lidar_times) >= max_frames:
                    continue
                if not point_fields:
                    point_fields = [str(field.name) for field in message.fields]
                xyz, relative_time = xyz_and_relative_time(message)
                odometry.register_frame(xyz, relative_time)
                # The LOCSP Hesai driver preserves an old sensor-clock epoch
                # (2020) in the PointCloud2 header although the synchronized
                # ROS bag record is dated 2024.  Its frame intervals agree
                # with bag time, so use the record timestamp for that topic.
                lidar_times.append(
                    timestamp_ns * 1e-9
                    if lidar_topic == "/hesai/pandar"
                    else ros_timestamp_s(message, timestamp_ns)
                )
                lidar_poses.append(odometry.last_pose.copy() @ lidar_to_state)
            elif connection.topic == M8T_TOPIC:
                record = _navsat_record(message, timestamp_ns)
                if record is not None:
                    m8t_records.append(record)
            elif connection.topic == SEPTENTRIO_TOPIC:
                record = _navsat_record(message, timestamp_ns)
                if record is not None:
                    sept_records.append(record)
            elif connection.topic == TRUTH_TOPIC:
                record = _truth_record(message, timestamp_ns)
                if record is not None:
                    truth_records.append(record)

    if len(lidar_times) < 1200:
        raise RuntimeError(f"insufficient LiDAR frames: {len(lidar_times)}")
    if min(len(m8t_records), len(sept_records)) < 240:
        raise RuntimeError("insufficient receiver observations")
    if len(truth_records) < 1200:
        raise RuntimeError("insufficient NovAtel reference observations")

    truth_times, truth_poses, origin = _truth_poses(
        np.asarray(truth_records, dtype=float), novatel_to_state
    )
    receiver_payload = {
        "locsp_m8t": _receiver_arrays(np.asarray(m8t_records), origin),
        "locsp_septentrio": _receiver_arrays(np.asarray(sept_records), origin),
    }
    lidar_times_array = np.asarray(lidar_times, dtype=float)
    lidar_poses_array = np.asarray(lidar_poses, dtype=float)
    case_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        cache_root / f"{dataset_id}.compact_sensor_cache.npz",
        lidar_timestamps_s=lidar_times_array,
        lidar_poses_local=lidar_poses_array,
        m8t_timestamps_s=receiver_payload["locsp_m8t"][0],
        m8t_positions_enu_m=receiver_payload["locsp_m8t"][1],
        m8t_covariances_m2=receiver_payload["locsp_m8t"][2],
        septentrio_timestamps_s=receiver_payload["locsp_septentrio"][0],
        septentrio_positions_enu_m=receiver_payload["locsp_septentrio"][1],
        septentrio_covariances_m2=receiver_payload["locsp_septentrio"][2],
        enu_origin_lat_lon_alt=np.asarray(origin),
    )
    np.savez_compressed(
        cache_root / f"{dataset_id}.evaluation_truth_cache.npz",
        truth_timestamps_s=truth_times,
        truth_poses_enu=truth_poses,
        enu_origin_lat_lon_alt=np.asarray(origin),
    )

    for stream, (timestamps, positions, covariances) in receiver_payload.items():
        stem = f"{dataset_id}__{stream}__natural"
        np.savez_compressed(
            case_root / f"{stem}.estimator_input.npz",
            lidar_timestamps_s=lidar_times_array,
            lidar_poses_local=lidar_poses_array,
            gnss_timestamps_s=timestamps,
            gnss_positions_enu_m=positions,
            gnss_covariances_m2=covariances,
        )
        np.savez_compressed(
            case_root / f"{stem}.hidden_labels.npz",
            truth_timestamps_s=truth_times,
            truth_poses_enu=truth_poses,
        )

    report = LocspExtractionReport(
        dataset_id=dataset_id,
        source_bag=str(bag_path),
        source_bag_bytes=bag_path.stat().st_size,
        lidar_frames=len(lidar_times),
        lidar_duration_s=float(lidar_times_array[-1] - lidar_times_array[0]),
        m8t_samples=len(m8t_records),
        septentrio_samples=len(sept_records),
        truth_samples=len(truth_records),
        kiss_runtime_s=float(time.perf_counter() - started),
        state_frame="antenne_frame",
        lidar_topic=lidar_topic,
        receiver_topics=[M8T_TOPIC, SEPTENTRIO_TOPIC],
        truth_topic=TRUTH_TOPIC,
        truth_excluded_from_estimator=True,
        tf_json=str(tf_json),
        point_fields=point_fields,
    )
    payload = asdict(report)
    payload["source_bag_sha256"] = (
        _sha256(bag_path) if compute_source_sha256 else None
    )
    payload["source_hash_note"] = (
        "computed for this extraction"
        if compute_source_sha256
        else "omitted because the identical bag was hashed in the preceding Velodyne extraction"
    )
    payload["case_files"] = sorted(path.name for path in case_root.glob("*.npz"))
    (cache_root / f"{dataset_id}.extraction_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default="locsp_cr2")
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--tf-json", type=Path, required=True)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--lidar-topic",
        choices=tuple(LIDAR_TOPICS),
        default="/velodyne_points",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--skip-source-sha256", action="store_true")
    args = parser.parse_args()
    report = extract_locsp(
        args.dataset_id,
        args.bag,
        args.tf_json,
        args.case_root,
        args.cache_root,
        args.lidar_topic,
        args.max_frames,
        not args.skip_source_sha256,
    )
    print(json.dumps(asdict(report), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
