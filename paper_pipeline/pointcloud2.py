"""Zero-copy conversion of ROS PointCloud2 messages to NumPy."""

from __future__ import annotations

import numpy as np


_DATATYPES = {
    1: "i1",
    2: "u1",
    3: "i2",
    4: "u2",
    5: "i4",
    6: "u4",
    7: "f4",
    8: "f8",
}


def structured_array(message) -> np.ndarray:
    endian = ">" if bool(message.is_bigendian) else "<"
    names = []
    formats = []
    offsets = []
    for field in message.fields:
        datatype = int(field.datatype)
        if datatype not in _DATATYPES:
            continue
        base = np.dtype(endian + _DATATYPES[datatype])
        count = int(field.count)
        names.append(str(field.name))
        formats.append(base if count == 1 else (base, (count,)))
        offsets.append(int(field.offset))
    dtype = np.dtype(
        {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": int(message.point_step),
        }
    )
    raw = message.data.tobytes() if hasattr(message.data, "tobytes") else bytes(message.data)
    width = int(message.width)
    height = int(message.height)
    if int(message.row_step) == width * int(message.point_step):
        return np.frombuffer(raw, dtype=dtype, count=width * height)
    rows = []
    for row in range(height):
        offset = row * int(message.row_step)
        rows.append(np.frombuffer(raw, dtype=dtype, count=width, offset=offset))
    return np.concatenate(rows)


def xyz_and_relative_time(message) -> tuple[np.ndarray, np.ndarray]:
    points = structured_array(message)
    names = points.dtype.names or ()
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"PointCloud2 lacks x/y/z fields: {names}")
    xyz = np.column_stack([points["x"], points["y"], points["z"]]).astype(
        np.float64, copy=False
    )
    finite = np.all(np.isfinite(xyz), axis=1)
    time = np.zeros(len(points), dtype=np.float64)
    for candidate in ("time", "timestamp", "t"):
        if candidate not in names:
            continue
        raw_time = np.asarray(points[candidate], dtype=np.float64).reshape(-1)
        if candidate == "t" and np.nanmax(np.abs(raw_time)) > 1e6:
            raw_time *= 1e-9
        span = np.nanmax(raw_time) - np.nanmin(raw_time)
        if span > 0:
            time = (raw_time - np.nanmin(raw_time)) / span
        break
    return xyz[finite], time[finite]
