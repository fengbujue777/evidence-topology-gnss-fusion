from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_receiver_panel_paths_are_portable_and_sources_are_explicit() -> None:
    panels = sorted((ROOT / "configs" / "receiver_panels").glob("*.json"))
    assert len(panels) == 4
    for path in panels:
        panel = json.loads(path.read_text(encoding="utf-8"))
        assert panel["streams"]
        for stream in panel["streams"]:
            solution = str(stream["solution"])
            assert not Path(solution).is_absolute()
            assert not PureWindowsPath(solution).is_absolute()
            assert stream["physical_source"]


def test_extrinsic_transforms_are_homogeneous() -> None:
    for path in (ROOT / "configs" / "extrinsics").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key in ("lidar_to_state", "truth_to_state"):
            if key not in payload:
                continue
            transform = np.asarray(payload[key], dtype=float)
            assert transform.shape == (4, 4)
            assert np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0])


def test_dependent_solution_streams_map_to_one_physical_source() -> None:
    registry = json.loads(
        (ROOT / "PROVENANCE_REGISTRY.json").read_text(encoding="utf-8")
    )["solution_stream_to_physical_source"]
    keys = ("ublox_m8t_gc", "ublox_m8t_gej", "ublox_m8t_gr")
    assert {registry[key] for key in keys} == {"ublox_m8t"}
    assert registry["locsp_m8t"] == registry["locsp_m8t_navpvt"]
