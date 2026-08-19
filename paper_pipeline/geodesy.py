"""Minimal WGS-84 geodesy used by all dataset adapters."""

from __future__ import annotations

import numpy as np


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def geodetic_to_ecef(
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    altitude_m: np.ndarray,
) -> np.ndarray:
    latitude = np.radians(np.asarray(latitude_deg, dtype=float))
    longitude = np.radians(np.asarray(longitude_deg, dtype=float))
    altitude = np.asarray(altitude_m, dtype=float)
    sin_latitude = np.sin(latitude)
    normal = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_latitude**2)
    x = (normal + altitude) * np.cos(latitude) * np.cos(longitude)
    y = (normal + altitude) * np.cos(latitude) * np.sin(longitude)
    z = (normal * (1.0 - WGS84_E2) + altitude) * sin_latitude
    return np.column_stack([x, y, z])


def ecef_to_enu(
    ecef_m: np.ndarray,
    origin_latitude_deg: float,
    origin_longitude_deg: float,
    origin_altitude_m: float,
) -> np.ndarray:
    origin = geodetic_to_ecef(
        np.array([origin_latitude_deg]),
        np.array([origin_longitude_deg]),
        np.array([origin_altitude_m]),
    )[0]
    latitude = np.radians(origin_latitude_deg)
    longitude = np.radians(origin_longitude_deg)
    rotation = np.array(
        [
            [-np.sin(longitude), np.cos(longitude), 0.0],
            [
                -np.sin(latitude) * np.cos(longitude),
                -np.sin(latitude) * np.sin(longitude),
                np.cos(latitude),
            ],
            [
                np.cos(latitude) * np.cos(longitude),
                np.cos(latitude) * np.sin(longitude),
                np.sin(latitude),
            ],
        ]
    )
    return (rotation @ (np.asarray(ecef_m) - origin).T).T


def geodetic_to_local_enu(
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    altitude_m: np.ndarray,
    origin: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    latitude = np.asarray(latitude_deg, dtype=float)
    longitude = np.asarray(longitude_deg, dtype=float)
    altitude = np.asarray(altitude_m, dtype=float)
    if origin is None:
        origin = (float(latitude[0]), float(longitude[0]), float(altitude[0]))
    ecef = geodetic_to_ecef(latitude, longitude, altitude)
    return ecef_to_enu(ecef, *origin), origin
