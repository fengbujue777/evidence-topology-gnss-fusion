"""Time-based, reproducible GNSS soft-fault injection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import FaultRealization


@dataclass(frozen=True)
class FaultConfig:
    sigma_horizontal_m: float = 0.8
    sigma_vertical_m: float = 1.2
    start_fraction_low: float = 0.20
    start_fraction_high: float = 0.35
    duration_fraction: float = 0.40
    minimum_duration_s: float = 60.0
    maximum_duration_s: float = 180.0
    recovery_tail_s: float = 30.0
    detection_sigma: float = 2.0
    earliest_fault_start_s: float | None = None


def sample_truth_positions(
    truth_timestamps_s: np.ndarray,
    truth_positions_m: np.ndarray,
    sample_timestamps_s: np.ndarray,
) -> np.ndarray:
    truth_timestamps_s = np.asarray(truth_timestamps_s, dtype=float)
    truth_positions_m = np.asarray(truth_positions_m, dtype=float)
    sample_timestamps_s = np.asarray(sample_timestamps_s, dtype=float)
    if truth_positions_m.shape != (len(truth_timestamps_s), 3):
        raise ValueError("truth_positions_m must have shape (N, 3)")
    if sample_timestamps_s[0] < truth_timestamps_s[0] or sample_timestamps_s[-1] > truth_timestamps_s[-1]:
        raise ValueError("sample timestamps extend beyond truth")
    return np.column_stack(
        [
            np.interp(sample_timestamps_s, truth_timestamps_s, truth_positions_m[:, axis])
            for axis in range(3)
        ]
    )


def _fault_interval(
    timestamps_s: np.ndarray, rng: np.random.Generator, config: FaultConfig
) -> tuple[float, float]:
    start_time = float(timestamps_s[0])
    end_time = float(timestamps_s[-1])
    total = end_time - start_time
    fraction = rng.uniform(config.start_fraction_low, config.start_fraction_high)
    fault_start = start_time + fraction * total
    if config.earliest_fault_start_s is not None:
        fault_start = max(fault_start, config.earliest_fault_start_s)
    available = max(end_time - fault_start - config.recovery_tail_s, 0.0)
    requested = np.clip(
        config.duration_fraction * total,
        config.minimum_duration_s,
        config.maximum_duration_s,
    )
    duration = min(requested, available)
    if duration < config.minimum_duration_s:
        raise ValueError(
            "trajectory is too short for the registered post-alignment "
            "guard, minimum fault duration and recovery tail"
        )
    return fault_start, fault_start + duration


def _random_direction(rng: np.random.Generator) -> np.ndarray:
    angle = rng.uniform(-np.pi, np.pi)
    vertical_sign = rng.choice([-1.0, 1.0])
    return np.array([np.cos(angle), np.sin(angle), 0.25 * vertical_sign])


def generate_controlled_gnss(
    truth_timestamps_s: np.ndarray,
    truth_positions_m: np.ndarray,
    scenario: str,
    seed: int,
    config: FaultConfig = FaultConfig(),
) -> FaultRealization:
    """Generate the exact controlled panel specified in the protocol.

    Sampling is fixed at 1 Hz in the time overlap of the supplied truth.
    The same seed produces the same nominal noise, onset, duration and fault
    direction for every estimator.
    """

    truth_timestamps_s = np.asarray(truth_timestamps_s, dtype=float)
    truth_positions_m = np.asarray(truth_positions_m, dtype=float)
    first = float(np.ceil(truth_timestamps_s[0]))
    last = float(np.floor(truth_timestamps_s[-1]))
    timestamps = np.arange(first, last + 0.5, 1.0)
    if len(timestamps) < 10:
        raise ValueError("trajectory does not contain enough 1 Hz GNSS epochs")
    truth = sample_truth_positions(truth_timestamps_s, truth_positions_m, timestamps)

    rng = np.random.default_rng(seed)
    noise = np.column_stack(
        [
            rng.normal(0.0, config.sigma_horizontal_m, len(timestamps)),
            rng.normal(0.0, config.sigma_horizontal_m, len(timestamps)),
            rng.normal(0.0, config.sigma_vertical_m, len(timestamps)),
        ]
    )
    covariance = np.repeat(
        np.diag(
            [
                config.sigma_horizontal_m**2,
                config.sigma_horizontal_m**2,
                config.sigma_vertical_m**2,
            ]
        )[None, :, :],
        len(timestamps),
        axis=0,
    )
    bias = np.zeros((len(timestamps), 3), dtype=float)
    faulty = np.zeros(len(timestamps), dtype=bool)
    direction = _random_direction(rng)

    if scenario == "clean":
        return FaultRealization(
            timestamps,
            truth + noise,
            covariance,
            bias,
            faulty,
            faulty.copy(),
            None,
            None,
            scenario,
            seed,
        )

    fault_start, fault_end = _fault_interval(timestamps, rng, config)
    active = (timestamps >= fault_start) & (timestamps <= fault_end)
    elapsed = np.maximum(timestamps - fault_start, 0.0)

    if scenario.startswith("ramp_"):
        rates = {"ramp_005": 0.05, "ramp_010": 0.10, "ramp_020": 0.20}
        if scenario not in rates:
            raise ValueError(f"unknown ramp scenario: {scenario}")
        horizontal = np.minimum(rates[scenario] * elapsed, 6.0)
        vertical = np.minimum(0.25 * rates[scenario] * elapsed, 1.5)
        unit_xy = direction[:2] / np.linalg.norm(direction[:2])
        bias[:, :2] = horizontal[:, None] * unit_xy
        bias[:, 2] = vertical * np.sign(direction[2])
    elif scenario == "gauss_markov_30":
        rho = np.exp(-1.0 / 30.0)
        stationary_sigma = np.array([3.0, 3.0, 1.0])
        innovation_sigma = np.sqrt(1.0 - rho**2) * stationary_sigma
        state = np.zeros(3)
        for index in np.flatnonzero(active):
            state = rho * state + rng.normal(0.0, innovation_sigma)
            bias[index] = state
    elif scenario == "false_fix_drift":
        horizontal = np.minimum(2.0 + 0.08 * elapsed, 8.0)
        unit_xy = direction[:2] / np.linalg.norm(direction[:2])
        bias[:, :2] = horizontal[:, None] * unit_xy
        bias[:, 2] = np.minimum(0.02 * elapsed, 2.0) * np.sign(direction[2])
        covariance[active, 0, 0] = 0.1**2
        covariance[active, 1, 1] = 0.1**2
        covariance[active, 2, 2] = 0.2**2
    elif scenario == "step_5m":
        unit_xy = direction[:2] / np.linalg.norm(direction[:2])
        bias[:, :2] = 5.0 * unit_xy
        bias[:, 2] = 1.0 * np.sign(direction[2])
    elif scenario == "spikes_8m":
        active[:] = False
        cursor = fault_start
        while cursor <= fault_end:
            index = int(np.argmin(np.abs(timestamps - cursor)))
            impulse_direction = _random_direction(rng)
            impulse_direction /= np.linalg.norm(impulse_direction)
            bias[index] = 8.0 * impulse_direction
            active[index] = True
            cursor += rng.uniform(10.0, 20.0)
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    bias[~active] = 0.0
    faulty = active & (np.linalg.norm(bias, axis=1) > 0)
    detectable = faulty & (
        np.linalg.norm(bias[:, :2], axis=1)
        >= config.detection_sigma * config.sigma_horizontal_m
    )
    return FaultRealization(
        timestamps,
        truth + noise + bias,
        covariance,
        bias,
        faulty,
        detectable,
        fault_start,
        fault_end,
        scenario,
        seed,
    )
