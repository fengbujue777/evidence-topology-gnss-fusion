"""Run the complete UrbanNav experiment matrix from prepared case folders.

Raw-data extraction and case construction are separate, auditable stages
documented in ``docs/REPRODUCTION.md``.  This launcher covers the frozen
natural panels, controlled faults, CI/CU, ablations, calibration/topology
diagnostics, sensitivity, statistics, runtime repeats and numerical
verification against the committed artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parent
NATURAL = ("hk", "deep", "harsh", "phone")
FAULTS = (
    "novatel_step",
    "novatel_ramp",
    "ublox_f9p_step",
    "ublox_f9p_ramp",
    "ublox_m8t_step",
    "ublox_m8t_ramp",
    "distributed_novatel_step_f9p_ramp",
)
FROZEN_ARGS = (
    "--motion-margin", "2.0",
    "--separation-margin", "2.0",
    "--window-epochs", "30",
    "--epoch-policy", "all_sources",
    "--alignment-prefix-epochs", "120",
    "--initial-prefix-epochs", "60",
)
STAGE_ORDER = (
    "fault-cases",
    "natural",
    "faults",
    "ci",
    "cu",
    "ablations",
    "sensitivity",
    "statistics",
    "diagnostics",
    "runtime",
    "verify",
)


def _resolve(value: str) -> Path:
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", value)
    if os.name != "nt" and match:
        value = (
            f"/mnt/{match.group(1).lower()}/"
            f"{match.group(2).replace(chr(92), '/')}"
        )
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPOSITORY / path).resolve()


def _run_logged(
    command: list[str],
    outputs: list[Path],
    log_path: Path,
    overwrite: bool,
) -> None:
    existing = [path.exists() for path in outputs]
    if outputs and all(existing) and not overwrite:
        print(f"SKIP {outputs[0]}")
        return
    if any(existing) and not all(existing) and not overwrite:
        raise RuntimeError(
            f"partial output set exists for {outputs}; use a clean output root"
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(command))
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=REPOSITORY,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _run_estimator(
    case_root: Path,
    destination: Path,
    name: str,
    overwrite: bool,
    extra: tuple[str, ...] = (),
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    report = destination / f"{name}.json"
    trajectory = destination / f"{name}.npz"
    factor_inputs = destination / f"{name}.selected_factor_inputs.npz"
    _run_logged(
        [
            sys.executable,
            str(REPOSITORY / "run_evidence_topology.py"),
            *FROZEN_ARGS,
            *extra,
            "--case-root", str(case_root),
            "--output", str(report),
            "--trajectory-output", str(trajectory),
        ],
        [report, trajectory, factor_inputs],
        destination / f"{name}.log",
        overwrite,
    )


def _build_fault_cases(source: Path, root: Path, overwrite: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for receiver in ("novatel", "ublox_f9p", "ublox_m8t"):
        for mode in ("step", "ramp"):
            name = f"{receiver}_{mode}"
            destination = root / name / "cases"
            marker = destination / ".complete"
            _run_logged(
                [
                    sys.executable,
                    str(REPOSITORY / "build_receiver_fault_cases.py"),
                    "--source-root", str(source),
                    "--output-root", str(destination),
                    "--receiver-token", receiver,
                    "--mode", mode,
                ],
                [marker],
                root / name / "build.log",
                overwrite,
            )
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()

    intermediate = root / "distributed_intermediate"
    first_marker = intermediate / ".complete"
    _run_logged(
        [
            sys.executable,
            str(REPOSITORY / "build_receiver_fault_cases.py"),
            "--source-root", str(source),
            "--output-root", str(intermediate),
            "--receiver-token", "novatel",
            "--mode", "step",
        ],
        [first_marker],
        root / "distributed_novatel.log",
        overwrite,
    )
    first_marker.touch()
    final = root / "distributed_novatel_step_f9p_ramp" / "cases"
    second_marker = final / ".complete"
    _run_logged(
        [
            sys.executable,
            str(REPOSITORY / "build_receiver_fault_cases.py"),
            "--source-root", str(intermediate),
            "--output-root", str(final),
            "--receiver-token", "ublox_f9p",
            "--mode", "ramp",
        ],
        [second_marker],
        root / "distributed_f9p.log",
        overwrite,
    )
    second_marker.touch()


def _fault_roots(work_root: Path) -> dict[str, Path]:
    return {name: work_root / "fault_cases" / name / "cases" for name in FAULTS}


def _run_ci(
    cases: dict[str, Path], output_root: Path, overwrite: bool
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for name, case_root in cases.items():
        report = output_root / f"{name}.json"
        trajectory = output_root / f"{name}.npz"
        _run_logged(
            [
                sys.executable,
                str(REPOSITORY / "run_ci_baseline.py"),
                "--epoch-policy", "all_sources",
                "--case-root", str(case_root),
                "--output", str(report),
                "--trajectory-output", str(trajectory),
            ],
            [report, trajectory],
            output_root / f"{name}.log",
            overwrite,
        )


def _run_cu(
    cases: dict[str, Path], output_root: Path, overwrite: bool
) -> None:
    """Run CU with exactly the subsets selected by the proposed topology."""

    for name, case_root in cases.items():
        _run_estimator(
            case_root,
            output_root,
            name,
            overwrite,
            ("--pair-factor-mode", "covariance_union"),
        )


def _revision_diagnostics(
    output: Path, fault_case_root: Path, overwrite: bool
) -> None:
    destination = output / "revision_diagnostics"
    expected = destination / "revision_diagnostics.json"
    _run_logged(
        [
            sys.executable,
            str(REPOSITORY / "analyze_revision_diagnostics.py"),
            "--proposed-natural-root", str(output / "natural"),
            "--proposed-fault-root", str(output / "fault"),
            "--cu-root", str(output / "cu_same_subset"),
            "--fault-case-root", str(fault_case_root),
            "--output-root", str(destination),
        ],
        [expected],
        output / "revision_diagnostics.log",
        overwrite,
    )


def _statistics(output: Path, trials: int, overwrite: bool) -> None:
    jobs = (
        (
            "paired",
            "analyze_paired_statistics.py",
            [
                "--natural-root", str(output / "natural"),
                "--fault-root", str(output / "fault"),
                "--output-root", str(output / "statistics" / "paired"),
                "--trials", str(trials),
            ],
            output / "statistics" / "paired" / "report.json",
        ),
        (
            "ci",
            "analyze_ci_statistics.py",
            [
                "--natural-root", str(output / "natural"),
                "--fault-root", str(output / "fault"),
                "--ci-root", str(output / "ci"),
                "--output-root", str(output / "statistics" / "ci"),
                "--trials", str(trials),
            ],
            output / "statistics" / "ci" / "report.json",
        ),
        (
            "crossrun",
            "analyze_crossrun_statistics.py",
            [
                "--full-natural-root", str(output / "natural"),
                "--full-fault-root", str(output / "fault"),
                "--component-root", str(output / "component_ablation"),
                "--factor-root", str(output / "factor_ablation"),
                "--output-root", str(output / "statistics" / "crossrun"),
                "--trials", str(trials),
            ],
            output / "statistics" / "crossrun" / "report.json",
        ),
    )
    for label, script, arguments, expected in jobs:
        _run_logged(
            [sys.executable, str(REPOSITORY / script), *arguments],
            [expected],
            output / "statistics" / f"{label}.log",
            overwrite,
        )
    lopo = output / "statistics" / "leave_one_panel_out.json"
    _run_logged(
        [
            sys.executable,
            str(REPOSITORY / "analyze_leave_one_panel_out.py"),
            "--grid-root", str(output / "sensitivity"),
            "--output", str(lopo),
        ],
        [lopo, lopo.with_suffix(".csv")],
        output / "statistics" / "leave_one_panel_out.log",
        overwrite,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stages",
        default="all",
        help="Comma-separated stages or 'all'.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    natural_cases = {
        name: _resolve(config["natural_case_roots"][name]) for name in NATURAL
    }
    for name, path in natural_cases.items():
        if not path.is_dir():
            raise FileNotFoundError(f"missing natural case root {name}: {path}")
    work_root = _resolve(config.get("work_root", "data/processed/reproduction"))
    output = _resolve(config.get("output_root", "outputs/paper_reproduction"))
    trials = int(config.get("statistics_trials", 10_000))
    requested = (
        set(STAGE_ORDER)
        if args.stages == "all"
        else {item.strip() for item in args.stages.split(",") if item.strip()}
    )
    unknown = requested - set(STAGE_ORDER)
    if unknown:
        raise ValueError(f"unknown stages: {sorted(unknown)}")

    if "fault-cases" in requested:
        _build_fault_cases(natural_cases["hk"], work_root / "fault_cases", args.overwrite)
    faults = _fault_roots(work_root)

    if "natural" in requested:
        for name, root in natural_cases.items():
            _run_estimator(root, output / "natural", name, args.overwrite)
    if "faults" in requested:
        for name, root in faults.items():
            _run_estimator(root, output / "fault", name, args.overwrite)
    if "ci" in requested:
        _run_ci({**natural_cases, **faults}, output / "ci", args.overwrite)
    if "cu" in requested:
        _run_cu(
            {**natural_cases, **faults},
            output / "cu_same_subset",
            args.overwrite,
        )
    if "ablations" in requested:
        for policy in ("motion_single", "separation_pair"):
            for name, root in natural_cases.items():
                _run_estimator(
                    root,
                    output / "component_ablation" / policy / "natural",
                    name,
                    args.overwrite,
                    ("--subset-policy", policy),
                )
            for name, root in faults.items():
                _run_estimator(
                    root,
                    output / "component_ablation" / policy / "fault",
                    name,
                    args.overwrite,
                    ("--subset-policy", policy),
                )
        for scope, cases in (("natural", natural_cases), ("fault", faults)):
            for name, root in cases.items():
                _run_estimator(
                    root,
                    output / "factor_ablation" / scope,
                    name,
                    args.overwrite,
                    ("--pair-factor-mode", "independent"),
                )
    if "sensitivity" in requested:
        for margin in (1.25, 1.5, 2.0):
            for window in (20, 30, 45):
                setting = f"margin{margin}_window{window}"
                for name, root in natural_cases.items():
                    _run_estimator(
                        root,
                        output / "sensitivity" / setting,
                        name,
                        args.overwrite,
                        (
                            "--motion-margin", str(margin),
                            "--separation-margin", str(margin),
                            "--window-epochs", str(window),
                        ),
                    )
    if "statistics" in requested:
        _statistics(output, trials, args.overwrite)
    if "diagnostics" in requested:
        _revision_diagnostics(output, work_root / "fault_cases", args.overwrite)
    if "runtime" in requested:
        repeats = output / "runtime" / "repeats"
        for index in range(1, 6):
            _run_estimator(
                natural_cases["hk"],
                repeats,
                f"repeat_{index}",
                args.overwrite,
            )
        summary = output / "runtime" / "summary" / "runtime_benchmark.json"
        _run_logged(
            [
                sys.executable,
                str(REPOSITORY / "summarize_runtime.py"),
                "--repeat-root", str(repeats),
                "--output-root", str(output / "runtime" / "summary"),
            ],
            [summary],
            output / "runtime" / "summary.log",
            args.overwrite,
        )
    if "verify" in requested:
        verification = output / "verification.json"
        _run_logged(
            [
                sys.executable,
                str(REPOSITORY / "verify_reproduction.py"),
                "--actual-root", str(output),
                "--output", str(verification),
            ],
            [verification],
            output / "verification.log",
            args.overwrite,
        )


if __name__ == "__main__":
    main()
