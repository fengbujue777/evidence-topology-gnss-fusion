"""End-to-end LOCSP CR1/CR2 source-quotient external validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def _run(arguments: list[str]) -> None:
    print("+", " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=ROOT, check=True)


def _graph(case_root: Path, output_root: Path, ablation: str, partial: bool) -> None:
    arguments = [
        sys.executable,
        "topology_factor_graph.py",
        "--case-root",
        str(case_root),
        "--output",
        str(output_root.with_suffix(".json")),
        "--trajectory-output",
        str(output_root.with_suffix(".npz")),
        "--epoch-policy",
        "all_streams",
        "--ablation",
        ablation,
    ]
    if partial:
        arguments.extend(
            [
                "--reference-max-gap-s",
                "0.5",
                "--reference-nearest-tolerance-s",
                "0.2",
            ]
        )
    _run(arguments)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--dataset-id", choices=("locsp_cr1", "locsp_cr2"), default="locsp_cr2")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--tf-json",
        type=Path,
        default=ROOT / "configs" / "extrinsics" / "locsp_tf_static.json",
    )
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_root = args.output_root.resolve()
    if args.output_root == ROOT or args.output_root in ROOT.parents:
        raise ValueError(
            "--output-root must be a dedicated generated-output directory, "
            "not the repository or one of its parents"
        )

    if args.output_root.exists() and any(args.output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_root} is not empty; use --overwrite to replace generated files"
            )
        shutil.rmtree(args.output_root)
    cases = args.output_root / "cases"
    cache = args.output_root / "cache"
    quotient = args.output_root / "source_quotient_cases"
    results = args.output_root / "results"
    cases.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    dataset = f"{args.dataset_id}_hesai"

    _run(
        [
            sys.executable,
            "extract_locsp_cache.py",
            "--dataset-id",
            dataset,
            "--bag",
            str(args.bag),
            "--tf-json",
            str(args.tf_json),
            "--case-root",
            str(cases),
            "--cache-root",
            str(cache),
            "--lidar-topic",
            "/hesai/pandar",
        ]
    )
    audit = results / "m8t_stream_audit.json"
    _run(
        [
            sys.executable,
            "audit_locsp_m8t_solution_streams.py",
            str(args.bag),
            "--output",
            str(audit),
        ]
    )

    quotient.mkdir(parents=True, exist_ok=True)
    for path in cases.glob("*.npz"):
        shutil.copy2(path, quotient / path.name)
    m8t = quotient / f"{dataset}__locsp_m8t__natural.estimator_input.npz"
    navpvt = quotient / f"{dataset}__locsp_m8t_navpvt__natural.estimator_input.npz"
    _run(
        [
            sys.executable,
            "build_locsp_m8t_navpvt_estimator_input.py",
            str(args.bag),
            str(m8t),
            str(navpvt),
        ]
    )

    partial = args.dataset_id == "locsp_cr1"
    _graph(cases, results / "two_source", "none", partial)
    _graph(quotient, results / "source_aware", "none", partial)
    _graph(quotient, results / "naive_streams", "no_dependency_grouping", partial)
    final = results / "final"
    _run(
        [
            sys.executable,
            "finalize_locsp_source_quotient.py",
            "--two-source-report",
            str(results / "two_source.json"),
            "--two-source-trajectory",
            str(results / "two_source.npz"),
            "--source-aware-report",
            str(results / "source_aware.json"),
            "--source-aware-trajectory",
            str(results / "source_aware.npz"),
            "--naive-report",
            str(results / "naive_streams.json"),
            "--naive-trajectory",
            str(results / "naive_streams.npz"),
            "--stream-audit",
            str(audit),
            "--output-root",
            str(final),
            "--dataset-label",
            args.dataset_id.upper().replace("_", " "),
            "--trials",
            str(args.trials),
        ]
    )
    print(json.dumps({"status": "complete", "report": str(final / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
