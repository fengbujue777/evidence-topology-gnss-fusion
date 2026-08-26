# Evidence Topology Reconstruction for Multi-Receiver GNSS Fusion

Research code and frozen evaluation artifacts for **Evidence Topology
Reconstruction and Conservative Factor Construction for Conflicting
Multi-Receiver GNSS Fusion**.

The repository addresses a problem that conventional robust weighting does
not represent: several registered solution streams need not be independent
pieces of physical evidence.  The implementation therefore decides evidence
cardinality before GNSS factors enter the graph.

## Method at a glance

1. Quotient solution streams by physical receiver provenance and remove exact
   replays.
2. Align KISS-ICP and GNSS without using the reference trajectory.
3. Compute a KISS-conditioned motion margin and a receiver-separation margin
   over non-overlapping windows.
4. Select a singleton or coherent receiver pair with frozen thresholds.
5. Insert one conservative Loewner-envelope factor for a selected pair.

The nominal parameters are `W=30`, `tau_m=tau_d=2.0`, covariance floor
`1.0 m`, and fixed Cauchy scale `2.5`.  They are recorded in
[`configs/frozen_parameters.json`](configs/frozen_parameters.json).

## Repository layout

```text
run_evidence_topology.py       main proposed estimator
topology_factor_graph.py       graph construction and evaluation boundary
paper_pipeline/                alignment, cases, provenance and covariance code
run_ci_baseline.py             covariance-intersection baseline
run_contrast_factor.py         independent-factor topology ablation
analyze_*.py                   statistics and leave-one-panel-out analysis
build_*.py                     controlled-fault and stress-case builders
tests/                         algorithmic regression tests
results/                       frozen JSON/CSV results used by the paper
docs/                          dataset and reproduction instructions
```

Legacy development filenames were renamed in this release, but estimator
logic and frozen parameters were retained.  The covariance center is written
explicitly as an arithmetic mean, matching the manuscript; for the evaluated
singleton/pair domain it is numerically identical to the earlier two-sample
median implementation.

## Installation

The experiments were run with Python 3.12 on Windows 11/WSL2.  A GPU is not
required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux or WSL, activate with `source .venv/bin/activate`.

For an exact Conda environment, use `conda env create -f environment.yml`.
RTKLIB is pinned and built by `scripts/install_rtklib.sh`.

## Quick verification

The synthetic check needs no downloaded dataset:

```powershell
python examples/synthetic_topology_demo.py
python -m pytest -q
```

## End-to-end reproduction

The release includes the full public-data preprocessing chain:

```text
ROS bags + RINEX
  -> KISS-ICP extraction + pinned RTKLIB SPP
  -> truth-isolated receiver cases
  -> proposed/baseline/ablation runs
  -> statistics, runtime and numerical verification
```

Prepare one panel with `prepare_urbannav_panel.py`; run the complete paper
matrix with `reproduce_paper.py`; reproduce the independent LOCSP source-
quotient experiment with `reproduce_locsp.py`. Exact commands are in
[`docs/REPRODUCTION.md`](docs/REPRODUCTION.md).

To run one already-prepared panel directly:

```powershell
python run_evidence_topology.py `
  --case-root data/processed/hk_medium/cases `
  --motion-margin 2.0 `
  --separation-margin 2.0 `
  --window-epochs 30 `
  --epoch-policy all_sources `
  --alignment-prefix-epochs 120 `
  --initial-prefix-epochs 60 `
  --output outputs/hk_medium.json `
  --trajectory-output outputs/hk_medium.npz
```

For the complete matrix, copy and edit
`configs/paper_reproduction.example.json` and run:

```powershell
python reproduce_paper.py --config configs/paper_reproduction.local.json --stages all
```

Full instructions and the correspondence between manuscript experiments and
scripts are in [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) and
[`docs/METHOD_TO_CODE.md`](docs/METHOD_TO_CODE.md).

## Data and frozen results

Raw UrbanNav and LOCSP files are not redistributed. They remain governed by
their original providers. This repository includes the readers, pinned SPP
configuration, extrinsics, panel definitions and orchestration needed to
regenerate cases, plus small frozen JSON/CSV summaries. Reference trajectories
are loaded only after candidate trajectories have been produced.

An independent clean-room rerun from freshly downloaded official files passed
all 11 registered cases and all 19 tests. See the
[clean-room audit](docs/CLEANROOM_REPRODUCTION_AUDIT.md) and its machine-readable
records under `results/reproduction_audit/`.

## Reproducibility scope

The released estimator is the offline route-level realization evaluated in
the manuscript.  It is not a causal fixed-lag deployment and does not claim a
formal integrity protection level.  Common-mode receiver faults and
incompatible coordinate/reference sequences remain disclosed failure cases.

## License and citation

The repository is private while the manuscript is being prepared.  A source
code license and final citation metadata will be added only after the authors
and target venue approve public release.
