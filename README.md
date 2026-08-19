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

## Quick verification

The synthetic check needs no downloaded dataset:

```powershell
python examples/synthetic_topology_demo.py
python -m pytest -q
```

## Running one real panel

After preparing the truth-separated case files described in
[`docs/DATASETS.md`](docs/DATASETS.md):

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

For several panels, edit `configs/paper_panels.example.json` and run:

```powershell
python run_paper_suite.py --config configs/paper_panels.example.json
```

Full instructions and the correspondence between manuscript experiments and
scripts are in [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) and
[`docs/METHOD_TO_CODE.md`](docs/METHOD_TO_CODE.md).

## Data and frozen results

Raw UrbanNav and LOCSP files are not redistributed.  They remain governed by
their original providers.  This repository includes only code, small frozen
JSON/CSV summaries, and runtime repeats required to verify reported numbers. Reference
trajectories are loaded only after candidate trajectories have been produced.

## Reproducibility scope

The released estimator is the offline route-level realization evaluated in
the manuscript.  It is not a causal fixed-lag deployment and does not claim a
formal integrity protection level.  Common-mode receiver faults and
incompatible coordinate/reference sequences remain disclosed failure cases.

## License and citation

The repository is private while the manuscript is being prepared.  A source
code license and final citation metadata will be added only after the authors
and target venue approve public release.
