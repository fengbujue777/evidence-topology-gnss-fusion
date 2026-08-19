# Reproduction guide

## 1. Verify the release without datasets

```powershell
python examples/synthetic_topology_demo.py
python -m pytest -q
```

The synthetic demo verifies the separation-triangle branch, physical-source
mapping and Loewner dominance.  Unit tests cover the selector, source
provenance, covariance construction, covariance intersection and temporal
information-multiplicity solver.

## 2. Prepare public datasets

Follow `docs/DATASETS.md`.  Store processed cases below `data/processed/` or
use any external location.  Do not place reference arrays inside estimator
input files.

## 3. Run the proposed method

```powershell
python run_evidence_topology.py `
  --case-root <CASE_DIRECTORY> `
  --motion-margin 2.0 `
  --separation-margin 2.0 `
  --window-epochs 30 `
  --epoch-policy all_sources `
  --alignment-prefix-epochs 120 `
  --initial-prefix-epochs 60 `
  --output outputs/panel.json `
  --trajectory-output outputs/panel.npz
```

The defaults in `run_evidence_topology.py` are not a substitute for the
frozen command above.  The complete parameter record is in
`configs/frozen_parameters.json`.

## 4. Run principal baselines

Inspect each CLI before running:

```powershell
python run_ci_baseline.py --help
python run_contrast_factor.py --help
python topology_factor_graph.py --help
```

Every comparison must use identical KISS constraints, initialization,
timestamps, receiver covariance inputs and evaluation start index.

## 5. Statistics

```powershell
python analyze_paired_statistics.py --help
python analyze_ci_statistics.py --help
python analyze_leave_one_panel_out.py --help
```

The manuscript uses paired moving-block bootstrap with 10,000 resamples,
30-epoch blocks and Holm correction at family-wise alpha 0.05.

## 6. Frozen artifacts

`results/frozen_json/` contains the compact JSON outputs for four natural and
seven controlled-fault cases.  `results/paper_summaries/` contains the exact
CSV/JSON summaries used for tables, CI comparison, sensitivity, external
LOCSP replication and runtime reporting.  Large trajectory NPZ files and
rendered publication figures are omitted because they are derived outputs.

The runtime table measures the fusion layer only.  It excludes file I/O and
KISS-ICP registration because those costs are shared by every fusion method.
