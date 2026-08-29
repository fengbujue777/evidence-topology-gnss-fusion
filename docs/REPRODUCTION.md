# Reproduction guide

This guide covers the complete path from public raw data to the paper tables.
Commands use PowerShell line continuation; Linux/WSL users can replace the
backticks with backslashes.

## 1. Create the environment

Python 3.12 is the frozen interpreter version.

```powershell
conda env create -f environment.yml
conda activate evidence-topology-gnss-fusion
```

Alternatively create a Python 3.12 virtual environment and install
`requirements.txt`.

Build the exact RTKLIB revision used for SPP:

```bash
bash scripts/install_rtklib.sh
export RTKLIB_RNX2RTKP="$PWD/.tools/RTKLIB/app/consapp/rnx2rtkp/gcc/rnx2rtkp"
```

The frozen RTKLIB commit is
`180043ee24b6d2b168f98b64be15f69d50046b1a`; the command records it and input
SHA-256 hashes in `spp_manifest.json`.

## 2. Verify code without a dataset

```powershell
python -m pytest -q
python examples/synthetic_topology_demo.py
```

## 3. Build one UrbanNav panel from raw data

Download the files listed in `docs/DATASETS.md`. Medium can then be prepared
with one command:

```powershell
python prepare_urbannav_panel.py `
  --panel configs/receiver_panels/urbannav_hk_medium.json `
  --bag <MEDIUM_ROS_BAG> `
  --rinex-root <MEDIUM_RINEX_DIRECTORY> `
  --extrinsics-json configs/extrinsics/urbannav_hk_medium.json `
  --cache-root data/processed/hk_medium/cache `
  --solution-root data/processed/hk_medium/solutions `
  --case-root data/processed/hk_medium/cases
```

For Deep, change the panel/extrinsic/output paths and provide every bag after
`--bag`. For Harsh, use the verified partial bag and add
`--truth-csv <OFFICIAL_HARSH_PARTIAL_TRUTH>`.

The three-phone panel reuses the Harsh KISS/truth cache:

```powershell
python prepare_urbannav_panel.py `
  --panel configs/receiver_panels/urbannav_three_phones.json `
  --rinex-root <PHONE_RINEX_DIRECTORY> `
  --cache-root data/processed/hk_harsh/cache `
  --solution-root data/processed/three_phones/solutions `
  --case-root data/processed/three_phones/cases
```

The phone command performs the registered half-second timestamp
canonicalization and records its hashes. Samsung Note8 is processed for the
audit but excluded from the frozen three-phone panel by configuration.

If KISS/truth caches already exist, omit `--bag`; this is useful for rerunning
SPP or case construction without decoding LiDAR again.

## 4. Run the complete UrbanNav paper matrix

Copy `configs/paper_reproduction.example.json`, replace its four case roots,
and keep generated outputs under ignored directories. Then run:

```powershell
python reproduce_paper.py `
  --config configs/paper_reproduction.local.json `
  --stages all
```

The launcher performs, in dependency order:

1. seven deterministic controlled-fault case builds;
2. four natural and seven fault evaluations;
3. covariance-intersection and same-selected-subset covariance-union baselines;
4. selector-component and factor-topology ablations;
5. the 3-by-3 parameter grid;
6. 10,000-replicate paired block-bootstrap, CI and cross-run statistics;
7. leave-one-route-out parameter selection;
8. five fusion-layer runtime repeats;
9. factor-level coverage, CU and direct topology-decision diagnostics;
10. separation-first versus motion-first priority ablation;
11. block-length sensitivity at 15, 30, 45 and 60 epochs;
12. numerical comparison with committed frozen summaries.

Run only selected stages with, for example,
`--stages natural,fault-cases,faults,cu,diagnostics,verify`. Existing complete outputs are
not overwritten unless `--overwrite` is given.

Verification requires exact epoch counts. Because nonlinear solvers can take
slightly different floating-point stopping paths across operating systems and
CPU libraries, RMSE bounds are explicitly fixed at 3 cm for the proposed
method and 15 cm for iterative baselines. In the release audit, the largest
observed deviations were 2.32 cm and 10.39 cm, respectively.

## 5. Reproduce the LOCSP external test

```powershell
python reproduce_locsp.py `
  --dataset-id locsp_cr2 `
  --bag <LOCSP_CR2_ROS_BAG> `
  --output-root outputs/locsp_cr2
```

The command extracts KISS-ICP and receiver streams, audits the duplicated M8T
encoding, runs two-source/source-aware/naive graphs, performs a 10,000-sample
paired moving-block bootstrap and verifies replay invariance to `1e-9 m`.
Use `--dataset-id locsp_cr1` for CR1; its partial-reference mask is applied
automatically.

## 6. Output-to-paper mapping

- `natural/*.json`, `fault/*.json`: principal RMSE/P95 tables;
- `ci/`: covariance-intersection comparison;
- `cu_same_subset/`: CU using exactly the proposed selected subsets;
- `component_ablation/`, `factor_ablation/`: ablation table;
- `component_ablation/motion_first/` and
  `statistics/priority_ablation/`: separation-first versus motion-first
  paired rerun and bootstrap comparison;
- `sensitivity/`, `statistics/leave_one_panel_out.*`: parameter analysis;
- `statistics/`: paired confidence intervals and corrected tests;
- `statistics/block_sensitivity/B15/`, `B30/`, `B45/`, and `B60/`:
  pre-defined block-length sensitivity runs;
- `revision_diagnostics/`: pair/singleton-stratified factor coverage,
  pre/post-fault topology decisions, same-subset envelope--CU paired
  intervals, pair-factor runtime/feasibility, and route-balanced descriptive
  aggregation;
- `runtime/summary/runtime_benchmark.json`: fusion-layer timing;
- `verification.json`: final pass/fail gate.

Committed small summaries live in `results/`; raw and generated NPZ files do
not need to be versioned because every stage above regenerates them.

After the natural and synchronized-fault stages have finished, regenerate the
three representative trajectory panels directly from their NPZ/JSON pairs:

```powershell
python plot_qualitative_trajectories.py `
  --natural-root <REPRODUCTION_OUTPUT>/natural `
  --fault-root <REPRODUCTION_OUTPUT>/fault `
  --output-stem <FIGURE_DIRECTORY>/figure2_qualitative_trajectories
```

The script reads `evaluation_start_index` from each result JSON and recomputes
every displayed RMSE from the plotted trajectory arrays.  No RMSE annotation is
stored as a hand-entered constant.
