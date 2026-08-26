# Clean-room reproduction audit

## Verdict

**PASS (2026-08-25/26).** The registered paper pipeline was rerun from a fresh clone of base revision `1cc87ca6f47d77c48a8d146c1fe12d627891fbac`, a new Python environment, RTKLIB revision `180043ee24b6d2b168f98b64be15f69d50046b1a`, and newly downloaded official UrbanNav and LOCSP provider files.

This audit checks data provenance, processing integrity, and numerical reproducibility. It is not a claim of universal generalization or of winning every comparison.

## Reproduction gates

- Registered natural/fault cases: **11/11 passed**.
- Evaluation-epoch counts: **exactly matched in every case**.
- Largest proposed-method 3-D RMSE deviation across platforms: **0.023123 m**, below the declared **0.03 m** tolerance.
- Largest iterative-baseline 3-D RMSE deviation: **0.103865 m**, below the declared **0.15 m** tolerance.
- Unit and release-contract tests: **19/19 passed** with `python -m pytest -q`.
- Dependency check: **passed**.
- `git diff --check`: **passed**.

The nonzero numerical tolerances accommodate different floating-point stopping paths in GTSAM and iterative robust solvers. They never relax epoch counts, case registration, or dataset hashes.

## Independent physical-source replication

The LOCSP experiment was rerun from official CR1 and CR2 ROS bags. Replacing a duplicated encoding of the same M8T solution preserved the trajectory below `5e-12` m. Counting that encoding as an independent receiver increased 3-D RMSE:

| Route | Source quotient (m) | Naive stream count (m) | Reduction | Paired 95% CI (m) |
|---|---:|---:|---:|---:|
| CR1 | 8.018547 | 11.657482 | 31.22% | [-5.429702, -1.722789] |
| CR2 | 6.055721 | 7.713238 | 21.49% | [-2.210959, -0.978134] |

## Negative and tied results retained

The release intentionally retains cases where the proposed method is not the lowest-error estimator, including UrbanNav Deep, the phone panel, and the distributed double fault. The supported claim is targeted robustness to evidence multiplicity and receiver disagreement, not universal RMSE dominance.

See `results/reproduction_audit/official_input_hashes.json` and `results/reproduction_audit/verification_summary.json` for machine-readable audit records. The complete pipeline remains `python reproduce_paper.py --config <local-config.json> --stage all`; provider files are not redistributed.
