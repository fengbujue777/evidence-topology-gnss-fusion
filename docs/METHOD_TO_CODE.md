# Manuscript-to-code map

| Manuscript component | Main implementation |
|---|---|
| Physical-source quotient and exact replay removal | `evidence_topology/topology_factor_graph.py::_deduplicate_exact_measurements`, `paper_pipeline/provenance.py` |
| Truth-free KISS-to-ENU alignment | `paper_pipeline/alignment.py` |
| KISS-conditioned motion margin | `evidence_topology/method.py::_candidate_subset` and window diagnostics in `evidence_topology/topology_factor_graph.py` |
| Receiver-separation margin | `evidence_topology/method.py::_candidate_subset` |
| Margin-constrained singleton/pair rule | `evidence_topology/method.py::_candidate_subset` |
| Separation-first versus motion-first priority audit | `python -m evidence_topology.method --subset-policy {full,motion_first}`, `scripts/analysis/analyze_priority_ablation.py` |
| Pair center, undivided scatter and Loewner loading | `paper_pipeline/loewner_envelope.py::provenance_loewner_envelope` |
| One factor versus two-factor topology ablation | `python -m evidence_topology.method --pair-factor-mode` |
| Factor graph and robust baselines | `evidence_topology/topology_factor_graph.py`, `evidence_topology/fusion_graph_backend.py` |
| Global covariance intersection | `scripts/experiments/run_ci_baseline.py` |
| Information-multiplicity validation | `scripts/experiments/validate_information_multiplicity.py` |
| Moving-block bootstrap and Holm correction | `scripts/analysis/analyze_paired_statistics.py`, `scripts/analysis/analyze_ci_statistics.py` |
| Bootstrap block-length sensitivity | `python -m scripts.analysis.analyze_paired_statistics --block {15,30,45,60}` |
| Leave-one-panel-out parameter audit | `scripts/analysis/analyze_leave_one_panel_out.py` |

The reference trajectory is not an argument to subset selection, provenance
mapping, covariance construction or graph optimization.  It is accessed by
the evaluation boundary after estimator output has been generated.
