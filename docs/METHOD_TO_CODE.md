# Manuscript-to-code map

| Manuscript component | Main implementation |
|---|---|
| Physical-source quotient and exact replay removal | `topology_factor_graph.py::_deduplicate_exact_measurements`, `paper_pipeline/provenance.py` |
| Truth-free KISS-to-ENU alignment | `paper_pipeline/alignment.py` |
| KISS-conditioned motion margin | `run_evidence_topology.py::_candidate_subset` and window diagnostics in `topology_factor_graph.py` |
| Receiver-separation margin | `run_evidence_topology.py::_candidate_subset` |
| Margin-constrained singleton/pair rule | `run_evidence_topology.py::_candidate_subset` |
| Separation-first versus motion-first priority audit | `run_evidence_topology.py --subset-policy {full,motion_first}`, `analyze_priority_ablation.py` |
| Pair center, undivided scatter and Loewner loading | `paper_pipeline/loewner_envelope.py::provenance_loewner_envelope` |
| One factor versus two-factor topology ablation | `run_evidence_topology.py --pair-factor-mode` |
| Factor graph and robust baselines | `topology_factor_graph.py`, `fusion_graph_backend.py` |
| Global covariance intersection | `run_ci_baseline.py` |
| Information-multiplicity validation | `validate_information_multiplicity.py` |
| Moving-block bootstrap and Holm correction | `analyze_paired_statistics.py`, `analyze_ci_statistics.py` |
| Bootstrap block-length sensitivity | `analyze_paired_statistics.py --block {15,30,45,60}` |
| Leave-one-panel-out parameter audit | `analyze_leave_one_panel_out.py` |

The reference trajectory is not an argument to subset selection, provenance
mapping, covariance construction or graph optimization.  It is accessed by
the evaluation boundary after estimator output has been generated.
