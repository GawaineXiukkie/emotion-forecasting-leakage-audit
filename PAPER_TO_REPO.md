# Paper → repository map

Every number and claim in the paper, matched to the file it comes from and the script that
regenerates it. If a number in the paper doesn't trace to a line below, that's a bug — open an
issue.

| Paper item | Repo file(s) | Regenerate with |
|---|---|---|
| Table I (dataset stats + F1 degeneracy: 151/7,433/0.429 etc.) | [`results/dataset_stats.md`](results/dataset_stats.md) | dialogue/utterance counts via `src.dataset.load_cosmic` / `load_mmdfn`; shift rate and transition-BA from `results/benchmark_table.md`, produced by `python -m src.experiments` |
| Table II (6 models × 6 configs, shift-AUC main table) | [`results/benchmark_table.md`](results/benchmark_table.md) | `python -m src.experiments` |
| Table III (MELD feature-source ablation: 0.554/0.724/0.722/0.705/0.712) | [`results/feature_ablation_meld.md`](results/feature_ablation_meld.md) | `python -m src.feature_ablation` |
| Fig. 1 (leakage dose-response, k ∈ {0,1,2,4,∞}) | [`results/dose_response.md`](results/dose_response.md), [`results/dose_response.json`](results/dose_response.json), [`results/figures/leakage_dose_response.png`](results/figures/leakage_dose_response.png), [`paper/figures/leakage_dose_response.pdf`](paper/figures/leakage_dose_response.pdf) | `python -m src.dose_response` |
| 36-row joint Holm-Bonferroni correction table (10/36 survive) | [`results/significance_tests.md`](results/significance_tests.md) | `python -m src.significance_tests` |
| Causality self-check (perturb t≥n → outputs t<n unchanged, exactly 0.0) | [`results/causality_check.md`](results/causality_check.md), [`src/check_causality.py`](src/check_causality.py) | `python -m src.check_causality` |
| Capacity-matched control (0.105 → 0.092*) | [`results/capacity_control.md`](results/capacity_control.md), [`results/capacity_control.json`](results/capacity_control.json) | `python -m src.capacity_control` |
| DailyDialog dedup robustness (13.4% dup, +0.045 → +0.038) | [`docs/robustness.md`](docs/robustness.md), [`results/dedup_dailydialog.json`](results/dedup_dailydialog.json), [`results/duplicate_audit.md`](results/duplicate_audit.md) | `python -m src.check_duplicate_dialogues` |
| 8-item leakage audit (per dataset) | [`results/leakage_audit.md`](results/leakage_audit.md) | `python -m src.experiments` (calls `write_audit()` in `src/leakage_audit.py`) |
| EFC-inspired strategy baselines (PEC-style, Pseudo-future-style) | [`results/efc_baselines.json`](results/efc_baselines.json), [`src/models_efc_baselines.py`](src/models_efc_baselines.py), [`src/efc_baselines.py`](src/efc_baselines.py) | `python -m src.efc_baselines` |
| Causal re-implementations (DialogueRNN/DialogueGCN/DAG-ERC) | [`docs/causal_reimplementations.md`](docs/causal_reimplementations.md), [`src/models_families.py`](src/models_families.py) | — (architecture description, not a generated result) |

`*` The paper text says "0.105 → 0.093"; the exact repo computation over the four per-dataset
gaps in `results/capacity_control.json` gives a mean of 0.0925, which rounds to 0.092, not
0.093. This is a ~0.001 rounding slip in the submitted paper text, not a repo error — the repo
number here is the correct one for anyone trying to independently recompute the figure.

## Full reproduction

See the [README](README.md) for the environment setup and the exact command sequence, from raw
feature files to every table above.
