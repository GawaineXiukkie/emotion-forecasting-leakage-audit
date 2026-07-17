# IEEE Access paper-to-repository map

This map corresponds to `paper/access_submission/manuscript/main.tex`. Historical fixed-budget
results remain in the repository but are not evidence for the revised manuscript.

| Paper evidence | Producer | Reviewable output |
|---|---|---|
| Table 1 threat model | protocol implementation | `docs/access_experiment_protocol.md`, `src/leakage_audit.py` |
| Table 2 corpus/F1 statistics | dataset loaders | `results/dataset_stats.md`, `docs/label_mapping.md` |
| Tables 3--5 speaker/target composition | `python -m src.access_speaker_analysis` | `results/access_speaker_analysis.{json,md}` |
| Table 6 full self-shift benchmark | `python -m src.access_self_shift_benchmark` | `results/access_self_shift_benchmark.{json,md}`, `results/cache_access_self_shift/` |
| Tables 7--8 information-matched benchmark | `python -m src.access_revision_experiments` | `results/access_revision_experiments.{json,md}`, `results/cache_access_revision/` |
| Table 9 independent local features | `python -m src.access_local_feature_control` | `results/access_local_feature_control.{json,md}` |
| Table 10 duplicate audit | `python -m src.access_data_quality` | `results/access_data_quality.{json,md}` |
| Figure 1 speaker relation | `python -m src.access_speaker_figure` | `results/figures/access_speaker_relation.png` |
| Figure 2 uniform leakage dose | `python -m src.access_uniform_dose_response` | `results/access_uniform_dose_response.{json,md}`, `results/figures/access_uniform_dose_response.png` |
| State-factorized control | `python -m src.access_state_factorized` | `results/access_state_factorized.{json,md}` |
| Structural causality | `python -m src.check_causality` | `results/causality_check.md` |

All final seed runs store compressed raw scores, targets, and dialogue IDs. Main inference uses a
seed-by-dialogue hierarchical bootstrap, paired dialogue-cluster permutation, plus-one p-values,
and the declared Holm family. Environment and data integrity are recorded in
`requirements-lock.txt`, `environment.yml`, `docs/environment.md`, and
`docs/data_manifest.sha256`.

The authoritative paper source/PDF is under `paper/access_submission/manuscript/`. Compile with
Tectonic 0.16.9 using `tectonic -X compile main.tex --keep-logs --keep-intermediates`.
