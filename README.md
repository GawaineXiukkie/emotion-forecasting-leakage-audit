# Information-Matched Auditing of Conversational Emotion-Shift Forecasting

Bin Wen, Dai-Qiao Zhang, and Tien-Ping Tan, Universiti Sains Malaysia.

This repository contains the executable audit and IEEE Access manuscript for leakage-safe
conversational emotion-shift forecasting. The task is to predict whether the next utterance will
express a different emotion using only information available after the current turn and before the
target turn begins.

The revision distinguishes two comparisons that should not be conflated:

- **Deployable:** learned models receive causal utterance features; the transition baseline uses a
  current-emotion label predicted by a train-only classifier.
- **Oracle diagnostic:** both learned models and the transition baseline receive gold current
  emotion. This measures forecasting value conditional on perfectly known current state; it is not
  a deployment claim.

Every learned family receives the same four-configuration validation search, validation-selected
checkpoint, and three training seeds. Inference jointly represents seed variability and clustered
test dialogues. DailyDialog results remove validation/test dialogues exactly duplicated in train.

## Main result

In the deployable comparison, all six causal models improve in point estimate over the
predicted-label transition baseline on all four primary text corpora. DailyDialog gains are
`+0.160` to `+0.167` ROC-AUC and survive the joint 72-comparison Holm correction; DialogueRNN's
smaller MELD gain also survives. In the information-matched oracle diagnostic, the gold-label
transition matrix remains competitive on IEMOCAP and is stronger on MELD, while all six learned
models retain large, corrected-significant DailyDialog gains.

The result is therefore conditional, not a universal claim that either architecture or inertia
always wins. Current-label availability, corpus construction, target definition, feature
provenance, and leakage discipline materially change the conclusion.

See `results/access_revision_experiments.md` for the complete comparison and
`docs/access_experiment_protocol.md` for the predeclared analysis rule.

## Environment

The recorded environment is macOS arm64, Python 3.9.6, PyTorch 2.8.0 with MPS, NumPy 2.0.2,
SciPy 1.13.1, and scikit-learn 1.6.1. Exact package versions are in `requirements-lock.txt` and
`environment.yml`; the full machine-readable description is in `docs/environment.md`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

Feature files are not redistributed. Expected filenames and SHA-256 hashes are recorded in
`docs/data_manifest.sha256`; provenance and locality boundaries are documented in
`docs/feature_provenance.md`.

## Reproduction

Run the primary revision first; the self-shift benchmark reuses its selected configurations.

```bash
python -m src.access_data_quality
python -m src.check_causality
python -m unittest discover -s tests -v
python -m src.access_revision_experiments
python -m src.access_state_factorized
python -m src.access_local_feature_control
python -m src.access_self_shift_benchmark
python -m src.access_uniform_dose_response
```

Long runs are resumable from isolated namespaces under `results/cache_access_*`. Search runs do
not evaluate the test fold. Every final seed stores raw scores, targets, and dialogue identifiers
in compressed NumPy files for paired re-analysis.

## Data

The primary text features are the released COSMIC RoBERTa pickles from
[declare-lab/conv-emotion](https://github.com/declare-lab/conv-emotion). The producer scripts for
IEMOCAP, MELD, and EmoryNLP were audited and operate per utterance. A separate control fits
TF-IDF vocabulary/IDF and truncated SVD only on training utterances, then transforms each
utterance independently.

MM-DFN IEMOCAP/MELD features are retained only as sensitivity configurations because the release
does not contain complete feature-producer code. They are not treated as independent primary
corpora or proof of causal upstream extraction.

Speaker roles in MELD, EmoryNLP, and DailyDialog are dialogue-local and are qualified by dialogue
ID before transition estimation. IEMOCAP retains its session-and-actor identities. Emotion label
mappings are listed in `docs/label_mapping.md`.

## Protocol summary

- Explicit commitment point and target-absent feature construction.
- Future-perturbation tests for every causal model implementation.
- Equal four-trial validation search and early-stopped checkpoint selection.
- ROC-AUC primary; PR-AUC, Brier, ECE, F1, precision, recall, and balanced accuracy retained.
- Seed-by-dialogue hierarchical bootstrap with 1,999 replicates.
- Paired dialogue-cluster permutation with 4,999 sign flips and plus-one p-values.
- Holm-Bonferroni correction across the declared deployable and oracle comparison family.
- Complete next-own-utterance self-shift benchmark, not only a prevalence diagnostic.
- Same-architecture Transformer leakage dose response; only the future-attention mask changes.

## Paper-to-artifact map

| Evidence | Producer | Reviewable output |
|---|---|---|
| Information-matched six-model benchmark | `src/access_revision_experiments.py` | `results/access_revision_experiments.{json,md}` |
| Data/split/alignment audit | `src/access_data_quality.py` | `results/access_data_quality.{json,md}` |
| Structural causality | `src/check_causality.py` | `results/causality_check.md` |
| State-factorized control | `src/access_state_factorized.py` | `results/access_state_factorized.{json,md}` |
| Independent local features | `src/access_local_feature_control.py` | `results/access_local_feature_control.{json,md}` |
| Full self-shift benchmark | `src/access_self_shift_benchmark.py` | `results/access_self_shift_benchmark.{json,md}` |
| Uniform leakage dose | `src/access_uniform_dose_response.py` | `results/access_uniform_dose_response.{json,md}` |
| Manuscript source | — | `paper/access_submission/manuscript/` |

## Repository status

The IEEE Access manuscript is in preparation and is not under consideration at ICASSP. A public
release/tag must be created from the final reviewed commit before the manuscript can claim an
immutable public artifact; no tag or DOI is asserted in advance.

## Citation

```bibtex
@article{wen2026informationmatched,
  title   = {Information-Matched Auditing of Conversational Emotion-Shift Forecasting},
  author  = {Wen, Bin and Zhang, Dai-Qiao and Tan, Tien-Ping},
  journal = {IEEE Access},
  note    = {Manuscript in preparation},
  year    = {2026}
}
```

## License

MIT; see `LICENSE`.
