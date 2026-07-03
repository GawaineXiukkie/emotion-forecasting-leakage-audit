# Inertia, Leakage, and the Illusion of Progress in Conversational Emotion Forecasting

Bin Wen, School of Computer Sciences, Universiti Sains Malaysia. Submitted to ICASSP 2027.
Full paper: [`paper/main.pdf`](paper/main.pdf).

This is a measurement paper, not a new method. We re-evaluate emotion-shift forecasting —
predicting whether the next utterance will carry a different emotion than the current one,
using only context up to the decision point, with the target utterance itself never seen —
under a protocol that enforces that constraint structurally, with a strong train-fold-only
speaker×emotion transition-matrix baseline. The question is how much of the field's apparent
progress survives once leakage is actually closed off. Runs entirely on a Mac; no GPU needed.

## Result

Six causal models — GRU and causal re-implementations of DialogueRNN, DialogueGCN, and
DAG-ERC, plus two emotion-forecasting-specific strategy baselines (PEC-style,
Pseudo-future-style) — land within about 0.03 AUC of each other on every one of six dataset
configurations. On four of six, every safe model loses to the transition matrix and the only
way to "win" is to let the model see the future; on one multimodal configuration, even that
isn't enough. Where safe models do win, the evidence points to feature source rather than
architecture, though we report that as a hypothesis rather than a settled conclusion. A
dose-response sweep separates two distinct leakage mechanisms with different inflation
profiles. Details in `docs/findings.md`.

![Leakage dose-response curve](results/figures/leakage_dose_response.png)

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Feature files aren't included (see **Data** below for download commands).

## Reproducing the results

```bash
python -m src.experiments               # base benchmark: four architectures, six datasets
python -m src.efc_baselines              # add the two EFC-inspired baselines
python -m src.significance_tests         # joint significance test across all six models
python -m src.check_duplicate_dialogues  # duplicate-dialogue audit + deduplicated re-test
python -m src.capacity_control           # capacity-matched leakage control
python -m src.dose_response              # dose-response curve and figure
```

Each script caches its results per dataset under `results/cache*/` and picks up where it left
off if interrupted. DailyDialog (13k dialogues) is the slow one; on a Mac, wrapping a run in
`caffeinate -dims <command>` stops the machine from sleeping mid-run.

## Data

Off-the-shelf features only — nothing is extracted from raw audio/video here.

**COSMIC RoBERTa** (text only; IEMOCAP, MELD, EmoryNLP, DailyDialog), from
[declare-lab/conv-emotion](https://github.com/declare-lab/conv-emotion):
```bash
pip install gdown
gdown 1TQYQYCoPtdXN2rQ1mR2jisjUztmOzfZr -O data/cosmic_features.zip
unzip -j data/cosmic_features.zip \
  "erc-training/iemocap/iemocap_features_roberta.pkl" \
  "erc-training/meld/meld_features_roberta.pkl" \
  "erc-training/emorynlp/emorynlp_features_roberta.pkl" \
  "erc-training/dailydialog/dailydialog_features_roberta.pkl" -d data/feat
```

**MM-DFN multimodal** (text + OpenSmile audio + DenseNet visual; IEMOCAP, MELD), from
[zerohd4869/MM-DFN](https://github.com/zerohd4869/MM-DFN):
```bash
curl -sL -o data/feat/IEMOCAP_features.pkl \
  https://raw.githubusercontent.com/zerohd4869/MM-DFN/main/data/iemocap/IEMOCAP_features.pkl
curl -sL -o data/feat/MELD_features_raw1.pkl \
  https://raw.githubusercontent.com/zerohd4869/MM-DFN/main/data/meld/MELD_features_raw1.pkl
```

Pickle layouts are documented next to `_COSMIC_IDX` / `_MMDFN_IDX` in `src/dataset.py`.
Emotion labels are converted to shift labels (`1[y_{n+1} != y_n]`, speaker-aligned) by
`src/labels.py`; per-split counts are in `docs/label_conversion.md`.

## Protocol

- **Causal by construction, checked directly.** Every model is structurally causal (position
  n never has access to x_{n+1}), and we verify this rather than assume it: perturbing every
  input at or after the decision point to random noise leaves earlier outputs unchanged to
  floating-point exactness.
- **Strong baseline.** A speaker×emotion transition matrix, Laplace-smoothed, train-fold
  only, with a global backoff for speakers with few training observations
  (`src/baselines.py`).
- **One fixed hyperparameter configuration** across all six models — no per-model tuning, so
  unequal tuning effort can't explain why some models do better than others.
- **Three fixed seeds (0, 1, 2)** for every model/dataset/config combination, averaged and
  reported with standard deviation (`src/train.py` default `--seeds 0 1 2`).
- **Metrics**: shift-AUC as the primary, threshold-free metric, plus F1/precision/recall/
  balanced accuracy with automatic flagging of degenerate (near-constant) predictors
  (`src/evaluate.py`).
- **Significance**: a paired, cluster-robust bootstrap (resampling whole dialogues, not
  utterances) against the transition matrix, Holm-Bonferroni corrected jointly across all
  36 model×dataset comparisons (`src/significance_tests.py`).
- **Automated leakage audit**, an 8-item checklist run on every experiment
  (`src/leakage_audit.py` → `results/leakage_audit.md`).
- A bidirectional model (`gru_leaky`) is used only to measure how much leakage inflates the
  numbers above — never reported as a legitimate result.

## Paper ↔ repository

| Paper artifact | File |
|---|---|
| Table I — dataset statistics, F1 degeneracy | `results/dataset_stats.md` |
| Table II — main result | `results/benchmark_table.md` |
| Table III — MELD feature-source ablation | `results/feature_ablation_meld.md` |
| Fig. 1 — leakage dose-response | `results/dose_response.md`, `paper/figures/leakage_dose_response.pdf` |
| Significance tests (36 comparisons) | `results/significance_tests.md` |
| Causal construction, robustness checks | `docs/causal_reimplementations.md`, `docs/robustness.md` |
| Findings, discussion | `docs/findings.md` |

## Layout

```
paper/                       submitted PDF and a vector copy of Fig. 1
src/dataset.py                load COSMIC / MM-DFN pickles, build shift labels
src/labels.py                 emotion labels -> shift labels (docs/label_conversion.md)
src/baselines.py              base-rate, no-change, transition matrix, text-history MLP
src/losses.py                 focal / class-balanced loss
src/models.py                 causal GRU/TCN/Transformer, the leaky control, LookaheadGRU
src/models_families.py        causal DialogueRNN, DialogueGCN, DAG-ERC
src/models_efc_baselines.py   causal PEC-style and Pseudo-future-style models
src/train.py                  training/eval loop shared by every model
src/evaluate.py                metrics, dialogue-level bootstrap, paired significance test
src/leakage_audit.py          automated leakage checklist
src/experiments.py             main harness: all datasets x the four core architectures
src/efc_baselines.py          adds the two EFC-inspired baselines to the benchmark
src/holm_correction.py        Holm-Bonferroni correction
src/significance_tests.py     joint significance test across all six models
src/check_duplicate_dialogues.py  duplicate-dialogue audit and deduplicated re-test
src/capacity_control.py       capacity-matched leakage control
src/dose_response.py          leakage dose-response curve and figure
```

## Citation

```bibtex
@inproceedings{wen2027inertia,
  title     = {Inertia, Leakage, and the Illusion of Progress in Conversational Emotion Forecasting},
  author    = {Wen, Bin},
  booktitle = {Proc. IEEE ICASSP},
  year      = {2027}
}
```

## License

MIT — see [`LICENSE`](LICENSE).
