# Robustness checks

Supplementary checks that back the claims in the paper but didn't fit in four pages.

## Duplicate dialogues across train/test

DailyDialog's official split is not fully train/test disjoint at the text level. We checked
every dataset for exact duplicate dialogues (identical utterance-text sequence appearing in
both a training and a test/val dialogue):

| dataset | test dialogues | test duplicates | val dialogues | val duplicates |
|---|---|---|---|---|
| IEMOCAP | 31 | 0 (0.0%) | 12 | 0 (0.0%) |
| MELD | 280 | 1 (0.4%) | 114 | 0 (0.0%) |
| EmoryNLP | 79 | 0 (0.0%) | 89 | 0 (0.0%) |
| DailyDialog | 1000 | 134 (13.4%) | 1000 | 101 (10.1%) |

IEMOCAP, MELD, and EmoryNLP are clean. DailyDialog is not — a known data-quality issue with
that corpus, not an artifact of our pipeline (`src/dataset.py::find_exact_duplicate_dialogues`).

We re-ran the GRU-vs-transition-matrix comparison on a deduplicated DailyDialog split to check
whether the model's win there survives:

| split | test size | transition AUC | GRU AUC | ΔAUC | p |
|---|---|---|---|---|---|
| original | 1000 | 0.695 | 0.733±0.005 | +0.045 [0.022, 0.067] | <.001 |
| deduplicated | 866 | 0.697 | 0.728±0.005 | +0.038 [0.013, 0.062] | 0.003 |

The win holds after deduplication, so it isn't an artifact of the duplicate dialogues.

## Capacity-matched leakage control

The leaky bidirectional GRU has roughly twice the parameters of the standard causal GRU at
the same hidden size, so some of its AUC advantage could be capacity rather than access to
future context. We built a capacity-matched causal GRU (hidden size 240, ~912k parameters,
within 3% of the leaky model's ~887k) and compared the resulting gap to the raw one:

| dataset | GRU (h=128) | GRU, matched capacity | leaky (bidirectional) | capacity-only gap | raw leakage gap | capacity-controlled gap |
|---|---|---|---|---|---|---|
| IEMOCAP | 0.621 | 0.645 | 0.651 | +0.024 | +0.030 | +0.006 |
| MELD | 0.554 | 0.555 | 0.752 | +0.001 | +0.198 | +0.197 |
| EmoryNLP | 0.511 | 0.540 | 0.617 | +0.029 | +0.106 | +0.077 |
| DailyDialog | 0.733 | 0.729 | 0.819 | -0.004 | +0.086 | +0.090 |

Mean raw gap 0.105 vs. mean capacity-controlled gap 0.092 — capacity accounts for a small
part of the effect on some datasets (IEMOCAP), essentially none on others (MELD, DailyDialog).
The leakage gap is not primarily a parameter-count artifact.

## EFC-inspired strategy baselines

Beyond the four ERC-derived architectures, we implemented two strategies specific to the
emotion-forecasting literature under the same protocol and hyperparameter budget: a PEC-style
sequence/self-dependency/recency model and a Pseudo-future-style predict-then-classify model
(`src/models_efc_baselines.py`). Both were verified causal directly before training: perturbing every
input at or after the decision point to random noise leaves earlier-position outputs
unchanged to floating-point exactness.

| dataset | transition | PEC-style | Pseudo-future-style |
|---|---|---|---|
| IEMOCAP | 0.642 | 0.628±0.004 | 0.620±0.016 |
| MELD | 0.683 | 0.562±0.005 | 0.555±0.011 |
| EmoryNLP | 0.616 | 0.556±0.010 | 0.520±0.008 |
| DailyDialog | 0.695 | 0.734±0.004 | 0.728±0.001 |
| IEMOCAP-mm | 0.642 | 0.576±0.006 | 0.563±0.027 |
| MELD-mm | 0.683 | 0.723±0.003 | 0.723±0.002 |

Both strategies lose on the same four configurations as the ERC-derived architectures and win
on the same two, without any tuning aimed at reproducing that pattern. This is the basis for
the paper's claim that inertia-dominance is a property of the task and evaluation protocol
rather than an artifact specific to repurposed recognition architectures.

Adding these two models to the joint significance test (`results/significance_tests.md`)
brings the family to 36 comparisons (6 models × 6 datasets); 10 survive Holm correction. All
six models' MELD losses are jointly significant — the single most robust result in the paper.
On DailyDialog, only GRU and DAG-ERC's wins clear the stricter threshold; PEC-style,
Pseudo-future-style, DialogueRNN, and DialogueGCN's wins are directionally consistent but do
not. Pseudo-future-style's IEMOCAP-mm loss is also jointly significant, reinforcing that
inertia dominates that configuration completely.

## Leakage dose-response

To break the leaky-vs-safe comparison into finer steps, we measured shift-AUC as a function
of k, the number of future utterances a model is allowed to see: k=0 is the safe causal
model, k in {1, 2, 4} use a mean-pooled peek at the next k utterances, and k=infinity is the
fully bidirectional model. Full table and figure: `results/dose_response.md`.

The curve isn't a smooth ramp. On MELD and DailyDialog, the jump from k=0 to k=1 accounts
for most of the inflation from finite-window leakage (MELD +0.059) — seeing the very next
utterance is doing most of the damage. Going from k=1 to k=4 is flat or even declines
slightly, because averaging in utterances further from the decision point dilutes the sharp
signal from the immediate next utterance rather than adding to it. A second, larger jump
appears only at k=infinity, where the bidirectional model encodes each future utterance with
its own representation instead of pooling them. On IEMOCAP-mm and MELD-mm the curve is flat
across the whole range, including k=infinity — where the legitimate multimodal signal is
already strong, extra access to the future doesn't add anything, and IEMOCAP-mm's fully
bidirectional model still doesn't clear the transition-matrix baseline.

## Causal construction, per model

- **DialogueRNN**: global, party, and emotion states update via chained `GRUCell` calls over
  a sequential loop across timesteps; no state ever has access to a future input.
- **DialogueGCN**: relational graph convolution over a causal, windowed adjacency — edge
  (i→j) exists only if j ≤ i and i − j ≤ 6 (a fixed 6-utterance lookback), with separate
  learned transforms for same- and different-speaker edges.
- **DAG-ERC**: relational attention over a causal but unbounded mask (i attends to all j ≤ i,
  any distance), with a learned same/different-speaker bias on the attention logits.
- **Leaky control**: a bidirectional GRU, the only intentionally non-causal model in the
  suite, used only to measure how much leakage inflates the numbers above.

## Statistical procedure

- Transition matrix: Laplace smoothing with α=1.0; a speaker's own transition row is used
  only if they have ≥20 observed training transitions from the current state, otherwise the
  estimator falls back to the global (all-speaker) table.
- Significance tests: a paired, cluster-robust bootstrap resampling whole dialogues (not
  individual utterances), 2000 resamples, computed on the same resampled dialogue set for
  model and baseline together (`src/evaluate.py::paired_bootstrap_auc`).
- Decision threshold: chosen by grid search on the validation split, maximizing shift-F1.
  Used only for F1/precision/recall/balanced-accuracy; AUC is threshold-free.
- Hyperparameters: identical, untuned configuration across all six models and all six
  datasets — Adam, learning rate 1e-3, batch size 32, hidden size 128, focal loss, 20 epochs,
  no early stopping or search (`src/train.py::train_one`).
