# Findings

Full numbers: `results/benchmark_table.md`. Leakage audit: `results/leakage_audit.md`.
Reproduce: `python -m src.experiments`.

## Main result

Six causal models — GRU and causal re-implementations of DialogueRNN, DialogueGCN, and
DAG-ERC, plus two EFC-inspired strategy baselines (PEC-style, Pseudo-future-style) — sit
within ≲0.03 AUC of each other on every dataset (max spread: IEMOCAP, 0.029). Whether any of
them beats the speaker-transition-matrix baseline is decided by the dataset, not the model:

| dataset | transition | GRU | DialogueRNN | DialogueGCN | DAG-ERC | PEC | Pseudo-future | leaky (upper bound) |
|---|---|---|---|---|---|---|---|---|
| IEMOCAP | 0.642 | 0.621 | 0.613 | 0.622 | 0.593 | 0.628 | 0.620 | 0.651 |
| MELD | 0.683 | 0.554* | 0.564* | 0.574* | 0.575* | 0.562 | 0.555 | 0.752 |
| EmoryNLP | 0.616 | 0.520* | 0.528* | 0.548* | 0.522* | 0.556 | 0.520 | 0.622 |
| DailyDialog | 0.695 | 0.735 | 0.721 | 0.726 | 0.727 | 0.734 | 0.728 | 0.816 |
| IEMOCAP-mm | 0.642 | 0.569 | 0.569 | 0.553 | 0.557* | 0.576 | 0.563 | 0.590 |
| MELD-mm | 0.683 | 0.725 | 0.707 | 0.724 | 0.718 | 0.723 | 0.723 | 0.723 |

`*` = degenerate (auto-flagged, prediction rate >98% one class). On 4 of 6 configurations —
IEMOCAP, MELD, EmoryNLP, and IEMOCAP-mm — every safe model loses. Only DailyDialog and
MELD-multimodal see all six win. On three of those four losing configurations the leaky
bidirectional model recovers a win by seeing the future; on IEMOCAP-mm, even the leaky upper
bound stays below the transition matrix, and Pseudo-future-style is a jointly significant
loser there — inertia dominates so completely that no amount of future-peeking is enough
(see `results/significance_tests.md`).

## Findings

1. **Apparent progress on 3/4 text corpora is largely inertia.** A train-fold-only
   speaker×emotion transition matrix is a hard baseline that current architectures don't beat
   once the future is genuinely hidden from the model.
2. **Standard F1 is degenerate on high-shift-rate corpora.** On MELD and EmoryNLP, every
   method — including the transition matrix and every safe model — collapses toward "always
   predict shift": shift-F1 of 0.73-0.84 with balanced accuracy near 0.50. AUC and balanced
   accuracy are needed to see this; see `results/dataset_stats.md`.
3. **Leakage manufactures up to +0.20 AUC**, is not primarily a capacity artifact (a
   parameter-matched causal control closes only part of the gap; see `docs/robustness.md`),
   and is largest exactly where the legitimate signal is weakest.
4. **Model choice contributes negligibly next to feature choice and leakage.** Six
   structurally and strategically distinct models land within ≲0.03 AUC of each other on
   every dataset. The MELD win/loss flips entirely on feature source at fixed architecture:
   COSMIC RoBERTa text loses, MM-DFN text+audio+visual wins, and the gain is carried by the
   text channel alone (`results/feature_ablation_meld.md`) — reported as a hypothesis, since
   we didn't run a controlled re-extraction to rule out preprocessing differences between the
   two feature pipelines.
5. **Leakage is at least two operationally distinct modes, not one.** A dose-response sweep
   over how many future utterances a model can see shows the target utterance itself accounts
   for most of the windowed leakage, while full bidirectional access is a separate, larger
   effect with a different profile (`docs/robustness.md`, `results/dose_response.md`).
6. **The two EFC-inspired baselines fail in the same pattern as the ERC-derived
   architectures.** PEC-style and Pseudo-future-style were added specifically to test whether
   inertia-dominance is an artifact of repurposing recognition architectures for forecasting;
   both reproduce the identical win/loss pattern without any tuning aimed at that outcome
   (`docs/robustness.md`).

## Caveats

- The leaky model conflates two effects (future access and roughly double the parameters);
  the capacity-matched control in `docs/robustness.md` addresses this but the raw leakage gap
  should be read as an upper-ish bound.
- MELD, EmoryNLP, and DailyDialog's speaker IDs are per-dialogue local indices, not verified
  global identities, so their splits are not confirmed speaker-independent the way IEMOCAP's
  session-based split is. We ruled out exact-duplicate-dialogue contamination but not subtler
  speaker-identity or writing-style leakage across non-duplicate dialogues from the same
  speaker.
