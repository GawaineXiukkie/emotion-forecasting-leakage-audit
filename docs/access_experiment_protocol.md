# Locked IEEE Access revision protocol

This file records the decision rules used for the reviewer-driven experiments.
It is written before the state-factorized, local-feature, self-shift, and uniform
dose-response test results are inspected.

## Data and target

- The primary task is immediate interaction-level shift at decision point n:
  `1[y_(n+1) != y_n]`, with inputs limited to utterances `<= n`.
- The four COSMIC text configurations are primary. DailyDialog validation and
  test dialogues that exactly duplicate a training dialogue are removed before
  model selection or evaluation.
- MM-DFN configurations are sensitivity analyses because their complete feature
  producer is unavailable and MELD repeated-string vectors are not utterance-local
  to numerical tolerance.
- IEMOCAP uses global session-and-actor IDs. MELD, EmoryNLP, and DailyDialog expose
  dialogue-local roles only; these are qualified by dialogue ID so unrelated
  `role_0` values are never pooled as one person. Unseen/sparse test roles back off
  to the global train-fold transition matrix.
- Next-own-utterance self-shift changes only the supervised target index; the full
  causal dialogue history is retained.

## Model selection

- Search candidates are fixed in each producer before test evaluation.
- Selection uses validation ROC-AUC only. Test predictions are not computed in a
  fresh search run (`evaluate_test=False`).
- A selected configuration is retrained with seeds 0, 1, and 2. Early stopping
  uses validation AUC, patience 5, and at least 5 epochs.
- The headline comparison is deployable: no gold emotion is provided. Learned
  models are compared with a speaker-conditioned transition matrix driven by a
  train-only linear ERC prediction of the current emotion.
- The oracle diagnostic gives the gold current emotion to both sides and is not
  presented as a deployable result.

## Metrics and inference

- Primary discrimination metric: decision-point ROC-AUC.
- Secondary metrics: PR-AUC, Brier score, ten-bin ECE, balanced accuracy, and
  shift F1 at a validation-selected threshold.
- Point estimates are means over the three training seeds.
- The 95% interval resamples both training seeds and whole test dialogues.
- The two-sided randomization test swaps complete dialogue score blocks between
  model and baseline, with one shared swap across seeds and a plus-one Monte Carlo
  correction.
- Holm correction is applied within each declared family of comparisons. Raw and
  adjusted p-values are both retained.

## State-factorized model: decision rule fixed before testing

The new model jointly predicts current emotion, target future emotion, and the
binary shift. Its factorized score is
`1 - sum_k p_current(k) * p_future(k)`; a convex blend with the direct shift head
is selected from `{0, .25, .5, .75, 1}` on validation only.

We call its result a robust breakthrough only if all of the following hold:

1. the mean AUC delta versus the predicted-label transition baseline is positive
   on at least three of the four primary text corpora;
2. at least two corpus-level 95% intervals exclude zero in the positive direction;
3. at least one positive comparison survives the four-way Holm correction; and
4. no causality, split-integrity, or test-selection audit fails.

A weaker outcome is reported as exploratory, not promoted by changing the rule
after seeing test scores. Failed search configurations retain their validation
curves; every selected seed retains raw test predictions in the audit artifact.

## Leakage dose response

The same two-layer Transformer, optimizer, positional encoding, pooling, and
checkpoint rule are used for all conditions. Only the attention mask changes to
expose 0, 1, 2, 4, or all future utterances. These are diagnostic controls and
never legitimate deployable models.
