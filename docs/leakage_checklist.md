# Forecasting validity checklist

## Temporal contract

- [ ] Decision `n` uses only utterances `<=n`; utterance `n+1` and later are absent.
- [ ] Future-perturbation tests leave every earlier logit unchanged.
- [ ] Terminal and padded positions use `IGNORE_INDEX` in every loss and metric.
- [ ] Any deliberately leaky mask is labeled sensitivity-only and excluded from headline results.

## Feature provenance

- [ ] Producer code or an independent local extraction establishes utterance scope.
- [ ] Feature/label/speaker lengths align and all values are finite.
- [ ] Repeated-text inconsistencies and cross-source ID/text/label alignment are audited.
- [ ] Externally precomputed features with incomplete producers are labeled sensitivity analyses.

## Information matching

- [ ] Deployable transition baselines use train-only predicted current labels.
- [ ] Oracle models and oracle baselines both receive the same gold current label.
- [ ] Gold-label diagnostics are not presented as deployable headline comparisons.
- [ ] Dialogue-local speaker roles are qualified by dialogue ID.

## Selection and inference

- [ ] Hyperparameter budgets are equal and test evaluation is disabled during search.
- [ ] Checkpoints and operating thresholds are selected on validation only.
- [ ] All training seeds are retained; inference resamples seeds and whole dialogues.
- [ ] Paired p-values use a valid null procedure and plus-one correction.
- [ ] The multiple-comparison family is declared before results are interpreted.

## Reporting

- [ ] Report ROC-AUC, PR-AUC, Brier/ECE, balanced accuracy, F1, and prediction rate.
- [ ] Name interaction-level versus next-own-utterance self-shift explicitly.
- [ ] Remove or clearly report exact cross-split dialogue duplicates.
- [ ] Release masks, split IDs, hashes, raw seed scores, environment locks, and table producers.
