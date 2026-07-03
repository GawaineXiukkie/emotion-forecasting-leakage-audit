# Anti-leakage checklist

Check every item before reporting a number. The distinction between forecasting (this work)
and shift-aware ERC (recognition, which sees the current utterance) depends on all of them
holding.

## Temporal boundary
- [ ] The prediction at decision point `n` uses **only** utterances `≤ n`. The target
      utterance `n+1` is never an input (text, audio, or visual).
- [ ] Models are **structurally causal**: unidirectional GRU / left-padded causal TCN /
      subsequent-masked Transformer. Verified in `src/models.py`.
- [ ] The last utterance of each dialogue (no successor) is `IGNORE_INDEX`, excluded from
      loss and metrics.

## Per-modality windows
- [ ] Each cached feature vector for utterance `i` summarizes **only** utterance `i`
      (off-the-shelf ERC features are per-utterance — confirm for your specific pickle).
- [ ] No rolling/context feature in the cache spans into `n+1` (confirm with the feature
      source's README; log if any do and exclude them).

## Current-emotion settings
- [ ] `oracle` uses gold `y_n` only (current, already happened) — never `y_{n+1}`.
      Reported as **upper bound only**, not headline.
- [ ] `predicted` uses `ŷ_n` from a model that itself saw only `x_≤n`
      (`fill_predicted_current_emotion` predicts `y_n` from `x_n`).
- [ ] `none` appends no emotion label. Headline = `predicted` + `none`.

## Splits & statistics
- [ ] Train/val/test split is by **session/speaker** where applicable (IEMOCAP: by session →
      test speakers unseen). Logged.
- [ ] Speaker-specific transition matrix is estimated on the **train fold only**; unseen test
      speakers fall back to the global train matrix (`src/baselines.py`). No test-speaker
      ground-truth transition is ever used.
- [ ] Decision threshold is tuned on **val**, never on test.
- [ ] Bootstrap CI resamples **dialogues**, not utterances (`src/evaluate.py`).

## Demo only (never a reported baseline)
- [ ] A causal-mask-OFF (bidirectional) run may be shown to demonstrate the leakage gap,
      clearly labeled as leaky. It is excluded from all headline tables.
