# Robustness and sensitivity checks

## Split contamination

The raw DailyDialog release contains 134/1,000 test and 101/1,000 validation dialogues exactly
duplicated in training. Every formal Access run removes them before search, checkpoint selection,
baseline fitting, or evaluation. MELD contains one exact test duplicate; this is disclosed and not
used as evidence of temporal leakage. See `results/access_data_quality.md`.

## Feature provenance

COSMIC producer code at audited commit `6128ca20e9c736605cce7e99d5d95db0356c35f5` extracts
IEMOCAP, MELD, and EmoryNLP RoBERTa rows per utterance. DailyDialog is supported by schema/output
checks. MM-DFN does not include complete producers and repeated identical MELD strings can have
different vectors, so MM-DFN is sensitivity evidence only. See `docs/feature_provenance.md`.

The independent control fits TF-IDF vocabulary/IDF and ARPACK truncated SVD on training utterances
only and transforms each row independently. All transformed values are checked finite. Its large
DailyDialog delta (+0.164) closely matches the RoBERTa delta (+0.166).

## Equal optimization budget

Every main architecture receives four validation candidates spanning hidden size, learning rate,
dropout, and focal/class-balanced loss. Seed 0 selects the configuration using validation ROC-AUC;
the selected configuration is trained with seeds 0/1/2 and early-stopped on validation checkpoints.
Test scores are not computed during search.

## Inference

Intervals use 1,999 seed-by-dialogue hierarchical bootstrap replicates. P-values use 4,999 paired
dialogue-cluster sign flips with a plus-one correction. Holm correction spans all 72 main
deployable/oracle comparisons. Raw scores for every seed are retained.

## Uniform leakage dose

A two-layer Transformer is fixed throughout; only its future-attention mask changes over
`k={0,1,2,4,all}`. MELD is 0.630/0.721/0.675/0.655/0.640 and duplicate-free DailyDialog is
0.741/0.809/0.791/0.774/0.771. This isolates access range from architecture and demonstrates a
target-utterance spike without a general monotonic dose effect.

## Target semantics

Immediate shift is mostly cross-speaker. A complete next-own-utterance self-shift benchmark retains
the causal history and changes only the supervised target index. Local role IDs in MELD, EmoryNLP,
and DailyDialog are dialogue-qualified before transition estimation.
