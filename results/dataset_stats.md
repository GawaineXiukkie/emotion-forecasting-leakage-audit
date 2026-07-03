# Dataset statistics and F1 degeneracy (paper Table I)

Always-shift F1 = 2r/(1+r) for shift rate r (analytic; always-shift balanced accuracy is
0.500 by construction, omitted). Transition-BA ≈0.50 on MELD/EmoryNLP/IEMOCAP-mm shows
standard F1 sits at the degenerate ceiling on exactly the corpora where shift is common;
only IEMOCAP and DailyDialog clear it.

| Dataset | #Dial | #Utt | Shift r | Always-shift F1 | Transition BA |
|---|---|---|---|---|---|
| IEMOCAP | 151 | 7,433 | 0.429 | 0.600 | 0.611 |
| MELD | 1,432 | 13,708 | 0.581 | 0.735 | 0.500* |
| EmoryNLP | 827 | 9,489 | 0.722 | 0.839 | 0.500* |
| DailyDialog | 13,118 | 102,979 | 0.198 | 0.331 | 0.681 |
| IEMOCAP-mm | 151 | 7,433 | 0.429 | 0.600 | 0.500* |
| MELD-mm | 1,432 | 13,708 | 0.581 | 0.735 | 0.500* |

`*` = the transition matrix's balanced accuracy is at (or indistinguishable from) the
degenerate floor of a constant "always shift" predictor on this dataset.

Reproduce: dialogue/utterance counts via `src.dataset.load_cosmic` / `load_mmdfn` on each
of the six feature files; shift rate and transition-BA read from `results/benchmark_table.md`
(test split, val-tuned threshold).
