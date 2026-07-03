# Label conversion log — ERC emotion → shift

Source features: `data/feat/dailydialog_features_roberta.pkl`

Rule: `shift[n] = 1[y_{n+1} != y_n]`; last utterance of each dialogue is IGNORE_INDEX.
Shift label is over consecutive dialogue utterances; speakers retained for the
speaker-specific transition-matrix baseline only.

| split | dialogues | utterances | decision pts | shift pts | shift rate | mean len |
|---|---|---|---|---|---|---|
| train | 11118 | 87170 | 76052 | 13654 | 0.180 | 7.8 |
| val | 1000 | 8069 | 7069 | 1086 | 0.154 | 8.1 |
| test | 1000 | 7740 | 6740 | 1337 | 0.198 | 7.7 |

Per-split emotion distribution:

- **train**: {0: 11182, 1: 72143, 2: 827, 3: 969, 4: 146, 5: 1600, 6: 303}
- **val**: {0: 684, 1: 7108, 2: 77, 3: 79, 4: 11, 5: 107, 6: 3}
- **test**: {0: 1019, 1: 6321, 2: 118, 3: 102, 4: 17, 5: 116, 6: 47}
