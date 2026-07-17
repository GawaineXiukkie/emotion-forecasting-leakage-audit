# Speaker-relation audit

All transition probabilities are fit on the training fold only. AUC is evaluated on the original test fold. `Immediate` is the paper's adjacent-turn target; `self-shift` predicts the current speaker's emotion at their next own utterance.

## Immediate-turn target

| dataset | test points | next speaker same | shift rate: same | shift rate: switch | global-trans AUC | speaker-trans AUC |
|---|---:|---:|---:|---:|---:|---:|
| iemocap | 1,592 | 0.278 | 0.222 | 0.509 | 0.642 | 0.642 |
| meld | 2,330 | 0.242 | 0.423 | 0.631 | 0.676 | 0.676 |
| emorynlp | 905 | 0.091 | 0.744 | 0.719 | 0.608 | 0.608 |
| dailydialog | 6,740 | 0.000 | -- | 0.198 | 0.691 | 0.691 |
| iemocap_mm | 1,592 | 0.278 | 0.222 | 0.509 | 0.642 | 0.642 |
| meld_mm | 2,330 | 0.242 | 0.423 | 0.631 | 0.676 | 0.676 |

## Current speaker's next-own-utterance target

| dataset | test points | shift rate | mean gap (turns) | p90 gap | global-trans AUC | speaker-trans AUC |
|---|---:|---:|---:|---:|---:|---:|
| iemocap | 1,561 | 0.263 | 1.98 | 3.0 | 0.586 | 0.586 |
| meld | 1,864 | 0.538 | 2.21 | 4.0 | 0.663 | 0.663 |
| emorynlp | 739 | 0.673 | 2.51 | 4.0 | 0.592 | 0.592 |
| dailydialog | 5,740 | 0.180 | 2.00 | 2.0 | 0.657 | 0.657 |
| iemocap_mm | 1,561 | 0.263 | 1.98 | 3.0 | 0.595 | 0.595 |
| meld_mm | 1,864 | 0.538 | 2.21 | 4.0 | 0.663 | 0.663 |
