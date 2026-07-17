# Causal state-factorized forecaster

No gold emotion label or future utterance is used at inference. DailyDialog is decontaminated.

| dataset | state-factorized AUC | predicted-transition AUC | ΔAUC [95% CI] | p | Holm p |
|---|---:|---:|---:|---:|---:|
| iemocap | 0.628±0.008 | 0.576±0.000 | +0.052 [-0.006, +0.116] | 0.4048 | 0.4792 |
| meld | 0.628±0.004 | 0.587±0.000 | +0.041 [+0.021, +0.061] | 0.0012 | 0.0036 |
| emorynlp | 0.546±0.008 | 0.503±0.000 | +0.043 [-0.015, +0.098] | 0.2396 | 0.4792 |
| dailydialog | 0.747±0.001 | 0.581±0.002 | +0.166 [+0.149, +0.182] | 0.0002 | 0.0008 |
