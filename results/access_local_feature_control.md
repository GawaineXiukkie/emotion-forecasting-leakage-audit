# Strict utterance-local feature control

TF-IDF vocabulary/IDF and SVD are fitted on training utterances only; each row is transformed independently without dialogue context. DailyDialog is decontaminated.

| dataset | GRU AUC | predicted-transition AUC | ΔAUC [95% CI] | permutation p |
|---|---:|---:|---:|---:|
| iemocap | 0.642±0.001 | 0.552±0.000 | +0.089 [+0.010, +0.154] | 0.043 |
| meld | 0.539±0.004 | 0.538±0.000 | +0.001 [-0.027, +0.028] | 0.9724 |
| emorynlp | 0.555±0.005 | 0.494±0.000 | +0.061 [+0.014, +0.112] | 0.1674 |
| dailydialog | 0.673±0.001 | 0.509±0.001 | +0.164 [+0.141, +0.185] | 0.0002 |
