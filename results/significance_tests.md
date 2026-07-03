# Significance tests

Paired, cluster-robust (dialogue-level) bootstrap comparisons of each model against the speaker-transition-matrix baseline, 6 models (gru, dialoguernn, dialoguegcn, dagerc, pec, pseudofuture) x 6 datasets = 36 comparisons, Holm-corrected jointly since they support one claim: model choice doesn't change whether a given dataset favors the baseline.

| dataset | model | delta AUC | raw p | rank | Holm threshold | significant |
|---|---|---|---|---|---|---|
| meld | gru | -0.142 | 0.0000 | 1 | 0.0014 | yes |
| meld | pec | -0.117 | 0.0000 | 2 | 0.0014 | yes |
| meld | pseudofuture | -0.115 | 0.0000 | 3 | 0.0015 | yes |
| meld | dialoguernn | -0.116 | 0.0000 | 4 | 0.0015 | yes |
| meld | dialoguegcn | -0.092 | 0.0000 | 5 | 0.0016 | yes |
| meld | dagerc | -0.099 | 0.0000 | 6 | 0.0016 | yes |
| emorynlp | gru | -0.113 | 0.0000 | 7 | 0.0017 | yes |
| emorynlp | dagerc | -0.100 | 0.0000 | 8 | 0.0017 | yes |
| dailydialog | dagerc | +0.038 | 0.0010 | 9 | 0.0018 | yes |
| iemocap_mm | pseudofuture | -0.117 | 0.0010 | 10 | 0.0019 | yes |
| dailydialog | gru | +0.040 | 0.0020 | 11 | 0.0019 | no |
| emorynlp | pseudofuture | -0.106 | 0.0030 | 12 | 0.0020 | no |
| dailydialog | pec | +0.035 | 0.0050 | 13 | 0.0021 | no |
| iemocap_mm | gru | -0.084 | 0.0050 | 14 | 0.0022 | no |
| emorynlp | dialoguegcn | -0.097 | 0.0070 | 15 | 0.0023 | no |
| dailydialog | dialoguernn | +0.034 | 0.0100 | 16 | 0.0024 | no |
| dailydialog | dialoguegcn | +0.029 | 0.0130 | 17 | 0.0025 | no |
| iemocap_mm | pec | -0.072 | 0.0140 | 18 | 0.0026 | no |
| iemocap_mm | dagerc | -0.089 | 0.0140 | 19 | 0.0028 | no |
| dailydialog | pseudofuture | +0.031 | 0.0170 | 20 | 0.0029 | no |
| meld_mm | pseudofuture | +0.039 | 0.0190 | 21 | 0.0031 | no |
| meld_mm | gru | +0.038 | 0.0210 | 22 | 0.0033 | no |
| emorynlp | dialoguernn | -0.066 | 0.0250 | 23 | 0.0036 | no |
| meld_mm | pec | +0.037 | 0.0310 | 24 | 0.0038 | no |
| iemocap_mm | dialoguegcn | -0.067 | 0.0360 | 25 | 0.0042 | no |
| iemocap_mm | dialoguernn | -0.075 | 0.0380 | 26 | 0.0045 | no |
| meld_mm | dagerc | +0.036 | 0.0390 | 27 | 0.0050 | no |
| meld_mm | dialoguegcn | +0.034 | 0.0460 | 28 | 0.0056 | no |
| emorynlp | pec | -0.071 | 0.0550 | 29 | 0.0063 | no |
| meld_mm | dialoguernn | +0.029 | 0.0720 | 30 | 0.0071 | no |
| iemocap | dialoguernn | -0.012 | 0.7180 | 31 | 0.0083 | no |
| iemocap | pec | -0.012 | 0.7240 | 32 | 0.0100 | no |
| iemocap | dialoguegcn | -0.013 | 0.7260 | 33 | 0.0125 | no |
| iemocap | dagerc | -0.009 | 0.7400 | 34 | 0.0167 | no |
| iemocap | gru | -0.009 | 0.7610 | 35 | 0.0250 | no |
| iemocap | pseudofuture | +0.001 | 0.9880 | 36 | 0.0500 | no |

10/36 comparisons remain significant after joint Holm correction.

Note: the paper's claims rest on effect size and per-comparison confidence intervals; the Holm-corrected column here is a supplementary check against multiple-comparison inflation.
