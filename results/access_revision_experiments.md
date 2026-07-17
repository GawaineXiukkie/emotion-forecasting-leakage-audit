# IEEE Access information-matched and tuned revision experiment

Every model receives the same four-trial validation search budget and uses the validation-selected checkpoint. Values are mean±SD over training seeds. Statistical intervals use a seed × dialogue hierarchical bootstrap; p-values use a paired dialogue-cluster permutation with a plus-one correction.

## Deployable setting: no gold current emotion

The transition baseline uses a train-only ERC prediction of the current label.

| dataset | predicted transition | gru | dialoguernn | dialoguegcn | dagerc | pec_fixed | pseudofuture_fixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| iemocap | 0.576 | 0.647±0.005 | 0.627±0.011 | 0.609±0.004 | 0.607±0.002 | 0.632±0.004 | 0.631±0.008 |
| meld | 0.588 | 0.621±0.010 | 0.629±0.014 | 0.623±0.005 | 0.608±0.014 | 0.620±0.015 | 0.600±0.015 |
| emorynlp | 0.501 | 0.521±0.014 | 0.533±0.011 | 0.536±0.014 | 0.554±0.011 | 0.539±0.022 | 0.542±0.008 |
| dailydialog | 0.580±0.000 | 0.746±0.002 | 0.744±0.002 | 0.748±0.002 | 0.744±0.004 | 0.743±0.002 | 0.741±0.004 |
| iemocap_mm | 0.566 | 0.560±0.003 | 0.598±0.005 | 0.593±0.007 | 0.559±0.002 | 0.581±0.015 | 0.578±0.019 |
| meld_mm | 0.584 | 0.723±0.003 | 0.716±0.004 | 0.724±0.002 | 0.720±0.005 | 0.720±0.002 | 0.723±0.004 |

## Oracle information-matched diagnostic

Both model and transition baseline receive gold current emotion.

| dataset | gold transition | gru | dialoguernn | dialoguegcn | dagerc | pec_fixed | pseudofuture_fixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| iemocap | 0.642 | 0.649±0.006 | 0.615±0.010 | 0.617±0.008 | 0.616±0.006 | 0.631±0.010 | 0.610±0.007 |
| meld | 0.676 | 0.630±0.005 | 0.629±0.008 | 0.628±0.007 | 0.644±0.005 | 0.632±0.002 | 0.612±0.014 |
| emorynlp | 0.608 | 0.553±0.012 | 0.561±0.011 | 0.561±0.015 | 0.563±0.012 | 0.591±0.010 | 0.525±0.024 |
| dailydialog | 0.694 | 0.808±0.004 | 0.820±0.002 | 0.826±0.003 | 0.822±0.001 | 0.804±0.003 | 0.800±0.005 |
| iemocap_mm | 0.642 | 0.566±0.024 | 0.590±0.011 | 0.598±0.007 | 0.577±0.003 | 0.594±0.008 | 0.569±0.008 |
| meld_mm | 0.676 | 0.802±0.001 | 0.791±0.007 | 0.805±0.005 | 0.807±0.006 | 0.798±0.005 | 0.798±0.010 |

## Seed- and dialogue-aware inference

| regime | dataset | model | ΔAUC | 95% CI | permutation p | Holm p | Holm significant |
|---|---|---|---:|---:|---:|---:|---|
| deployable | iemocap | gru | +0.071 | [+0.026, +0.119] | 0.0436 | 1 | no |
| deployable | iemocap | dialoguernn | +0.051 | [+0.011, +0.090] | 0.0708 | 1 | no |
| deployable | iemocap | dialoguegcn | +0.033 | [-0.010, +0.073] | 0.1452 | 1 | no |
| deployable | iemocap | dagerc | +0.031 | [-0.001, +0.061] | 0.5832 | 1 | no |
| deployable | iemocap | pec_fixed | +0.055 | [+0.015, +0.098] | 0.2784 | 1 | no |
| deployable | iemocap | pseudofuture_fixed | +0.055 | [+0.016, +0.094] | 0.2178 | 1 | no |
| deployable | meld | gru | +0.033 | [+0.011, +0.053] | 0.0188 | 0.7176 | no |
| deployable | meld | dialoguernn | +0.041 | [+0.016, +0.063] | 0.0008 | 0.0352 | yes |
| deployable | meld | dialoguegcn | +0.036 | [+0.017, +0.053] | 0.0084 | 0.3528 | no |
| deployable | meld | dagerc | +0.020 | [-0.007, +0.045] | 0.0878 | 1 | no |
| deployable | meld | pec_fixed | +0.032 | [+0.005, +0.056] | 0.025 | 0.925 | no |
| deployable | meld | pseudofuture_fixed | +0.013 | [-0.016, +0.039] | 0.4134 | 1 | no |
| deployable | emorynlp | gru | +0.027 | [-0.038, +0.094] | 0.4966 | 1 | no |
| deployable | emorynlp | dialoguernn | +0.040 | [-0.022, +0.101] | 0.1996 | 1 | no |
| deployable | emorynlp | dialoguegcn | +0.042 | [-0.016, +0.100] | 0.2078 | 1 | no |
| deployable | emorynlp | dagerc | +0.061 | [+0.001, +0.121] | 0.1888 | 1 | no |
| deployable | emorynlp | pec_fixed | +0.046 | [-0.016, +0.112] | 0.2076 | 1 | no |
| deployable | emorynlp | pseudofuture_fixed | +0.049 | [-0.005, +0.105] | 0.1674 | 1 | no |
| deployable | dailydialog | gru | +0.166 | [+0.148, +0.183] | 0.0002 | 0.0144 | yes |
| deployable | dailydialog | dialoguernn | +0.164 | [+0.146, +0.182] | 0.0002 | 0.0144 | yes |
| deployable | dailydialog | dialoguegcn | +0.167 | [+0.150, +0.185] | 0.0002 | 0.0144 | yes |
| deployable | dailydialog | dagerc | +0.164 | [+0.145, +0.182] | 0.0002 | 0.0144 | yes |
| deployable | dailydialog | pec_fixed | +0.163 | [+0.145, +0.180] | 0.0002 | 0.0144 | yes |
| deployable | dailydialog | pseudofuture_fixed | +0.160 | [+0.142, +0.179] | 0.0002 | 0.0144 | yes |
| deployable | iemocap_mm | gru | -0.006 | [-0.046, +0.039] | 0.8974 | 1 | no |
| deployable | iemocap_mm | dialoguernn | +0.032 | [-0.020, +0.079] | 0.1774 | 1 | no |
| deployable | iemocap_mm | dialoguegcn | +0.027 | [-0.014, +0.072] | 0.252 | 1 | no |
| deployable | iemocap_mm | dagerc | -0.007 | [-0.049, +0.037] | 0.76 | 1 | no |
| deployable | iemocap_mm | pec_fixed | +0.015 | [-0.031, +0.063] | 0.577 | 1 | no |
| deployable | iemocap_mm | pseudofuture_fixed | +0.012 | [-0.037, +0.062] | 0.5484 | 1 | no |
| deployable | meld_mm | gru | +0.139 | [+0.115, +0.166] | 0.0002 | 0.0144 | yes |
| deployable | meld_mm | dialoguernn | +0.132 | [+0.107, +0.160] | 0.0002 | 0.0144 | yes |
| deployable | meld_mm | dialoguegcn | +0.140 | [+0.114, +0.169] | 0.0002 | 0.0144 | yes |
| deployable | meld_mm | dagerc | +0.137 | [+0.110, +0.165] | 0.0002 | 0.0144 | yes |
| deployable | meld_mm | pec_fixed | +0.136 | [+0.110, +0.165] | 0.0002 | 0.0144 | yes |
| deployable | meld_mm | pseudofuture_fixed | +0.139 | [+0.115, +0.167] | 0.0002 | 0.0144 | yes |
| oracle | iemocap | gru | +0.007 | [-0.063, +0.079] | 0.8616 | 1 | no |
| oracle | iemocap | dialoguernn | -0.027 | [-0.080, +0.028] | 0.441 | 1 | no |
| oracle | iemocap | dialoguegcn | -0.025 | [-0.078, +0.031] | 0.4446 | 1 | no |
| oracle | iemocap | dagerc | -0.026 | [-0.072, +0.023] | 0.668 | 1 | no |
| oracle | iemocap | pec_fixed | -0.010 | [-0.072, +0.055] | 0.8432 | 1 | no |
| oracle | iemocap | pseudofuture_fixed | -0.032 | [-0.083, +0.028] | 0.5606 | 1 | no |
| oracle | meld | gru | -0.046 | [-0.069, -0.021] | 0.0006 | 0.027 | yes |
| oracle | meld | dialoguernn | -0.046 | [-0.070, -0.022] | 0.0002 | 0.0144 | yes |
| oracle | meld | dialoguegcn | -0.048 | [-0.071, -0.024] | 0.0002 | 0.0144 | yes |
| oracle | meld | dagerc | -0.032 | [-0.053, -0.011] | 0.0098 | 0.4018 | no |
| oracle | meld | pec_fixed | -0.044 | [-0.067, -0.019] | 0.0018 | 0.0774 | no |
| oracle | meld | pseudofuture_fixed | -0.064 | [-0.094, -0.037] | 0.0002 | 0.0144 | yes |
| oracle | emorynlp | gru | -0.056 | [-0.114, +0.006] | 0.1266 | 1 | no |
| oracle | emorynlp | dialoguernn | -0.048 | [-0.106, +0.010] | 0.0502 | 1 | no |
| oracle | emorynlp | dialoguegcn | -0.048 | [-0.109, +0.015] | 0.1648 | 1 | no |
| oracle | emorynlp | dagerc | -0.045 | [-0.095, +0.007] | 0.1124 | 1 | no |
| oracle | emorynlp | pec_fixed | -0.018 | [-0.079, +0.049] | 0.6042 | 1 | no |
| oracle | emorynlp | pseudofuture_fixed | -0.084 | [-0.147, -0.011] | 0.0184 | 0.7176 | no |
| oracle | dailydialog | gru | +0.115 | [+0.101, +0.129] | 0.0002 | 0.0144 | yes |
| oracle | dailydialog | dialoguernn | +0.126 | [+0.113, +0.140] | 0.0002 | 0.0144 | yes |
| oracle | dailydialog | dialoguegcn | +0.132 | [+0.119, +0.146] | 0.0002 | 0.0144 | yes |
| oracle | dailydialog | dagerc | +0.129 | [+0.116, +0.141] | 0.0002 | 0.0144 | yes |
| oracle | dailydialog | pec_fixed | +0.110 | [+0.094, +0.125] | 0.0002 | 0.0144 | yes |
| oracle | dailydialog | pseudofuture_fixed | +0.106 | [+0.091, +0.120] | 0.0002 | 0.0144 | yes |
| oracle | iemocap_mm | gru | -0.076 | [-0.135, -0.009] | 0.0986 | 1 | no |
| oracle | iemocap_mm | dialoguernn | -0.052 | [-0.109, +0.014] | 0.0892 | 1 | no |
| oracle | iemocap_mm | dialoguegcn | -0.044 | [-0.101, +0.022] | 0.2236 | 1 | no |
| oracle | iemocap_mm | dagerc | -0.065 | [-0.108, -0.013] | 0.1494 | 1 | no |
| oracle | iemocap_mm | pec_fixed | -0.048 | [-0.093, +0.008] | 0.1488 | 1 | no |
| oracle | iemocap_mm | pseudofuture_fixed | -0.073 | [-0.127, -0.010] | 0.0102 | 0.408 | no |
| oracle | meld_mm | gru | +0.126 | [+0.106, +0.146] | 0.0002 | 0.0144 | yes |
| oracle | meld_mm | dialoguernn | +0.116 | [+0.092, +0.140] | 0.0002 | 0.0144 | yes |
| oracle | meld_mm | dialoguegcn | +0.129 | [+0.104, +0.154] | 0.0002 | 0.0144 | yes |
| oracle | meld_mm | dagerc | +0.131 | [+0.106, +0.155] | 0.0002 | 0.0144 | yes |
| oracle | meld_mm | pec_fixed | +0.122 | [+0.101, +0.143] | 0.0002 | 0.0144 | yes |
| oracle | meld_mm | pseudofuture_fixed | +0.122 | [+0.100, +0.143] | 0.0002 | 0.0144 | yes |

## Additional metrics

Complete ROC-AUC, PR-AUC, Brier, ECE, balanced accuracy, F1, prediction-rate, ERC diagnostics, selected configurations, training curves, and per-seed runs are stored in `/Users/gawain/学术简历/new epc experiment/results/access_revision_experiments.json` and the cache namespace.
