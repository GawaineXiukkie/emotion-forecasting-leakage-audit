# Posterior-marginalized deployable baseline

P(shift|x_n) = sum_k P(y_n=k|x_n) * P(y_{n+1}!=k | y_n=k, speaker), with the
class posterior from a train-only multinomial logistic regression and the
continuation terms from the train-fold transition matrix. Model scores are the
cached main-experiment runs; inference is the same seed x dialogue hierarchical
bootstrap and paired dialogue-cluster permutation, Holm-corrected across one
declared 36-comparison family.

| dataset | soft baseline AUC | gru | dialoguernn | dialoguegcn | dagerc | pec_fixed | pseudofuture_fixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| iemocap | 0.569±0.000 | +0.079 | +0.058 | +0.041 | +0.039 | +0.063 | +0.062 |
| meld | 0.596±0.000 | +0.025 | +0.033 | +0.027 | +0.012 | +0.024 | +0.004 |
| emorynlp | 0.525±0.000 | -0.004 | +0.008 | +0.011 | +0.030 | +0.014 | +0.017 |
| dailydialog | 0.662±0.004 | +0.083* | +0.082* | +0.085* | +0.082* | +0.081* | +0.078* |
| iemocap_mm | 0.580±0.000 | -0.020 | +0.018 | +0.013 | -0.021 | +0.001 | -0.002 |
| meld_mm | 0.605±0.000 | +0.117* | +0.111* | +0.118* | +0.115* | +0.115* | +0.118* |

`*` = survives the 36-comparison Holm correction (12/36 significant).
Deltas are model minus soft baseline (three-seed mean).
