# Benchmark: emotion-shift forecasting under a leakage-safe protocol

Metric = test **shift-AUC** (mean±std over seeds; threshold-free). `*` flags a degenerate (near-constant) predictor. COSMIC RoBERTa-1 features.

## iemocap  (shift base rate = 0.429)

| method | shift-AUC | shift-F1 | balanced-acc |
|---|---|---|---|
| baseline:base_rate | 0.500 * | 0.600 | 0.500 |
| baseline:no_change | 0.500 * | 0.000 | 0.500 |
| baseline:speaker_transition | 0.642 | 0.621 | 0.611 |
| baseline:text_history_mlp | 0.606 | 0.610 | 0.530 |
| gru:none | 0.621±0.009 | 0.606 | 0.552 |
| gru:predicted | 0.615±0.002 | 0.608 | 0.561 |
| gru:oracle | 0.622±0.002 | 0.609 | 0.561 |
| dialoguernn:none | 0.613±0.012 | 0.605 | 0.532 |
| dialoguegcn:none | 0.622±0.007 | 0.613 | 0.531 |
| dagerc:none | 0.593±0.029 | 0.610 | 0.542 |
| pec:none | 0.628±0.004 | 0.604 | 0.581 |
| pseudofuture:none | 0.620±0.016 | 0.617 | 0.570 |
| gru_leaky:none | 0.651±0.010 | 0.613 | 0.576 |

- **GRU(none) vs transition matrix** (paired, cluster-robust): ΔAUC = -0.009 [-0.078, +0.062], p=0.761 → does NOT beat baseline
- **Leakage gap** (leaky bidir-GRU − safe GRU, AUC): +0.031 (how much future-peeking inflates the score)

## meld  (shift base rate = 0.581)

| method | shift-AUC | shift-F1 | balanced-acc |
|---|---|---|---|
| baseline:base_rate | 0.500 * | 0.735 | 0.500 |
| baseline:no_change | 0.500 * | 0.000 | 0.500 |
| baseline:speaker_transition | 0.683 * | 0.735 | 0.500 |
| baseline:text_history_mlp | 0.550 * | 0.734 | 0.502 |
| gru:none | 0.554±0.013 * | 0.732 | 0.501 |
| gru:predicted | 0.564±0.007 * | 0.735 | 0.501 |
| gru:oracle | 0.599±0.006 * | 0.735 | 0.500 |
| dialoguernn:none | 0.564±0.009 * | 0.732 | 0.502 |
| dialoguegcn:none | 0.574±0.013 * | 0.734 | 0.501 |
| dagerc:none | 0.575±0.010 * | 0.735 | 0.500 |
| pec:none | 0.562±0.005 | 0.732 | 0.505 |
| pseudofuture:none | 0.555±0.011 * | 0.734 | 0.500 |
| gru_leaky:none | 0.752±0.001 | 0.771 | 0.645 |

- **GRU(none) vs transition matrix** (paired, cluster-robust): ΔAUC = -0.142 [-0.175, -0.108], p=0.000 → does NOT beat baseline
- **Leakage gap** (leaky bidir-GRU − safe GRU, AUC): +0.198 (how much future-peeking inflates the score)

## emorynlp  (shift base rate = 0.722)

| method | shift-AUC | shift-F1 | balanced-acc |
|---|---|---|---|
| baseline:base_rate | 0.500 * | 0.838 | 0.500 |
| baseline:no_change | 0.500 * | 0.000 | 0.500 |
| baseline:speaker_transition | 0.616 * | 0.838 | 0.500 |
| baseline:text_history_mlp | 0.517 * | 0.838 | 0.500 |
| gru:none | 0.520±0.013 * | 0.836 | 0.500 |
| gru:predicted | 0.525±0.007 * | 0.837 | 0.501 |
| gru:oracle | 0.545±0.004 * | 0.833 | 0.500 |
| dialoguernn:none | 0.528±0.004 * | 0.837 | 0.499 |
| dialoguegcn:none | 0.548±0.008 * | 0.837 | 0.500 |
| dagerc:none | 0.522±0.016 * | 0.838 | 0.500 |
| pec:none | 0.556±0.010 * | 0.837 | 0.499 |
| pseudofuture:none | 0.520±0.008 * | 0.835 | 0.499 |
| gru_leaky:none | 0.622±0.008 | 0.840 | 0.544 |

- **GRU(none) vs transition matrix** (paired, cluster-robust): ΔAUC = -0.113 [-0.175, -0.040], p=0.000 → does NOT beat baseline
- **Leakage gap** (leaky bidir-GRU − safe GRU, AUC): +0.102 (how much future-peeking inflates the score)

## dailydialog  (shift base rate = 0.198)

| method | shift-AUC | shift-F1 | balanced-acc |
|---|---|---|---|
| baseline:base_rate | 0.500 * | 0.331 | 0.500 |
| baseline:no_change | 0.500 * | 0.000 | 0.500 |
| baseline:speaker_transition | 0.695 | 0.498 | 0.681 |
| baseline:text_history_mlp | 0.687 | 0.408 | 0.636 |
| gru:none | 0.735±0.001 | 0.444 | 0.661 |
| gru:predicted | 0.730±0.008 | 0.439 | 0.659 |
| gru:oracle | 0.807±0.007 | 0.535 | 0.717 |
| dialoguernn:none | 0.721±0.009 | 0.436 | 0.654 |
| dialoguegcn:none | 0.726±0.009 | 0.437 | 0.653 |
| dagerc:none | 0.727±0.002 | 0.433 | 0.656 |
| pec:none | 0.734±0.004 | 0.433 | 0.654 |
| pseudofuture:none | 0.728±0.001 | 0.431 | 0.653 |
| gru_leaky:none | 0.816±0.001 | 0.532 | 0.716 |

- **GRU(none) vs transition matrix** (paired, cluster-robust): ΔAUC = +0.040 [+0.017, +0.061], p=0.002 → BEATS baseline
- **Leakage gap** (leaky bidir-GRU − safe GRU, AUC): +0.081 (how much future-peeking inflates the score)

## iemocap_mm  (shift base rate = 0.429)

| method | shift-AUC | shift-F1 | balanced-acc |
|---|---|---|---|
| baseline:base_rate | 0.500 * | 0.600 | 0.500 |
| baseline:no_change | 0.500 * | 0.000 | 0.500 |
| baseline:speaker_transition | 0.642 * | 0.600 | 0.500 |
| baseline:text_history_mlp | 0.600 | 0.616 | 0.540 |
| gru:none | 0.569±0.011 | 0.604 | 0.519 |
| gru:predicted | 0.564±0.012 | 0.597 | 0.526 |
| gru:oracle | 0.564±0.011 | 0.598 | 0.527 |
| dialoguernn:none | 0.569±0.003 | 0.590 | 0.527 |
| dialoguegcn:none | 0.553±0.020 | 0.602 | 0.535 |
| dagerc:none | 0.557±0.005 * | 0.597 | 0.515 |
| pec:none | 0.576±0.006 | 0.603 | 0.532 |
| pseudofuture:none | 0.563±0.027 | 0.598 | 0.512 |
| gru_leaky:none | 0.590±0.015 | 0.551 | 0.567 |

- **GRU(none) vs transition matrix** (paired, cluster-robust): ΔAUC = -0.084 [-0.140, -0.021], p=0.005 → does NOT beat baseline
- **Leakage gap** (leaky bidir-GRU − safe GRU, AUC): +0.021 (how much future-peeking inflates the score)

## meld_mm  (shift base rate = 0.581)

| method | shift-AUC | shift-F1 | balanced-acc |
|---|---|---|---|
| baseline:base_rate | 0.500 * | 0.735 | 0.500 |
| baseline:no_change | 0.500 * | 0.000 | 0.500 |
| baseline:speaker_transition | 0.683 * | 0.735 | 0.500 |
| baseline:text_history_mlp | 0.682 | 0.753 | 0.617 |
| gru:none | 0.725±0.003 | 0.758 | 0.629 |
| gru:predicted | 0.727±0.002 | 0.760 | 0.632 |
| gru:oracle | 0.793±0.002 | 0.790 | 0.678 |
| dialoguernn:none | 0.707±0.004 | 0.757 | 0.617 |
| dialoguegcn:none | 0.724±0.006 | 0.757 | 0.625 |
| dagerc:none | 0.718±0.002 | 0.762 | 0.637 |
| pec:none | 0.723±0.003 | 0.759 | 0.643 |
| pseudofuture:none | 0.723±0.002 | 0.758 | 0.641 |
| gru_leaky:none | 0.723±0.004 | 0.757 | 0.635 |

- **GRU(none) vs transition matrix** (paired, cluster-robust): ΔAUC = +0.038 [+0.008, +0.069], p=0.021 → BEATS baseline
- **Leakage gap** (leaky bidir-GRU − safe GRU, AUC): -0.002 (how much future-peeking inflates the score)

