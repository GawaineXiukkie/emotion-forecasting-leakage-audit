# Capacity-matched leakage control

gru_wide is a causal GRU with hidden size 240 (~912k parameters, matching the leaky bidirectional GRU's ~887k within 3%).

| dataset | GRU (h=128) | GRU, matched capacity | leaky (bidirectional) | capacity-only gap | raw leakage gap | capacity-controlled gap |
|---|---|---|---|---|---|---|
| iemocap | 0.621 | 0.645 | 0.651 | +0.024 | +0.030 | +0.006 |
| meld | 0.554 | 0.555 | 0.752 | +0.001 | +0.198 | +0.197 |
| emorynlp | 0.511 | 0.540 | 0.617 | +0.029 | +0.106 | +0.077 |
| dailydialog | 0.733 | 0.729 | 0.819 | -0.004 | +0.086 | +0.090 |

Mean raw leakage gap: +0.105. Mean capacity-controlled gap: +0.092.
