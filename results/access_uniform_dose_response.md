# Same-architecture leakage dose-response

A single one-layer Transformer is used throughout; only the future-attention mask changes. One layer keeps the receptive field exactly k rather than the ~2k a stacked multi-layer encoder would give under a per-layer window mask.

| dataset | k=0 | k=1 | k=2 | k=4 | full | full-k0 |
|---|---:|---:|---:|---:|---:|---:|
| iemocap | 0.605 | 0.609 | 0.609 | 0.608 | 0.605 | +0.000 |
| meld | 0.620 | 0.685 | 0.651 | 0.637 | 0.632 | +0.012 |
| emorynlp | 0.550 | 0.547 | 0.544 | 0.552 | 0.548 | -0.002 |
| dailydialog | 0.736 | 0.780 | 0.774 | 0.758 | 0.751 | +0.015 |
| iemocap_mm | 0.580 | 0.579 | 0.578 | 0.575 | 0.568 | -0.012 |
| meld_mm | 0.699 | 0.717 | 0.721 | 0.715 | 0.709 | +0.010 |
