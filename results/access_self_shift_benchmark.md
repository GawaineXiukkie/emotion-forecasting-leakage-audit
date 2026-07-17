# Full next-own-utterance self-shift benchmark

Complete causal dialogue history is retained. DailyDialog uses the duplicate-free split.

| dataset | predicted transition | gru | dialoguernn | dialoguegcn | dagerc | pec_fixed | pseudofuture_fixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| iemocap | 0.541 | 0.609±0.007 | 0.602±0.008 | 0.597±0.011 | 0.598±0.004 | 0.593±0.006 | 0.605±0.014 |
| meld | 0.584 | 0.593±0.010 | 0.600±0.007 | 0.608±0.003 | 0.610±0.006 | 0.593±0.009 | 0.593±0.009 |
| emorynlp | 0.518 | 0.509±0.022 | 0.503±0.002 | 0.514±0.005 | 0.515±0.001 | 0.501±0.017 | 0.523±0.009 |
| dailydialog | 0.577 | 0.699±0.006 | 0.706±0.002 | 0.698±0.002 | 0.691±0.003 | 0.695±0.005 | 0.693±0.010 |
