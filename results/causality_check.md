# Causality check

For each model, every input at or after position 5 is replaced with random noise, and outputs strictly before that position are compared to the unperturbed run. A causal model's outputs before the cutpoint must be exactly unchanged, since they never had access to the perturbed positions.

| model | max output difference at t < cutpoint | causal |
|---|---|---|
| gru | 0.00e+00 | yes |
| tcn | 0.00e+00 | yes |
| transformer | 0.00e+00 | yes |
| dialoguernn | 0.00e+00 | yes |
| dialoguegcn | 0.00e+00 | yes |
| dagerc | 0.00e+00 | yes |
| pec | 0.00e+00 | yes |
| pseudofuture | 0.00e+00 | yes |
| pec_fixed | 0.00e+00 | yes |
| pseudofuture_fixed | 0.00e+00 | yes |
| transformer_future0 | 0.00e+00 | yes |
| state_factorized | 0.00e+00 | yes |

All 12 checked models pass (the six benchmark families, corrected EFC variants, optional causal encoders, the k=0 dose control, and the state-factorized forecaster).
