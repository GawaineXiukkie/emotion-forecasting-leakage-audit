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

All six models pass.
