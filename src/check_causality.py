"""
Direct causality check for all six models: perturb every input at or after the decision
point to random noise and confirm outputs strictly before it are unchanged.

This is the check the paper's Protocol section describes: "perturbing every input at t >= n
to random noise leaves outputs at t < n changed by exactly 0.0 (floating-point identical) for
every model." It's checked structurally here (random weights, no training needed) since
causality is a property of the architecture, not the learned parameters.

Run:    python -m src.check_causality
Writes: results/causality_check.md
"""
from __future__ import annotations

import torch

from .models import build_model

MODELS = ["gru", "tcn", "transformer", "dialoguernn", "dialoguegcn", "dagerc",
         "pec", "pseudofuture"]
CUTPOINT = 5


def check_one(name: str, d_in: int = 16, T: int = 10, B: int = 2, seed: int = 0) -> float:
    torch.manual_seed(seed)
    model = build_model(name, d_in)
    model.eval()
    x = torch.randn(B, T, d_in)
    spk = torch.randint(0, 3, (B, T))
    with torch.no_grad():
        logits_a = model(x.clone(), spk)
        x_perturbed = x.clone()
        x_perturbed[:, CUTPOINT:, :] = torch.randn(B, T - CUTPOINT, d_in) * 100
        logits_b = model(x_perturbed, spk)
    return (logits_a[:, :CUTPOINT] - logits_b[:, :CUTPOINT]).abs().max().item()


def main():
    lines = ["# Causality check", "",
             f"For each model, every input at or after position {CUTPOINT} is replaced with "
             "random noise, and outputs strictly before that position are compared to the "
             "unperturbed run. A causal model's outputs before the cutpoint must be exactly "
             "unchanged, since they never had access to the perturbed positions.", "",
             "| model | max output difference at t < cutpoint | causal |",
             "|---|---|---|"]
    all_causal = True
    for name in MODELS:
        diff = check_one(name)
        causal = diff == 0.0
        all_causal &= causal
        lines.append(f"| {name} | {diff:.2e} | {'yes' if causal else 'NO -- LEAK'} |")
        print(f"{name:14s} max_diff={diff:.2e}  {'causal' if causal else 'LEAK'}", flush=True)
    lines += ["", "All six models pass." if all_causal else
             "WARNING: at least one model is not causal -- see the table above."]
    open("results/causality_check.md", "w").write("\n".join(lines) + "\n")
    print("Wrote results/causality_check.md")
    assert all_causal, "A model failed the causality check -- this must be fixed before reporting any result."


if __name__ == "__main__":
    main()
