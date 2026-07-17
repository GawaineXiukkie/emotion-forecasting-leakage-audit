"""Holm-Bonferroni step-down correction for multiple comparisons."""
from __future__ import annotations


def holm_bonferroni(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Standard Holm step-down procedure, including adjusted p-values.

    The adjusted value at ordered rank i is the running maximum of
    ``(m-i+1) * p_i`` (clipped to one). Reporting it as well as the sequential
    reject decision makes the multiplicity correction auditable without having
    to reconstruct it from rounded raw p-values.
    """
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    still_rejecting = True
    running_adjusted = 0.0
    for rank, (key, p) in enumerate(items, start=1):
        thresh = alpha / (m - rank + 1)
        running_adjusted = max(running_adjusted, (m - rank + 1) * p)
        adjusted = min(1.0, running_adjusted)
        reject = still_rejecting and (p <= thresh)
        if not reject:
            still_rejecting = False   # step-down: once we fail to reject, stop rejecting further
        out[key] = {"p": p, "p_holm": adjusted, "rank": rank,
                    "holm_threshold": thresh, "significant_holm": reject}
    return out
