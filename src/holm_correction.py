"""Holm-Bonferroni step-down correction for multiple comparisons."""
from __future__ import annotations


def holm_bonferroni(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Standard Holm step-down procedure. Returns per-key {p, rank, threshold, reject}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    still_rejecting = True
    for rank, (key, p) in enumerate(items, start=1):
        thresh = alpha / (m - rank + 1)
        reject = still_rejecting and (p <= thresh)
        if not reject:
            still_rejecting = False   # step-down: once we fail to reject, stop rejecting further
        out[key] = {"p": p, "rank": rank, "holm_threshold": thresh, "significant_holm": reject}
    return out
