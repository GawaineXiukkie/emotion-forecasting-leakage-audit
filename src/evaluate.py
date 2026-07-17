"""
Metrics for shift forecasting, plus dialogue-level bootstrap confidence intervals.

Headline metrics are computed over decision points (binary shift / no-shift):
  - weighted-F1, macro-F1 (overall)
  - shift-only F1, recall, AUC (shift = positive class)

The bootstrap resamples whole dialogues, not individual points, since utterances within a
dialogue are correlated and point-level resampling would understate the true variance.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score)


def expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray,
                               n_bins: int = 10) -> float:
    """Equal-width expected calibration error for binary probabilities.

    ECE is reported as a descriptive calibration diagnostic, not used for model
    selection or significance testing. Scores are clipped because a few external
    baselines may produce values infinitesimally outside [0, 1].
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.clip(np.asarray(y_score, dtype=np.float64), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Include probability 1.0 in the final bin.
    bin_id = np.minimum(np.digitize(y_score, edges[1:-1], right=False), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_id == b
        if not np.any(mask):
            continue
        ece += mask.mean() * abs(y_score[mask].mean() - y_true[mask].mean())
    return float(ece)


def shift_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    frac_pos = float(np.mean(y_pred))
    out = {
        "base_rate": float(np.mean(y_true)),            # shift prevalence (context for F1)
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "shift_f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "noshift_f1": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "shift_precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "shift_recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "frac_pred_shift": frac_pos,
        # degenerate predictor flag: predicts (almost) one class only
        "degenerate": bool(frac_pos > 0.98 or frac_pos < 0.02),
    }
    out["shift_auc"] = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    out["shift_pr_auc"] = (average_precision_score(y_true, y_score)
                           if len(np.unique(y_true)) > 1 else float("nan"))
    clipped = np.clip(np.asarray(y_score, dtype=np.float64), 0.0, 1.0)
    out["brier"] = brier_score_loss(y_true, clipped)
    out["ece10"] = expected_calibration_error(y_true, clipped, n_bins=10)
    return out


def _as_seed_matrix(scores: np.ndarray, n_points: int) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != n_points:
        raise ValueError(f"expected [seed,{n_points}] scores, got {arr.shape}")
    return arr


def hierarchical_bootstrap_auc(y_true: np.ndarray, scores_a: np.ndarray,
                               scores_b: np.ndarray, dialogue_ids: np.ndarray,
                               n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean seed-level AUC difference with seed x dialogue bootstrap uncertainty.

    Each replicate resamples training seeds and whole test dialogues. This aligns
    the point estimate (mean AUC across seeds) with its interval while preserving
    within-dialogue dependence. ``scores_b`` may be deterministic [N] or have one
    matched row per seed [S,N].
    """
    y_true = np.asarray(y_true)
    dialogue_ids = np.asarray(dialogue_ids, dtype=object)
    a = _as_seed_matrix(scores_a, len(y_true))
    b = _as_seed_matrix(scores_b, len(y_true))
    if b.shape[0] == 1 and a.shape[0] > 1:
        b = np.repeat(b, a.shape[0], axis=0)
    if a.shape != b.shape:
        raise ValueError(f"seed matrices must align, got {a.shape} and {b.shape}")

    uniq = np.unique(dialogue_ids)
    idx_by = {d: np.where(dialogue_ids == d)[0] for d in uniq}

    def delta_for(seed_idx: np.ndarray, idx: np.ndarray) -> float:
        vals = []
        yy = y_true[idx]
        if len(np.unique(yy)) < 2:
            return float("nan")
        for s in seed_idx:
            vals.append(roc_auc_score(yy, a[s, idx]) - roc_auc_score(yy, b[s, idx]))
        return float(np.mean(vals))

    full = np.arange(len(y_true))
    per_seed = [float(roc_auc_score(y_true, a[s]) - roc_auc_score(y_true, b[s]))
                for s in range(a.shape[0])]
    observed = float(np.mean(per_seed))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sampled_seeds = rng.integers(0, a.shape[0], size=a.shape[0])
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[d] for d in chosen])
        val = delta_for(sampled_seeds, idx)
        if not np.isnan(val):
            boots.append(val)
    if not boots:
        raise ValueError("all hierarchical bootstrap replicates were single-class")
    lo, hi = np.quantile(np.asarray(boots), [0.025, 0.975])
    return {
        "delta_auc": observed,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "per_seed_delta": per_seed,
        "n_seeds": int(a.shape[0]),
        "n_dialogues": int(len(uniq)),
        "n_boot": int(len(boots)),
    }


def paired_cluster_permutation_auc(y_true: np.ndarray, scores_a: np.ndarray,
                                   scores_b: np.ndarray, dialogue_ids: np.ndarray,
                                   n_perm: int = 1999, seed: int = 0) -> dict:
    """Dialogue-cluster paired randomization test for mean seed-level AUC.

    Under the sharp null, model and baseline score vectors are exchangeable. One
    coin flip per dialogue is shared across seeds, retaining the dependence induced
    by a deterministic baseline evaluated against several training seeds. The
    plus-one correction guarantees a valid non-zero Monte Carlo p-value.
    """
    y_true = np.asarray(y_true)
    dialogue_ids = np.asarray(dialogue_ids, dtype=object)
    a = _as_seed_matrix(scores_a, len(y_true))
    b = _as_seed_matrix(scores_b, len(y_true))
    if b.shape[0] == 1 and a.shape[0] > 1:
        b = np.repeat(b, a.shape[0], axis=0)
    if a.shape != b.shape:
        raise ValueError(f"seed matrices must align, got {a.shape} and {b.shape}")

    uniq, inverse = np.unique(dialogue_ids, return_inverse=True)

    def mean_delta(x: np.ndarray, z: np.ndarray) -> float:
        return float(np.mean([
            roc_auc_score(y_true, x[s]) - roc_auc_score(y_true, z[s])
            for s in range(x.shape[0])
        ]))

    observed = mean_delta(a, b)
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        swap = rng.integers(0, 2, size=len(uniq), dtype=np.int8)[inverse].astype(bool)
        pa = np.where(swap[None, :], b, a)
        pb = np.where(swap[None, :], a, b)
        if abs(mean_delta(pa, pb)) >= abs(observed) - 1e-15:
            extreme += 1
    p = (extreme + 1) / (n_perm + 1)
    return {
        "delta_auc": observed,
        "p_value": float(p),
        "n_perm": int(n_perm),
        "n_dialogues": int(len(uniq)),
        "alternative": "two-sided",
    }


def paired_bootstrap_auc(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray,
                         dialogue_ids: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired, cluster-robust test: AUC(a) - AUC(b), resampling dialogues. a = model, b = baseline.
    Returns delta, CI, and a two-sided bootstrap p-value for delta != 0."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(dialogue_ids)
    idx_by = {d: np.where(dialogue_ids == d)[0] for d in uniq}

    def auc(idx, s):
        return roc_auc_score(y_true[idx], s[idx]) if len(np.unique(y_true[idx])) > 1 else np.nan

    full = np.arange(len(y_true))
    delta = auc(full, score_a) - auc(full, score_b)
    deltas = []
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[d] for d in chosen])
        deltas.append(auc(idx, score_a) - auc(idx, score_b))
    deltas = np.array([d for d in deltas if not np.isnan(d)])
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())  # two-sided bootstrap p
    return {"delta_auc": float(delta), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(min(p, 1.0)), "beats_baseline": bool(lo > 0)}


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray,
                 dialogue_ids: np.ndarray, metric: str = "shift_f1",
                 n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Returns (point_estimate, ci_low, ci_high) for `metric`, resampling dialogues."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(dialogue_ids)
    idx_by_dlg = {d: np.where(dialogue_ids == d)[0] for d in uniq}

    def compute(idx):
        return shift_metrics(y_true[idx], y_pred[idx], y_score[idx])[metric]

    point = compute(np.arange(len(y_true)))
    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_dlg[d] for d in chosen])
        boots.append(compute(idx))
    return point, float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def summarize_seeds(per_seed: list[dict]) -> dict:
    """Mean ± std across seeds for each metric key."""
    keys = per_seed[0].keys()
    return {k: (float(np.mean([s[k] for s in per_seed])),
                float(np.std([s[k] for s in per_seed]))) for k in keys}
