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
from sklearn.metrics import (balanced_accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


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
    return out


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
