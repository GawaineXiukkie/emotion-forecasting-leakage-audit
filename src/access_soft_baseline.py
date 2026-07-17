"""Posterior-marginalized deployable transition baseline (soft-label control).

The main deployable baseline commits to the ERC classifier's single hardest label
and looks up one transition-matrix row. That discards the classifier's uncertainty:
when the posterior is split between two emotions, the hard baseline bets everything
on one of them. This control marginalizes instead,

    P(shift | x_n) = sum_k  P(y_n = k | x_n) * P(y_{n+1} != k | y_n = k, speaker_n),

with the class posterior from a train-only multinomial logistic regression (same
scaler, feature rows, and 20k row cap as the hard baseline's ridge classifier) and
the per-class continuation terms from the same train-fold transition matrix. It is
the strongest predicted-label transition baseline we know how to build without
giving the baseline information the learned models do not have.

No neural training: model scores are read from the revision experiment's cache,
so this compares against exactly the runs reported in the paper. Inference reuses
the seed x dialogue hierarchical bootstrap and the paired dialogue-cluster
permutation, with one declared 36-comparison Holm family (6 models x 6 configs).

Run (resumable):
    python -m src.access_soft_baseline
Writes:
    results/access_soft_baseline.{json,md}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .access_revision_experiments import (align_cached_arrays, jsonable,
                                          load_revision_split, point_index)
from .baselines import SpeakerTransitionMatrix, iter_decision_points
from .evaluate import hierarchical_bootstrap_auc, paired_cluster_permutation_auc
from .experiments import ALL_KEYS
from .holm_correction import holm_bonferroni

MODELS = ["gru", "dialoguernn", "dialoguegcn", "dagerc", "pec_fixed", "pseudofuture_fixed"]
CACHE = Path("results/cache_access_revision")
OUT_CACHE = Path("results/cache_access_soft_baseline")


def _utterance_rows(dialogues) -> np.ndarray:
    return np.concatenate([
        np.concatenate(list(d.features.values()), axis=1) for d in dialogues
    ], axis=0).astype(np.float64, copy=False)


def _decision_rows(dialogues) -> np.ndarray:
    rows = [np.concatenate([d.features[m][n] for m in d.features], axis=0)
            for d, n in iter_decision_points(dialogues)]
    return np.asarray(rows, dtype=np.float64)


def soft_scores_for_seed(split, trans: SpeakerTransitionMatrix, seed: int) -> np.ndarray:
    """Marginalized shift score at every test decision point, one ERC fit per seed."""
    Xtr = _utterance_rows(split.train)
    ytr = np.concatenate([d.labels for d in split.train])
    if len(Xtr) > 20000:  # same cap and rng discipline as the hard baseline's classifier
        sel = np.random.default_rng(seed).choice(len(Xtr), 20000, replace=False)
        Xtr, ytr = Xtr[sel], ytr[sel]
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, random_state=seed).fit(
        scaler.transform(Xtr), ytr)

    proba = clf.predict_proba(scaler.transform(_decision_rows(split.test)))
    spk = np.asarray([d.speakers[n] for d, n in iter_decision_points(split.test)],
                     dtype=object)
    shift_given_k = np.asarray([
        [trans._shift_prob_for(int(k), s) for k in clf.classes_] for s in spk
    ])
    return (proba * shift_given_k).sum(axis=1)


def run_dataset(ds: str, seeds=(0, 1, 2), n_boot=1999, n_perm=4999) -> dict:
    OUT_CACHE.mkdir(parents=True, exist_ok=True)
    cache = OUT_CACHE / f"{ds}.json"
    if cache.exists():
        print(f"[{ds}] (cached)", flush=True)
        return json.loads(cache.read_text(encoding="utf-8"))

    split = load_revision_split(ds)
    y, dids = point_index(split.test)
    trans = SpeakerTransitionMatrix(split.num_emotions).fit(split.train)

    soft = np.stack([soft_scores_for_seed(split, trans, s) for s in seeds])
    per_seed_auc = [float(roc_auc_score(y, row)) for row in soft]
    out = {"soft_auc_mean": float(np.mean(per_seed_auc)),
           "soft_auc_std": float(np.std(per_seed_auc)),
           "per_seed_auc": per_seed_auc, "models": {}}
    print(f"[{ds}] soft baseline AUC={out['soft_auc_mean']:.3f}±{out['soft_auc_std']:.3f}",
          flush=True)

    for model in MODELS:
        rows = []
        for seed in seeds:
            with np.load(CACHE / ds / model / f"final_none_seed{seed}.npz",
                         allow_pickle=False) as z:
                arr = align_cached_arrays(
                    {"scores": z["scores"], "y": z["y"], "dids": z["dids"]}, dids)
            if not np.array_equal(arr["y"], y):
                raise AssertionError(f"{ds}/{model}: target misalignment")
            rows.append(arr["scores"])
        scores = np.stack(rows)
        ci = hierarchical_bootstrap_auc(y, scores, soft, dids, n_boot=n_boot, seed=3701)
        perm = paired_cluster_permutation_auc(y, scores, soft, dids, n_perm=n_perm, seed=3701)
        out["models"][model] = {**ci, **perm}
        print(f"[{ds}] {model}: Δ={ci['delta_auc']:+.3f}, p={perm['p_value']:.4g}", flush=True)

    cache.write_text(json.dumps(jsonable(out), indent=2), encoding="utf-8")
    return out


def main():
    results = {ds: run_dataset(ds) for ds in ALL_KEYS}

    pvals = {f"{ds}:{m}": results[ds]["models"][m]["p_value"]
             for ds in results for m in results[ds]["models"]}
    corrected = holm_bonferroni(pvals)
    for key, val in corrected.items():
        ds, m = key.split(":", 1)
        results[ds]["models"][m]["holm"] = val

    Path("results/access_soft_baseline.json").write_text(
        json.dumps(jsonable(results), indent=2), encoding="utf-8")

    lines = ["# Posterior-marginalized deployable baseline", "",
             "P(shift|x_n) = sum_k P(y_n=k|x_n) * P(y_{n+1}!=k | y_n=k, speaker), with the",
             "class posterior from a train-only multinomial logistic regression and the",
             "continuation terms from the train-fold transition matrix. Model scores are the",
             "cached main-experiment runs; inference is the same seed x dialogue hierarchical",
             "bootstrap and paired dialogue-cluster permutation, Holm-corrected across one",
             "declared 36-comparison family.", "",
             "| dataset | soft baseline AUC | " + " | ".join(MODELS) + " |",
             "|---|---:|" + "---:|" * len(MODELS)]
    for ds in results:
        r = results[ds]
        cells = []
        for m in MODELS:
            row = r["models"][m]
            star = "*" if row["holm"]["significant_holm"] else ""
            cells.append(f"{row['delta_auc']:+.3f}{star}")
        lines.append(f"| {ds} | {r['soft_auc_mean']:.3f}±{r['soft_auc_std']:.3f} | "
                     + " | ".join(cells) + " |")
    n_sig = sum(1 for ds in results for m in results[ds]["models"]
                if results[ds]["models"][m]["holm"]["significant_holm"])
    lines += ["", f"`*` = survives the 36-comparison Holm correction ({n_sig}/36 significant).",
              "Deltas are model minus soft baseline (three-seed mean)."]
    Path("results/access_soft_baseline.md").write_text("\n".join(lines) + "\n",
                                                       encoding="utf-8")
    print(f"Wrote results/access_soft_baseline.{{json,md}} ({n_sig}/36 survive Holm)")


if __name__ == "__main__":
    main()
