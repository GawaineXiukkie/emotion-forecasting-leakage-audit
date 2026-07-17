"""Independent utterance-local feature control for the Access revision.

Runs a GRU and deployable predicted-label transition baseline on four text corpora
using TF-IDF/SVD features fitted only on training utterances. DailyDialog uses the
duplicate-free split. Raw seed predictions and validation curves are retained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .access_revision_experiments import SEARCH_SPACE, jsonable, point_index
from .baselines import PredictedLabelTransitionMatrix, collect_shift_arrays, tune_threshold
from .evaluate import (hierarchical_bootstrap_auc, paired_cluster_permutation_auc,
                       shift_metrics)
from .experiments import COSMIC
from .local_features import load_tfidf_svd
from .train import fill_predicted_current_emotion, train_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(COSMIC), choices=list(COSMIC))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n-boot", type=int, default=1999)
    ap.add_argument("--n-perm", type=int, default=4999)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cache = Path("results/cache_access_local_features")
    cache.mkdir(parents=True, exist_ok=True)
    results = {}

    for dataset in args.datasets:
        split, feature_meta = load_tfidf_svd(
            COSMIC[dataset], dataset, decontaminate=(dataset == "dailydialog"))
        y, dids = point_index(split.test)
        val_y = collect_shift_arrays(split.val)[2]
        ds_dir = cache / dataset; ds_dir.mkdir(parents=True, exist_ok=True)

        search_rows = []
        for config in SEARCH_SPACE:
            path = ds_dir / f"search_{config['id']}.json"
            if path.exists() and not args.force:
                row = json.loads(path.read_text(encoding="utf-8"))
            else:
                metrics, ex = train_one(
                    split, "gru", config["loss"], "none", seed=0, epochs=args.epochs,
                    lr=config["lr"], hidden=config["hidden"], dropout=config["dropout"],
                    compute_ci=False, early_stopping=True, patience=5, min_epochs=5,
                    track_history=True, evaluate_test=False)
                row = {"config": config, "best_val_auc": ex["best_val_auc"],
                       "best_epoch": ex["best_epoch"], "history": ex["history"]}
                path.write_text(json.dumps(jsonable(row), indent=2), encoding="utf-8")
            search_rows.append(row)
        selected = max(search_rows,
                       key=lambda r: (r["best_val_auc"], -r["config"]["hidden"]))["config"]

        model_scores, model_metrics = [], []
        pred_scores, pred_metrics, erc_rows = [], [], []
        for seed in args.seeds:
            model_npz = ds_dir / f"gru_seed{seed}.npz"
            model_json = ds_dir / f"gru_seed{seed}.json"
            if model_npz.exists() and model_json.exists() and not args.force:
                with np.load(model_npz, allow_pickle=False) as z:
                    score = z["scores"]
                meta = json.loads(model_json.read_text(encoding="utf-8"))
            else:
                metrics, ex = train_one(
                    split, "gru", selected["loss"], "none", seed=seed, epochs=args.epochs,
                    lr=selected["lr"], hidden=selected["hidden"], dropout=selected["dropout"],
                    compute_ci=False, early_stopping=True, patience=5, min_epochs=5,
                    track_history=True)
                score = ex["scores"]
                meta = {"metrics": metrics, "history": ex["history"],
                        "best_val_auc": ex["best_val_auc"], "best_epoch": ex["best_epoch"],
                        "threshold": ex["threshold"], "config": selected}
                np.savez_compressed(model_npz, scores=np.asarray(score, dtype=np.float32),
                                    y=y.astype(np.int8), dids=dids)
                model_json.write_text(json.dumps(jsonable(meta), indent=2), encoding="utf-8")
            model_scores.append(score); model_metrics.append(meta["metrics"])

            erc = fill_predicted_current_emotion(split, seed=seed, tune_c=True)
            baseline = PredictedLabelTransitionMatrix(split.num_emotions).fit(split.train)
            val_score = baseline.predict_score(split.val)
            threshold = tune_threshold(val_score, val_y)
            score_b = baseline.predict_score(split.test)
            metrics_b = shift_metrics(y, (score_b >= threshold).astype(int), score_b)
            pred_scores.append(score_b); pred_metrics.append(metrics_b); erc_rows.append(erc)

        a, b = np.stack(model_scores), np.stack(pred_scores)
        ci = hierarchical_bootstrap_auc(y, a, b, dids, n_boot=args.n_boot, seed=3901)
        perm = paired_cluster_permutation_auc(y, a, b, dids, n_perm=args.n_perm, seed=3901)
        results[dataset] = {
            "features": feature_meta, "selected": selected, "search": search_rows,
            "gru_auc": {"mean": float(np.mean([m["shift_auc"] for m in model_metrics])),
                        "std": float(np.std([m["shift_auc"] for m in model_metrics]))},
            "predicted_transition_auc": {
                "mean": float(np.mean([m["shift_auc"] for m in pred_metrics])),
                "std": float(np.std([m["shift_auc"] for m in pred_metrics]))},
            "gru_pr_auc": {"mean": float(np.mean([m["shift_pr_auc"] for m in model_metrics])),
                           "std": float(np.std([m["shift_pr_auc"] for m in model_metrics]))},
            "erc": erc_rows, "inference": {**ci, **perm},
        }
        print(dataset, results[dataset]["gru_auc"], results[dataset]["inference"], flush=True)

    Path("results/access_local_feature_control.json").write_text(
        json.dumps(jsonable(results), indent=2), encoding="utf-8")
    lines = ["# Strict utterance-local feature control", "",
             "TF-IDF vocabulary/IDF and SVD are fitted on training utterances only; each row is "
             "transformed independently without dialogue context. DailyDialog is decontaminated.", "",
             "| dataset | GRU AUC | predicted-transition AUC | ΔAUC [95% CI] | permutation p |",
             "|---|---:|---:|---:|---:|"]
    for ds, row in results.items():
        g, p, r = row["gru_auc"], row["predicted_transition_auc"], row["inference"]
        lines.append(f"| {ds} | {g['mean']:.3f}±{g['std']:.3f} | "
                     f"{p['mean']:.3f}±{p['std']:.3f} | {r['delta_auc']:+.3f} "
                     f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {r['p_value']:.4g} |")
    Path("results/access_local_feature_control.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote results/access_local_feature_control.{json,md}")


if __name__ == "__main__":
    main()
