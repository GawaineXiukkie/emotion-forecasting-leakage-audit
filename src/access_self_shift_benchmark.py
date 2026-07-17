"""Full six-model next-own-utterance self-shift benchmark.

The full dialogue remains the causal input. Only the supervised decision target at
position n changes to the current speaker's next own utterance. Configurations are
selected by the main Access revision experiment, not retuned on this target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .access_revision_experiments import (MODELS, jsonable, load_revision_split,
                                          point_index)
from .baselines import (PredictedLabelTransitionMatrix, SpeakerTransitionMatrix,
                        collect_shift_arrays, tune_threshold)
from .dataset import apply_self_shift_target
from .evaluate import (hierarchical_bootstrap_auc,
                       paired_cluster_permutation_auc, shift_metrics)
from .experiments import COSMIC
from .holm_correction import holm_bonferroni
from .train import fill_predicted_current_emotion, train_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(COSMIC), choices=list(COSMIC))
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n-boot", type=int, default=1999)
    ap.add_argument("--n-perm", type=int, default=4999)
    ap.add_argument("--out-json", default="results/access_self_shift_benchmark.json")
    ap.add_argument("--out-md", default="results/access_self_shift_benchmark.md")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cache = Path("results/cache_access_self_shift")
    cache.mkdir(parents=True, exist_ok=True)
    results = {}

    for ds in args.datasets:
        split = load_revision_split(ds)
        for part in (split.train, split.val, split.test):
            apply_self_shift_target(part)
        y, dids = point_index(split.test)
        val_y = collect_shift_arrays(split.val)[2]

        gold = SpeakerTransitionMatrix(split.num_emotions).fit(split.train)
        gold_val = gold.predict_score(split.val); gold_scores = gold.predict_score(split.test)
        gold_thr = tune_threshold(gold_val, val_y)
        gold_metrics = shift_metrics(y, (gold_scores >= gold_thr).astype(int), gold_scores)

        pred_scores, erc_rows = [], []
        for seed in args.seeds:
            erc_rows.append(fill_predicted_current_emotion(split, seed=seed, tune_c=True))
            pred = PredictedLabelTransitionMatrix(split.num_emotions).fit(split.train)
            pred_scores.append(pred.predict_score(split.test))
        pred_scores = np.stack(pred_scores)

        ds_out = {"n_test": int(len(y)), "n_dialogues": int(len(np.unique(dids))),
                  "gold_transition_auc": gold_metrics["shift_auc"], "erc": erc_rows,
                  "predicted_transition_auc": float(np.mean([
                      shift_metrics(y, (s >= .5).astype(int), s)["shift_auc"]
                      for s in pred_scores])), "models": {}}
        for model in args.models:
            selected_path = Path("results/cache_access_revision") / ds / model / "selected.json"
            if not selected_path.exists():
                raise FileNotFoundError(f"Missing {selected_path}; finish main revision run first")
            config = json.loads(selected_path.read_text(encoding="utf-8"))["selected"]
            scores, metric_rows, run_rows = [], [], []
            for seed in args.seeds:
                folder = cache / ds / model; folder.mkdir(parents=True, exist_ok=True)
                meta_path = folder / f"seed{seed}.json"
                npz_path = folder / f"seed{seed}.npz"
                if meta_path.exists() and npz_path.exists() and not args.force:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    with np.load(npz_path, allow_pickle=False) as z:
                        score = z["scores"]
                else:
                    metrics, ex = train_one(
                        split, model, config["loss"], "none", seed=seed,
                        epochs=args.epochs, lr=config["lr"], hidden=config["hidden"],
                        dropout=config["dropout"], compute_ci=False, early_stopping=True,
                        patience=5, min_epochs=5, track_history=True)
                    score = ex["scores"]
                    meta = {"seed": seed, "config": config, "metrics": metrics,
                            "history": ex["history"], "best_epoch": ex["best_epoch"],
                            "best_val_auc": ex["best_val_auc"], "threshold": ex["threshold"]}
                    np.savez_compressed(npz_path, scores=np.asarray(score, dtype=np.float32),
                                        y=y.astype(np.int8), dids=dids)
                    meta_path.write_text(json.dumps(jsonable(meta), indent=2), encoding="utf-8")
                scores.append(score); metric_rows.append(meta["metrics"]); run_rows.append(meta)
            score_matrix = np.stack(scores)
            ci = hierarchical_bootstrap_auc(y, score_matrix, pred_scores, dids,
                                            n_boot=args.n_boot, seed=4901)
            perm = paired_cluster_permutation_auc(y, score_matrix, pred_scores, dids,
                                                  n_perm=args.n_perm, seed=4901)
            ds_out["models"][model] = {
                "auc_mean": float(np.mean([m["shift_auc"] for m in metric_rows])),
                "auc_std": float(np.std([m["shift_auc"] for m in metric_rows])),
                "pr_auc_mean": float(np.mean([m["shift_pr_auc"] for m in metric_rows])),
                "inference_vs_predicted_transition": {**ci, **perm}, "runs": run_rows,
            }
            print(ds, model, ds_out["models"][model]["auc_mean"], flush=True)
        results[ds] = ds_out

    corrected = holm_bonferroni({
        f"{ds}:{model}": row["inference_vs_predicted_transition"]["p_value"]
        for ds, d in results.items() for model, row in d["models"].items()
    })
    for key, value in corrected.items():
        ds, model = key.split(":", 1)
        results[ds]["models"][model]["inference_vs_predicted_transition"]["holm"] = value

    Path(args.out_json).write_text(
        json.dumps(jsonable(results), indent=2), encoding="utf-8")
    models = args.models
    lines = ["# Full next-own-utterance self-shift benchmark", "",
             "Complete causal dialogue history is retained. DailyDialog uses the duplicate-free split.", "",
             "| dataset | predicted transition | " + " | ".join(models) + " |",
             "|---|" + "---:|" * (len(models) + 1)]
    for ds, row in results.items():
        vals = [f"{row['predicted_transition_auc']:.3f}"] + [
            f"{row['models'][m]['auc_mean']:.3f}±{row['models'][m]['auc_std']:.3f}"
            for m in models]
        lines.append(f"| {ds} | " + " | ".join(vals) + " |")
    Path(args.out_md).write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
