"""Reviewer-driven IEEE Access revision experiments.

This producer separates two information regimes and aligns every inferential
quantity with the three-seed descriptive estimate:

1. deployable: causal models receive no gold emotion label and are compared with
   a transition matrix driven by a train-only ERC prediction of the current label;
2. oracle diagnostic: every model and the transition matrix receive gold y_n.

All six models receive the same four-trial validation-search budget. The selected
configuration is retrained with seeds 0/1/2 using validation-AUC early stopping.
Every seed's raw prediction, training curve, threshold, and selected configuration
is saved. Inference combines a seed x dialogue hierarchical bootstrap interval
with a dialogue-cluster paired permutation p-value and joint Holm correction.

Run (resumable):
    python -m src.access_revision_experiments
Quick smoke test:
    python -m src.access_revision_experiments --datasets iemocap --models gru \
        --search-limit 1 --seeds 0 --epochs 2 --n-boot 49 --n-perm 49 --force
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .baselines import (PredictedLabelTransitionMatrix, SpeakerTransitionMatrix,
                        collect_shift_arrays, iter_decision_points, tune_threshold)
from .evaluate import (hierarchical_bootstrap_auc,
                       paired_cluster_permutation_auc, shift_metrics)
from .dataset import load_cosmic
from .experiments import ALL_KEYS, COSMIC, load_split
from .holm_correction import holm_bonferroni
from .train import fill_predicted_current_emotion, train_one


MODELS = ["gru", "dialoguernn", "dialoguegcn", "dagerc", "pec_fixed",
          "pseudofuture_fixed"]
MODEL_ALIASES = {"dialoguernnn": "dialoguernn"}

SEARCH_SPACE = [
    {"id": "h64_lr1e3_d10_focal", "hidden": 64, "lr": 1e-3,
     "dropout": 0.10, "loss": "focal"},
    {"id": "h128_lr1e3_d10_focal", "hidden": 128, "lr": 1e-3,
     "dropout": 0.10, "loss": "focal"},
    {"id": "h128_lr3e4_d30_focal", "hidden": 128, "lr": 3e-4,
     "dropout": 0.30, "loss": "focal"},
    {"id": "h256_lr3e4_d30_cbce", "hidden": 256, "lr": 3e-4,
     "dropout": 0.30, "loss": "cb_ce"},
]


def canonical_model(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def load_revision_split(dataset: str):
    # The Access headline uses the duplicate-free DailyDialog test/validation split.
    if dataset == "dailydialog":
        return load_cosmic(COSMIC[dataset], dataset, "roberta1", decontaminate=True)
    return load_split(dataset)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def point_index(dialogues):
    y = collect_shift_arrays(dialogues)[2]
    dids = np.asarray([d.did for d, _ in iter_decision_points(dialogues)], dtype=str)
    return y, dids


def cache_paths(cache_root: Path, dataset: str, model: str, tag: str):
    folder = cache_root / dataset / model
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{tag}.json", folder / f"{tag}.npz"


def save_run(meta_path: Path, score_path: Path, metrics: dict, extras: dict,
             config: dict, setting: str, seed: int):
    np.savez_compressed(score_path, scores=np.asarray(extras["scores"], dtype=np.float32),
                        y=np.asarray(extras["y"], dtype=np.int8),
                        dids=np.asarray(extras["dids"], dtype=str))
    payload = {
        "setting": setting,
        "seed": seed,
        "config": config,
        "metrics": metrics,
        "threshold": extras["threshold"],
        "history": extras["history"],
        "best_val_auc": extras["best_val_auc"],
        "best_epoch": extras["best_epoch"],
        "epochs_ran": extras["epochs_ran"],
        "scores_file": score_path.name,
    }
    meta_path.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")


def load_run(meta_path: Path, score_path: Path):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(score_path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return meta, arrays


def execute_run(split, dataset: str, model: str, config: dict, setting: str,
                seed: int, epochs: int, cache_root: Path, tag: str, force: bool):
    meta_path, score_path = cache_paths(cache_root, dataset, model, tag)
    if meta_path.exists() and score_path.exists() and not force:
        cached_meta, cached_arrays = load_run(meta_path, score_path)
        if (cached_meta.get("config") == config and cached_meta.get("setting") == setting
                and cached_meta.get("seed") == seed):
            return cached_meta, cached_arrays
    metrics, extras = train_one(
        split, canonical_model(model), config["loss"], setting, seed=seed,
        epochs=epochs, lr=config["lr"], hidden=config["hidden"],
        dropout=config["dropout"], compute_ci=False, early_stopping=True,
        patience=5, min_epochs=5, track_history=True,
    )
    save_run(meta_path, score_path, metrics, extras, config, setting, seed)
    return load_run(meta_path, score_path)


def select_config(split, dataset: str, model: str, epochs: int, cache_root: Path,
                  search_space: list[dict], force: bool) -> tuple[dict, list[dict]]:
    rows = []
    for config in search_space:
        tag = f"search_{config['id']}_seed0"
        meta_path, _ = cache_paths(cache_root, dataset, model, tag)
        meta = None
        if meta_path.exists() and not force:
            candidate = json.loads(meta_path.read_text(encoding="utf-8"))
            if (candidate.get("config") == config and candidate.get("setting") == "none"
                    and candidate.get("seed") == 0):
                meta = candidate
        if meta is None:
            _, extras = train_one(
                split, canonical_model(model), config["loss"], "none", seed=0,
                epochs=epochs, lr=config["lr"], hidden=config["hidden"],
                dropout=config["dropout"], compute_ci=False, early_stopping=True,
                patience=5, min_epochs=5, track_history=True, evaluate_test=False)
            meta = {"setting": "none", "seed": 0, "config": config,
                    "history": extras["history"],
                    "best_val_auc": extras["best_val_auc"],
                    "best_epoch": extras["best_epoch"],
                    "epochs_ran": extras["epochs_ran"],
                    "test_evaluated": False}
            meta_path.write_text(json.dumps(jsonable(meta), indent=2), encoding="utf-8")
        rows.append({"config": config, "best_val_auc": meta["best_val_auc"],
                     "best_epoch": meta["best_epoch"], "history": meta["history"]})
        print(f"[{dataset}] {model} search {config['id']}: "
              f"val={meta['best_val_auc']:.4f} epoch={meta['best_epoch']}", flush=True)
    selected = max(rows, key=lambda r: (r["best_val_auc"], -r["config"]["hidden"]))["config"]
    selected_path = cache_root / dataset / model / "selected.json"
    selected_path.write_text(json.dumps(jsonable({"selected": selected, "trials": rows}), indent=2),
                             encoding="utf-8")
    print(f"[{dataset}] {model} selected {selected['id']}", flush=True)
    return selected, rows


def baseline_runs(split, dataset: str, seeds: list[int], cache_root: Path,
                  force: bool) -> dict:
    folder = cache_root / dataset / "baselines"
    folder.mkdir(parents=True, exist_ok=True)
    y, dids = point_index(split.test)
    val_y = collect_shift_arrays(split.val)[2]

    gold_model = SpeakerTransitionMatrix(split.num_emotions).fit(split.train)
    gold_val = gold_model.predict_score(split.val)
    gold_scores = gold_model.predict_score(split.test)
    gold_thr = tune_threshold(gold_val, val_y)
    gold_metrics = shift_metrics(y, (gold_scores >= gold_thr).astype(int), gold_scores)

    predicted_scores, erc_rows, predicted_metrics = [], [], []
    for seed in seeds:
        path = folder / f"predicted_transition_seed{seed}.npz"
        meta_path = folder / f"predicted_transition_seed{seed}.json"
        if path.exists() and meta_path.exists() and not force:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (str(meta.get("erc", {}).get("method", "")).startswith(
                    ("train-only StandardScaler + Ridge"))
                    and meta.get("speaker_scope") == "dialogue-qualified-local-roles-v1"):
                with np.load(path, allow_pickle=False) as z:
                    predicted_scores.append(z["scores"])
                erc_rows.append(meta["erc"]); predicted_metrics.append(meta["metrics"])
                continue
        erc = fill_predicted_current_emotion(split, seed=seed, tune_c=True)
        pred_model = PredictedLabelTransitionMatrix(split.num_emotions).fit(split.train)
        pred_val = pred_model.predict_score(split.val)
        pred_thr = tune_threshold(pred_val, val_y)
        scores = pred_model.predict_score(split.test)
        metrics = shift_metrics(y, (scores >= pred_thr).astype(int), scores)
        np.savez_compressed(path, scores=np.asarray(scores, dtype=np.float32))
        meta_path.write_text(json.dumps(jsonable({"erc": erc, "metrics": metrics,
                                                  "threshold": pred_thr,
                                                  "speaker_scope":
                                                  "dialogue-qualified-local-roles-v1"}), indent=2),
                             encoding="utf-8")
        predicted_scores.append(scores); erc_rows.append(erc); predicted_metrics.append(metrics)

    np.savez_compressed(folder / "index_and_gold.npz", y=y.astype(np.int8), dids=dids,
                        gold_scores=np.asarray(gold_scores, dtype=np.float32))
    return {
        "y": y, "dids": dids, "gold_scores": np.asarray(gold_scores),
        "gold_metrics": jsonable(gold_metrics),
        "predicted_scores": np.stack(predicted_scores),
        "predicted_metrics": predicted_metrics, "erc": erc_rows,
    }


def aggregate_metrics(metas: list[dict]) -> dict:
    keys = ["shift_auc", "shift_pr_auc", "brier", "ece10", "balanced_acc",
            "shift_f1", "frac_pred_shift"]
    return {k: {"mean": float(np.mean([m["metrics"][k] for m in metas])),
                "std": float(np.std([m["metrics"][k] for m in metas]))}
            for k in keys}


def align_cached_arrays(arrays: dict, target_dids: np.ndarray) -> dict:
    """Reorder a self-indexed cache to the current deterministic dialogue order.

    Older caches may have traversed an upstream set-valued split under a different
    PYTHONHASHSEED. Each dialogue block retains its internal utterance order, so a
    stable (dialogue id, occurrence number) alignment is exact and avoids retraining.
    """
    source = arrays["dids"].astype(str)
    target = np.asarray(target_dids).astype(str)
    if np.array_equal(source, target):
        return arrays
    from collections import defaultdict, deque
    positions = defaultdict(deque)
    for i, did in enumerate(source):
        positions[did].append(i)
    try:
        order = np.asarray([positions[did].popleft() for did in target], dtype=np.int64)
    except (KeyError, IndexError) as exc:
        raise AssertionError("cached dialogue IDs cannot be aligned") from exc
    if any(positions.values()):
        raise AssertionError("cached dialogue multiplicities do not match current split")
    out = dict(arrays)
    for key in ("scores", "y", "dids"):
        out[key] = arrays[key][order]
    return out


def run_all(args) -> dict:
    cache_root = Path(args.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    search_space = SEARCH_SPACE[:args.search_limit]
    results = {}
    for dataset in args.datasets:
        print(f"\n=== {dataset} ===", flush=True)
        split = load_revision_split(dataset)
        baselines = baseline_runs(split, dataset, args.seeds, cache_root, args.force)
        ds_out = {
            "n_test": int(len(baselines["y"])),
            "n_dialogues": int(len(np.unique(baselines["dids"]))),
            "gold_transition": baselines["gold_metrics"],
            "predicted_transition": {
                "metrics": aggregate_metrics([{"metrics": m} for m in baselines["predicted_metrics"]]),
                "erc": baselines["erc"],
            },
            "models": {},
        }
        for model in args.models:
            selected, search_rows = select_config(split, dataset, model, args.epochs,
                                                   cache_root, search_space, args.force)
            model_out = {"selected": selected, "search": search_rows}
            setting_arrays = {}
            for setting in ("none", "oracle"):
                metas, arrays = [], []
                for seed in args.seeds:
                    tag = f"final_{setting}_seed{seed}"
                    meta, arr = execute_run(split, dataset, model, selected, setting, seed,
                                            args.epochs, cache_root, tag, args.force)
                    arr = align_cached_arrays(arr, baselines["dids"])
                    if not np.array_equal(arr["y"], baselines["y"]):
                        raise AssertionError(f"{dataset}/{model}/{setting}: target misalignment")
                    if not np.array_equal(arr["dids"].astype(str), baselines["dids"].astype(str)):
                        raise AssertionError(f"{dataset}/{model}/{setting}: dialogue misalignment")
                    metas.append(meta); arrays.append(arr["scores"])
                score_matrix = np.stack(arrays)
                setting_arrays[setting] = score_matrix
                model_out[setting] = {"metrics": aggregate_metrics(metas),
                                      "runs": metas}

            deploy_ci = hierarchical_bootstrap_auc(
                baselines["y"], setting_arrays["none"], baselines["predicted_scores"],
                baselines["dids"], n_boot=args.n_boot, seed=1701)
            deploy_perm = paired_cluster_permutation_auc(
                baselines["y"], setting_arrays["none"], baselines["predicted_scores"],
                baselines["dids"], n_perm=args.n_perm, seed=1701)
            oracle_ci = hierarchical_bootstrap_auc(
                baselines["y"], setting_arrays["oracle"], baselines["gold_scores"],
                baselines["dids"], n_boot=args.n_boot, seed=2701)
            oracle_perm = paired_cluster_permutation_auc(
                baselines["y"], setting_arrays["oracle"], baselines["gold_scores"],
                baselines["dids"], n_perm=args.n_perm, seed=2701)
            model_out["deployable_test"] = {**deploy_ci, **deploy_perm}
            model_out["oracle_test"] = {**oracle_ci, **oracle_perm}
            ds_out["models"][model] = model_out
            print(f"[{dataset}] {model}: deploy Δ={deploy_ci['delta_auc']:+.3f}, "
                  f"p={deploy_perm['p_value']:.4g}; oracle Δ={oracle_ci['delta_auc']:+.3f}, "
                  f"p={oracle_perm['p_value']:.4g}", flush=True)
        results[dataset] = ds_out
        (cache_root / dataset / "aggregate.json").write_text(
            json.dumps(jsonable(ds_out), indent=2), encoding="utf-8")
    return results


def add_holm(results: dict):
    """One Holm family across both information regimes (72 = 36 deployable + 36 oracle).

    A per-regime split would silently halve the effective family size and
    under-correct every adjusted p-value (m=36 instead of m=72).
    """
    pvals = {f"{family}:{ds}:{model}": row[key]["p_value"]
             for family, key in (("deployable", "deployable_test"), ("oracle", "oracle_test"))
             for ds, d in results.items() for model, row in d["models"].items()}
    corrected = holm_bonferroni(pvals)
    for compound, correction in corrected.items():
        family, ds, model = compound.split(":", 2)
        key = "deployable_test" if family == "deployable" else "oracle_test"
        results[ds]["models"][model][key]["holm"] = correction


def fmt(mean: float, std: float) -> str:
    return f"{mean:.3f}±{std:.3f}" if std > 0 else f"{mean:.3f}"


def write_outputs(results: dict, out_json: str, out_md: str):
    Path(out_json).write_text(json.dumps(jsonable(results), indent=2), encoding="utf-8")
    models = next(iter(results.values()))["models"].keys()
    lines = [
        "# IEEE Access information-matched and tuned revision experiment",
        "",
        "Every model receives the same four-trial validation search budget and uses the "
        "validation-selected checkpoint. Values are mean±SD over training seeds. Statistical "
        "intervals use a seed × dialogue hierarchical bootstrap; p-values use a paired "
        "dialogue-cluster permutation with a plus-one correction.",
        "",
        "## Deployable setting: no gold current emotion",
        "",
        "The transition baseline uses a train-only ERC prediction of the current label.",
        "",
        "| dataset | predicted transition | " + " | ".join(models) + " |",
        "|---|" + "---:|" * (len(models) + 1),
    ]
    for ds, row in results.items():
        p = row["predicted_transition"]["metrics"]["shift_auc"]
        vals = [fmt(p["mean"], p["std"])]
        for model in models:
            m = row["models"][model]["none"]["metrics"]["shift_auc"]
            vals.append(fmt(m["mean"], m["std"]))
        lines.append(f"| {ds} | " + " | ".join(vals) + " |")
    lines += ["", "## Oracle information-matched diagnostic", "",
              "Both model and transition baseline receive gold current emotion.", "",
              "| dataset | gold transition | " + " | ".join(models) + " |",
              "|---|" + "---:|" * (len(models) + 1)]
    for ds, row in results.items():
        vals = [f"{row['gold_transition']['shift_auc']:.3f}"]
        for model in models:
            m = row["models"][model]["oracle"]["metrics"]["shift_auc"]
            vals.append(fmt(m["mean"], m["std"]))
        lines.append(f"| {ds} | " + " | ".join(vals) + " |")

    lines += ["", "## Seed- and dialogue-aware inference", "",
              "| regime | dataset | model | ΔAUC | 95% CI | permutation p | Holm p | Holm significant |",
              "|---|---|---|---:|---:|---:|---:|---|"]
    for regime, key in (("deployable", "deployable_test"), ("oracle", "oracle_test")):
        for ds, row in results.items():
            for model in models:
                r = row["models"][model][key]
                lines.append(f"| {regime} | {ds} | {model} | {r['delta_auc']:+.3f} | "
                             f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | "
                             f"{r['p_value']:.4g} | {r['holm']['p_holm']:.4g} | "
                             f"{'yes' if r['holm']['significant_holm'] else 'no'} |")
    lines += ["", "## Additional metrics", "",
              "Complete ROC-AUC, PR-AUC, Brier, ECE, balanced accuracy, F1, prediction-rate, "
              "ERC diagnostics, selected configurations, training curves, and per-seed runs "
              f"are stored in `{out_json}` and the cache namespace."]
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=ALL_KEYS, choices=ALL_KEYS)
    ap.add_argument("--models", nargs="+", default=MODELS,
                    choices=sorted(set(MODELS + list(MODEL_ALIASES))))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--search-limit", type=int, default=len(SEARCH_SPACE),
                    choices=range(1, len(SEARCH_SPACE) + 1))
    ap.add_argument("--n-boot", type=int, default=1999)
    ap.add_argument("--n-perm", type=int, default=4999)
    ap.add_argument("--cache-dir", default="results/cache_access_revision")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out-json", default="results/access_revision_experiments.json")
    ap.add_argument("--out-md", default="results/access_revision_experiments.md")
    args = ap.parse_args()
    args.models = [canonical_model(m) for m in args.models]
    # Avoid duplicate aliases while preserving order.
    args.models = list(dict.fromkeys(args.models))
    results = run_all(args)
    add_holm(results)
    write_outputs(results, args.out_json, args.out_md)
    print(f"Wrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
