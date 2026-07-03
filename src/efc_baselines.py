"""
Trains the two EFC-inspired strategy baselines (PEC-style, Pseudo-future-style) and merges
them into the main benchmark, using the same protocol as the four ERC-derived architectures
(same fixed hyperparameter budget, same three-seed plus seed-0-paired-test procedure).

Everything else -- the transition matrix, GRU, the leaky control, the three representative
family models -- is reused from src/experiments.py's cache; only PEC-style and Pseudo-future-
style are trained here. Results are merged directly into results/cache/<ds>.json so
results/benchmark_table.md and the joint significance test pick them up without a full rerun.

Run:  python -m src.efc_baselines
Then: python -m src.experiments --force     # regenerate the table with the new rows
      python -m src.significance_tests      # extend the joint test to all six models
"""
from __future__ import annotations

import json
import os

from .baselines import collect_shift_arrays
from .evaluate import paired_bootstrap_auc, summarize_seeds
from .experiments import COSMIC, MMDFN, load_split
from .train import run_baselines, train_one

NEW_MODELS = ["pec", "pseudofuture"]
SEEDS = [0, 1, 2]
EPOCHS = 20
CACHE_DIR = "results/cache_efc_baselines"


def run_one(ds: str) -> dict:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = f"{CACHE_DIR}/{ds}.json"
    if os.path.exists(cache):
        print(f"[{ds}] (cached)", flush=True)
        return json.load(open(cache))

    split = load_split(ds)
    test_y = collect_shift_arrays(split.test)[2]
    _, base_scores = run_baselines(split, return_scores=True)
    trans_scores = base_scores["speaker_transition"]

    out = {}
    for model in NEW_MODELS:
        per_seed, seed0_scores, dids = [], None, None
        for s in SEEDS:
            m, ex = train_one(split, model, "focal", "none", seed=s, epochs=EPOCHS)
            per_seed.append({k: v for k, v in m.items()
                             if isinstance(v, (int, float)) and not isinstance(v, bool)})
            if s == SEEDS[0]:
                seed0_scores, dids = ex["scores"], ex["dids"]
        agg = summarize_seeds(per_seed)
        paired = paired_bootstrap_auc(test_y, seed0_scores, trans_scores, dids)
        out[model] = {
            "auc": list(agg["shift_auc"]), "f1": list(agg["shift_f1"]),
            "bacc": list(agg["balanced_acc"]),
            "degen": bool(per_seed[0]["frac_pred_shift"] > 0.98 or per_seed[0]["frac_pred_shift"] < 0.02),
            "paired_vs_transition": paired,
        }
        print(f"[{ds}] {model:14s} AUC={agg['shift_auc'][0]:.3f}±{agg['shift_auc'][1]:.3f} "
              f"delta={paired['delta_auc']:+.3f} p={paired['p_value']:.4f}", flush=True)

    json.dump(out, open(cache, "w"), indent=2)
    return out


def merge_into_main_cache(ds: str, result: dict):
    """Patch the new rows directly into results/cache/<ds>.json rather than rerunning the
    full benchmark for this dataset, which would needlessly retrain the other six models."""
    main_cache = f"results/cache/{ds}.json"
    if not os.path.exists(main_cache):
        print(f"[{ds}] no main cache found, skipping merge"); return
    main = json.load(open(main_cache))
    for model, r in result.items():
        main["rows"][f"{model}:none"] = {"auc": r["auc"], "f1": r["f1"], "bacc": r["bacc"],
                                         "degen": r["degen"]}
    json.dump(main, open(main_cache, "w"), indent=2)
    print(f"[{ds}] merged into {main_cache}")


def main():
    all_datasets = list(COSMIC) + list(MMDFN)
    results = {}
    for ds in all_datasets:
        results[ds] = run_one(ds)
        merge_into_main_cache(ds, results[ds])
    json.dump(results, open("results/efc_baselines.json", "w"), indent=2)
    print("Wrote results/efc_baselines.json and merged rows into results/cache/*.json")


if __name__ == "__main__":
    main()
