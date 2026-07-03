"""
Joint significance testing across all six models and six dataset configurations.

Runs a paired, cluster-robust bootstrap comparison of each model against the
speaker-transition-matrix baseline, then applies Holm-Bonferroni correction across the full
36-comparison family (6 models x 6 datasets) at once, since they jointly support a single
claim: that model choice does not change which side of the baseline a given dataset falls on.

Results already computed by other scripts (the four representative architectures via
src/experiments.py, the two EFC-inspired baselines via src/efc_baselines.py) are reused
rather than retrained, so this script is close to instant once those have run.

Run:    python -m src.significance_tests
Writes: results/significance_tests.md
Cache:  results/cache_significance/ (per dataset/model, resumable)
"""
from __future__ import annotations

import json
import os

from .baselines import collect_shift_arrays
from .evaluate import paired_bootstrap_auc
from .experiments import COSMIC, MMDFN, load_split
from .holm_correction import holm_bonferroni
from .train import run_baselines, train_one

FAMILY_MODELS = ["gru", "dialoguernn", "dialoguegcn", "dagerc", "pec", "pseudofuture"]
CACHE_DIR = "results/cache_significance"
SEED = 0
EPOCHS = 20


def existing_gru_result(ds: str) -> dict | None:
    """GRU's paired test is already computed by src/experiments.py; reuse it directly so the
    numbers here match results/benchmark_table.md exactly instead of drifting from a
    fresh retrain (training isn't perfectly deterministic on MPS)."""
    path = f"results/cache/{ds}.json"
    if os.path.exists(path):
        return json.load(open(path))["paired_vs_transition"]
    return None


def existing_efc_result(ds: str, model: str) -> dict | None:
    """PEC-style and Pseudo-future-style results are already computed by
    src/efc_baselines.py; reuse them rather than retraining."""
    path = f"results/cache_efc_baselines/{ds}.json"
    if os.path.exists(path):
        d = json.load(open(path))
        if model in d:
            return d[model]["paired_vs_transition"]
    return None


def test_one(ds: str) -> dict:
    """Return {model_name: paired_bootstrap_auc dict} for all six models on this dataset."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    out = {}
    gru_cached = existing_gru_result(ds)
    if gru_cached is not None:
        out["gru"] = gru_cached
        print(f"[{ds}] gru (reused) delta={gru_cached['delta_auc']:+.3f}", flush=True)

    for model in ("pec", "pseudofuture"):
        cached = existing_efc_result(ds, model)
        if cached is not None:
            out[model] = cached
            print(f"[{ds}] {model} (reused) delta={cached['delta_auc']:+.3f}", flush=True)

    new_models = [m for m in FAMILY_MODELS if m not in out]
    if not new_models:
        return out

    split = load_split(ds)
    test_y = collect_shift_arrays(split.test)[2]
    _, base_scores = run_baselines(split, return_scores=True)
    trans_scores = base_scores["speaker_transition"]

    for model in new_models:
        cache = f"{CACHE_DIR}/{ds}__{model}.json"
        if os.path.exists(cache):
            out[model] = json.load(open(cache))
            print(f"[{ds}] {model} (cached) delta={out[model]['delta_auc']:+.3f}", flush=True)
            continue
        _, ex = train_one(split, model, "focal", "none", seed=SEED, epochs=EPOCHS)
        paired = paired_bootstrap_auc(test_y, ex["scores"], trans_scores, ex["dids"])
        json.dump(paired, open(cache, "w"), indent=2)
        out[model] = paired
        print(f"[{ds}] {model} delta={paired['delta_auc']:+.3f} p={paired['p_value']:.4f}", flush=True)
    return out


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    all_datasets = list(COSMIC) + list(MMDFN)

    results = {}
    for ds in all_datasets:
        for model, paired in test_one(ds).items():
            results[(ds, model)] = paired

    pvals = {f"{ds}:{model}": r["p_value"] for (ds, model), r in results.items()}
    corrected = holm_bonferroni(pvals)
    n_total = len(corrected)

    lines = ["# Significance tests",
             "",
             f"Paired, cluster-robust (dialogue-level) bootstrap comparisons of each model "
             f"against the speaker-transition-matrix baseline, {len(FAMILY_MODELS)} models "
             f"({', '.join(FAMILY_MODELS)}) x 6 datasets = {n_total} comparisons, "
             "Holm-corrected jointly since they support one claim: model choice doesn't "
             "change whether a given dataset favors the baseline.",
             "",
             "| dataset | model | delta AUC | raw p | rank | Holm threshold | significant |",
             "|---|---|---|---|---|---|---|"]
    for key in sorted(corrected, key=lambda k: corrected[k]["rank"]):
        ds, model = key.split(":")
        c = corrected[key]
        r = results[(ds, model)]
        lines.append(f"| {ds} | {model} | {r['delta_auc']:+.3f} | {c['p']:.4f} | {c['rank']} | "
                     f"{c['holm_threshold']:.4f} | {'yes' if c['significant_holm'] else 'no'} |")

    n_sig = sum(1 for c in corrected.values() if c["significant_holm"])
    lines += ["", f"{n_sig}/{n_total} comparisons remain significant after joint Holm correction.",
             "", "Note: the paper's claims rest on effect size and per-comparison confidence "
             "intervals; the Holm-corrected column here is a supplementary check against "
             "multiple-comparison inflation."]
    os.makedirs("results", exist_ok=True)
    open("results/significance_tests.md", "w").write("\n".join(lines) + "\n")
    print(f"\n{n_sig}/{n_total} significant after correction.")
    print("Wrote results/significance_tests.md")


if __name__ == "__main__":
    main()
