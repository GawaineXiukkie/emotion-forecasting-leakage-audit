"""Re-run the two EFC-style baselines after implementation corrections.

The historical ``pec`` and ``pseudofuture`` names remain reproducible.  This
script evaluates ``pec_fixed`` (recent utterances receive the largest recency
weight) and ``pseudofuture_fixed`` (padding excluded from auxiliary MSE) without
overwriting any result reported in the original manuscript.

Run:
    python -m src.access_fixed_efc [--datasets ...] [--epochs 20]
"""
from __future__ import annotations

import argparse
import json
import os

from .baselines import collect_shift_arrays
from .evaluate import paired_bootstrap_auc, summarize_seeds
from .experiments import ALL_KEYS, load_split
from .baselines import SpeakerTransitionMatrix
from .train import train_one

MODELS = ["pec_fixed", "pseudofuture_fixed"]


def run_dataset(ds: str, seeds: list[int], epochs: int, force: bool) -> dict:
    os.makedirs("results/cache_access_fixed", exist_ok=True)
    cache = f"results/cache_access_fixed/{ds}.json"
    if os.path.exists(cache) and not force:
        print(f"[{ds}] cached", flush=True)
        with open(cache, encoding="utf-8") as f:
            return json.load(f)

    split = load_split(ds)
    test_y = collect_shift_arrays(split.test)[2]
    transition = SpeakerTransitionMatrix(split.num_emotions).fit(split.train).predict_score(split.test)
    out = {}
    for model in MODELS:
        seed_rows = []
        seed0_scores = seed0_dids = None
        for seed in seeds:
            metrics, extra = train_one(split, model, "focal", "none", seed,
                                       epochs=epochs, compute_ci=False)
            seed_rows.append({k: v for k, v in metrics.items()
                              if isinstance(v, (int, float)) and not isinstance(v, bool)})
            if seed == seeds[0]:
                seed0_scores, seed0_dids = extra["scores"], extra["dids"]
        agg = summarize_seeds(seed_rows)
        paired = paired_bootstrap_auc(test_y, seed0_scores, transition, seed0_dids)
        out[model] = {
            "auc": list(agg["shift_auc"]),
            "f1": list(agg["shift_f1"]),
            "balanced_accuracy": list(agg["balanced_acc"]),
            "paired_vs_transition": paired,
        }
        print(f"[{ds}] {model}: AUC {out[model]['auc'][0]:.3f}±"
              f"{out[model]['auc'][1]:.3f}; delta {paired['delta_auc']:+.3f}", flush=True)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def write_markdown(results: dict):
    lines = [
        "# Corrected EFC-style baselines",
        "",
        "Three seeds, fixed 20-epoch budget. Historical results are preserved separately.",
        "",
        "| dataset | PEC-fixed AUC | PseudoFuture-fixed AUC | transition AUC |",
        "|---|---:|---:|---:|",
    ]
    for ds, rows in results.items():
        p, q = rows["pec_fixed"], rows["pseudofuture_fixed"]
        with open(f"results/cache/{ds}.json", encoding="utf-8") as f:
            trans = json.load(f)["rows"]["baseline:speaker_transition"]["auc"][0]
        lines.append(f"| {ds} | {p['auc'][0]:.3f}±{p['auc'][1]:.3f} | "
                     f"{q['auc'][0]:.3f}±{q['auc'][1]:.3f} | {trans:.3f} |")
    with open("results/access_fixed_efc.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=ALL_KEYS, choices=ALL_KEYS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    results = {ds: run_dataset(ds, args.seeds, args.epochs, args.force)
               for ds in args.datasets}
    with open("results/access_fixed_efc.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    write_markdown(results)
    print("Wrote results/access_fixed_efc.{json,md}")


if __name__ == "__main__":
    main()
