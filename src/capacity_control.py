"""
Capacity-matched control for the leakage-gap measurement.

The raw "leakage gap" (leaky bidirectional GRU AUC minus safe causal GRU AUC) conflates two
effects: the leaky model can see future utterances, but it also has roughly twice the
parameters of the safe GRU at the same hidden size (bidirectional recurrence doubles the
weight matrices). This isolates the two: gru_wide is a CAUSAL GRU with hidden size 240,
matched to the leaky model's parameter count within 3% (see CAPACITY_MATCHED_HIDDEN in
models.py). If the capacity-matched causal model still trails the leaky model by roughly the
same margin as the plain GRU did, the gap is leakage rather than capacity.

Run:    python -m src.capacity_control
Writes: results/capacity_control.md
"""
from __future__ import annotations

import json
import os

from .dataset import load_cosmic
from .evaluate import summarize_seeds
from .models import CAPACITY_MATCHED_HIDDEN
from .train import train_one

COSMIC = {
    "iemocap": "data/feat/iemocap_features_roberta.pkl",
    "meld": "data/feat/meld_features_roberta.pkl",
    "emorynlp": "data/feat/emorynlp_features_roberta.pkl",
    "dailydialog": "data/feat/dailydialog_features_roberta.pkl",
}
CACHE_DIR = "results/cache_capacity"


def run_one_dataset(ds: str, path: str, seeds=(0, 1, 2), epochs=20) -> dict:
    split = load_cosmic(path, ds, "roberta1")
    row = {}
    for model in ("gru", "gru_wide", "gru_leaky"):
        per_seed = []
        for s in seeds:
            m, _ = train_one(split, model, "focal", "none", seed=s, epochs=epochs)
            per_seed.append({k: v for k, v in m.items()
                             if isinstance(v, (int, float)) and not isinstance(v, bool)})
        row[model] = list(summarize_seeds(per_seed)["shift_auc"])
        print(f"[{ds}] {model:10s} AUC={row[model][0]:.3f}±{row[model][1]:.3f}", flush=True)
    row["capacity_gap"] = row["gru_wide"][0] - row["gru"][0]
    row["leakage_gap_raw"] = row["gru_leaky"][0] - row["gru"][0]
    row["leakage_gap_controlled"] = row["gru_leaky"][0] - row["gru_wide"][0]
    print(f"  capacity_gap={row['capacity_gap']:+.3f}  "
          f"leakage_gap_controlled={row['leakage_gap_controlled']:+.3f}", flush=True)
    return row


def run(seeds=(0, 1, 2), epochs=20):
    """Per-dataset results are cached immediately, so an interrupted run resumes cleanly."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    results = {}
    for ds, path in COSMIC.items():
        cache = f"{CACHE_DIR}/{ds}.json"
        if os.path.exists(cache):
            print(f"[{ds}] (cached)", flush=True)
            results[ds] = json.load(open(cache))
            continue
        row = run_one_dataset(ds, path, seeds, epochs)
        json.dump(row, open(cache, "w"), indent=2)
        results[ds] = row
    return results


def write(results: dict, path="results/capacity_control.md"):
    lines = ["# Capacity-matched leakage control", "",
             f"gru_wide is a causal GRU with hidden size {CAPACITY_MATCHED_HIDDEN} "
             "(~912k parameters, matching the leaky bidirectional GRU's ~887k within 3%).",
             "",
             "| dataset | GRU (h=128) | GRU, matched capacity | leaky (bidirectional) | "
             "capacity-only gap | raw leakage gap | capacity-controlled gap |",
             "|---|---|---|---|---|---|---|"]
    for ds, r in results.items():
        lines.append(f"| {ds} | {r['gru'][0]:.3f} | {r['gru_wide'][0]:.3f} | {r['gru_leaky'][0]:.3f} "
                     f"| {r['capacity_gap']:+.3f} | {r['leakage_gap_raw']:+.3f} "
                     f"| {r['leakage_gap_controlled']:+.3f} |")
    avg_raw = sum(r["leakage_gap_raw"] for r in results.values()) / len(results)
    avg_ctrl = sum(r["leakage_gap_controlled"] for r in results.values()) / len(results)
    lines += ["", f"Mean raw leakage gap: {avg_raw:+.3f}. Mean capacity-controlled gap: {avg_ctrl:+.3f}."]
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    results = run()
    write(results)
    print("Wrote results/capacity_control.md")


if __name__ == "__main__":
    main()
