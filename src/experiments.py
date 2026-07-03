"""
Main benchmark harness: one protocol, one metric set, applied to every dataset and model.

Produces:
  results/benchmark_table.md   methods x datasets: shift-AUC/F1/balanced-acc, degenerate
                               flag, paired test vs. the transition matrix, leaky-vs-safe gap.
  results/leakage_audit.md     automated 8-item audit per dataset.

Methods: base_rate, no_change (inertia), speaker-transition matrix, text-history MLP,
         causal GRU (none / predicted / oracle), and a leaky bidirectional GRU used only to
         quantify how much future-peeking inflates AUC.

Run:  python -m src.experiments
      python -m src.experiments --datasets iemocap meld
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .dataset import load_cosmic, load_mmdfn
from .baselines import collect_shift_arrays
from .evaluate import paired_bootstrap_auc, summarize_seeds
from .leakage_audit import audit, write_audit
from .train import fill_predicted_current_emotion, run_baselines, train_one

# text-only (COSMIC RoBERTa-1)
COSMIC = {
    "iemocap": "data/feat/iemocap_features_roberta.pkl",
    "meld": "data/feat/meld_features_roberta.pkl",
    "emorynlp": "data/feat/emorynlp_features_roberta.pkl",
    "dailydialog": "data/feat/dailydialog_features_roberta.pkl",
}
# multimodal family (MM-DFN: text + OpenSmile audio + visual)
MMDFN = {
    "iemocap_mm": ("data/feat/IEMOCAP_features.pkl", "iemocap"),
    "meld_mm": ("data/feat/MELD_features_raw1.pkl", "meld"),
}
ALL_KEYS = list(COSMIC) + list(MMDFN)
MODEL_RUNS = [("gru", "none"), ("gru", "predicted"), ("gru", "oracle"), ("gru_leaky", "none"),
              ("dialoguernn", "none"), ("dialoguegcn", "none"), ("dagerc", "none")]


def load_split(key: str):
    if key in COSMIC:
        return load_cosmic(COSMIC[key], key, "roberta1")
    path, ds = MMDFN[key]
    return load_mmdfn(path, ds, ("text", "audio", "visual"))


def run_dataset(name: str, seeds, epochs: int) -> dict:
    split = load_split(name)
    rep = audit(split, name, "gru")
    fill_predicted_current_emotion(split)
    test_y = collect_shift_arrays(split.test)[2]

    base_out, base_scores = run_baselines(split, return_scores=True)
    rows = {f"baseline:{k}": {"auc": [v["shift_auc"], 0.0], "f1": [v["shift_f1"], 0.0],
                              "bacc": [v["balanced_acc"], 0.0], "degen": v["degenerate"]}
            for k, v in base_out.items()}

    seed0_scores = {}
    for model, setting in MODEL_RUNS:
        per_seed, dids = [], None
        for s in seeds:
            m, ex = train_one(split, model, "focal", setting, seed=s, epochs=epochs)
            per_seed.append({k: v for k, v in m.items()
                             if isinstance(v, (int, float)) and not isinstance(v, bool)})
            if s == seeds[0]:
                seed0_scores[(model, setting)] = ex["scores"]; dids = ex["dids"]
        agg = summarize_seeds(per_seed)
        rows[f"{model}:{setting}"] = {"auc": list(agg["shift_auc"]), "f1": list(agg["shift_f1"]),
                                      "bacc": list(agg["balanced_acc"]),
                                      "degen": bool(per_seed[0]["frac_pred_shift"] > 0.98 or
                                                    per_seed[0]["frac_pred_shift"] < 0.02)}

    # paired test: safe GRU(none) vs speaker-transition matrix (cluster-robust on dialogues)
    paired = paired_bootstrap_auc(test_y, seed0_scores[("gru", "none")],
                                  base_scores["speaker_transition"], dids)
    # leakage gap: leaky bidir GRU AUC - safe GRU AUC (both 'none')
    leak_gap = rows["gru_leaky:none"]["auc"][0] - rows["gru:none"]["auc"][0]
    return {"rows": rows, "audit": rep, "paired_vs_transition": paired,
            "leakage_gap_auc": leak_gap, "base_rate": float(test_y.mean())}


def write_table(results: dict, path="results/benchmark_table.md"):
    os.makedirs("results", exist_ok=True)
    method_order = ["baseline:base_rate", "baseline:no_change", "baseline:speaker_transition",
                    "baseline:text_history_mlp", "gru:none", "gru:predicted", "gru:oracle",
                    "dialoguernn:none", "dialoguegcn:none", "dagerc:none",
                    "pec:none", "pseudofuture:none", "gru_leaky:none"]
    L = ["# Benchmark: emotion-shift forecasting under a leakage-safe protocol",
         "", "Metric = test **shift-AUC** (mean±std over seeds; threshold-free). "
         "`*` flags a degenerate (near-constant) predictor. COSMIC RoBERTa-1 features.", ""]
    for ds, r in results.items():
        L += [f"## {ds}  (shift base rate = {r['base_rate']:.3f})", "",
              "| method | shift-AUC | shift-F1 | balanced-acc |", "|---|---|---|---|"]
        for mth in method_order:
            if mth not in r["rows"]:
                continue
            v = r["rows"][mth]
            degen = " *" if v["degen"] else ""
            auc = f"{v['auc'][0]:.3f}±{v['auc'][1]:.3f}" if v["auc"][1] else f"{v['auc'][0]:.3f}"
            L.append(f"| {mth} | {auc}{degen} | {v['f1'][0]:.3f} | {v['bacc'][0]:.3f} |")
        p = r["paired_vs_transition"]
        L += ["",
              f"- **GRU(none) vs transition matrix** (paired, cluster-robust): "
              f"ΔAUC = {p['delta_auc']:+.3f} [{p['ci_low']:+.3f}, {p['ci_high']:+.3f}], "
              f"p={p['p_value']:.3f} → {'BEATS baseline' if p['beats_baseline'] else 'does NOT beat baseline'}",
              f"- **Leakage gap** (leaky bidir-GRU − safe GRU, AUC): {r['leakage_gap_auc']:+.3f} "
              f"(how much future-peeking inflates the score)", ""]
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=ALL_KEYS, choices=ALL_KEYS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    args = ap.parse_args()
    os.makedirs("results/cache", exist_ok=True)

    results = {}
    for ds in args.datasets:
        cache = f"results/cache/{ds}.json"
        if os.path.exists(cache) and not args.force:
            results[ds] = json.load(open(cache)); print(f"=== {ds} (cached) ===", flush=True)
        else:
            print(f"=== {ds} ===", flush=True)
            r = run_dataset(ds, args.seeds, args.epochs)
            json.dump(r, open(cache, "w"), indent=2)   # save immediately -> crash-safe / resumable
            results[ds] = r
        p = results[ds]["paired_vs_transition"]
        print(f"  GRU(none) vs transition ΔAUC={p['delta_auc']:+.3f} "
              f"[{p['ci_low']:+.3f},{p['ci_high']:+.3f}] beats={p['beats_baseline']}; "
              f"leak_gap={results[ds]['leakage_gap_auc']:+.3f}", flush=True)
        # rewrite outputs after EVERY dataset so partial progress always persists
        write_table(results)
        write_audit([results[d]["audit"] for d in results])
        json.dump({k: {"paired": v["paired_vs_transition"], "leak_gap": v["leakage_gap_auc"]}
                   for k, v in results.items()}, open("results/summary.json", "w"), indent=2)
    print("Wrote results/benchmark_table.md, results/leakage_audit.md, results/summary.json")


if __name__ == "__main__":
    main()
