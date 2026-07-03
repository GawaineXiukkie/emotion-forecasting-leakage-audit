"""
MELD feature-source ablation (paper Table III): safe GRU, shift-AUC, across four MM-DFN
modality configurations plus the COSMIC RoBERTa text baseline for comparison.

Run:    python -m src.feature_ablation
Writes: results/feature_ablation_meld.md
"""
from __future__ import annotations

import json
import os

from .dataset import load_cosmic, load_mmdfn
from .evaluate import summarize_seeds
from .train import train_one

MMDFN_PATH = "data/feat/MELD_features_raw1.pkl"
COSMIC_PATH = "data/feat/meld_features_roberta.pkl"
MODALITY_CONFIGS = [
    ("MM-DFN text-only", ("text",)),
    ("MM-DFN text+audio", ("text", "audio")),
    ("MM-DFN text+visual", ("text", "visual")),
    ("MM-DFN text+audio+visual", ("text", "audio", "visual")),
]
CACHE_DIR = "results/cache_feature_ablation"


def run_config(name: str, modalities: tuple, seeds=(0, 1, 2), epochs=20) -> list:
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = "_".join(modalities)
    cache = f"{CACHE_DIR}/{key}.json"
    if os.path.exists(cache):
        print(f"[{name}] (cached)", flush=True)
        return json.load(open(cache))

    split = load_mmdfn(MMDFN_PATH, "meld", modalities)
    per_seed = []
    for s in seeds:
        m, _ = train_one(split, "gru", "focal", "none", seed=s, epochs=epochs)
        per_seed.append({k: v for k, v in m.items()
                         if isinstance(v, (int, float)) and not isinstance(v, bool)})
    auc = list(summarize_seeds(per_seed)["shift_auc"])
    print(f"[{name}] AUC={auc[0]:.3f}±{auc[1]:.3f}", flush=True)
    json.dump(auc, open(cache, "w"), indent=2)
    return auc


def cosmic_text_baseline(seeds=(0, 1, 2), epochs=20) -> list:
    cache = f"{CACHE_DIR}/cosmic_text.json"
    if os.path.exists(cache):
        print("[COSMIC RoBERTa text] (cached)", flush=True)
        return json.load(open(cache))
    split = load_cosmic(COSMIC_PATH, "meld", "roberta1")
    per_seed = []
    for s in seeds:
        m, _ = train_one(split, "gru", "focal", "none", seed=s, epochs=epochs)
        per_seed.append({k: v for k, v in m.items()
                         if isinstance(v, (int, float)) and not isinstance(v, bool)})
    auc = list(summarize_seeds(per_seed)["shift_auc"])
    print(f"[COSMIC RoBERTa text] AUC={auc[0]:.3f}±{auc[1]:.3f}", flush=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    json.dump(auc, open(cache, "w"), indent=2)
    return auc


def main():
    rows = [("COSMIC RoBERTa text (1024-d)", cosmic_text_baseline())]
    dims = {"MM-DFN text-only": "600-d", "MM-DFN text+audio": "600+300-d",
           "MM-DFN text+visual": "600+342-d", "MM-DFN text+audio+visual": "600+300+342-d"}
    for name, modalities in MODALITY_CONFIGS:
        rows.append((f"{name} ({dims[name]})", run_config(name, modalities)))

    lines = ["# MELD feature-source ablation (paper Table III)", "",
             "Safe GRU, shift-AUC. The gain over RoBERTa text is carried entirely by MM-DFN's "
             "text channel; audio/visual channels do not improve over MM-DFN text-only in "
             "this ablation.", "",
             "| Features | AUC (none) |", "|---|---|"]
    for name, auc in rows:
        lines.append(f"| {name} | {auc[0]:.3f} |")
    lines += ["", "Reported in the paper as a hypothesis, not a conclusion. Two candidate "
             "explanations for the gain, neither adjudicated between:",
             "- MM-DFN's 600-d text encoding is task-adapted (trained on MELD-adjacent data), "
             "while COSMIC's 1024-d RoBERTa embedding is frozen and generic -- a task-adapted, "
             "lower-dimensional representation could simply carry a cleaner shift signal.",
             "- The two pipelines' text preprocessing was never verified to be identical "
             "(tokenization, speaker-tag handling, punctuation normalization); a preprocessing "
             "difference could explain some or all of the gap independent of the embedding "
             "method.",
             "",
             "We did not run a controlled re-extraction to distinguish these."]
    open("results/feature_ablation_meld.md", "w").write("\n".join(lines) + "\n")
    print("Wrote results/feature_ablation_meld.md")


if __name__ == "__main__":
    main()
