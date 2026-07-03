"""
Leakage dose-response curve: shift-AUC as a function of k, the number of future
utterances a model is allowed to see (k in {0, 1, 2, 4, infinity}).

k=0 and k=infinity are read from existing results rather than retrained (results/cache/ for
gru:none and gru_leaky, results/cache_capacity/ as a fallback for the four text datasets,
which also store those two models). Only k in {1, 2, 4} are trained here, via
models.LookaheadGRU -- a causal GRU with an explicit mean-pooled peek at the next k
utterances, vectorized so it stays fast on MPS even for the larger datasets.

Run:    python -m src.dose_response
Writes: results/dose_response.md, results/dose_response.json,
        results/figures/leakage_dose_response.png
"""
from __future__ import annotations

import json
import os

from .dataset import load_cosmic, load_mmdfn
from .evaluate import summarize_seeds
from .train import train_one

COSMIC = {
    "iemocap": "data/feat/iemocap_features_roberta.pkl",
    "meld": "data/feat/meld_features_roberta.pkl",
    "emorynlp": "data/feat/emorynlp_features_roberta.pkl",
    "dailydialog": "data/feat/dailydialog_features_roberta.pkl",
}
MMDFN = {
    "iemocap_mm": ("data/feat/IEMOCAP_features.pkl", "iemocap"),
    "meld_mm": ("data/feat/MELD_features_raw1.pkl", "meld"),
}
ALL_DATASETS = list(COSMIC) + list(MMDFN)
K_VALUES = [1, 2, 4]
CACHE_DIR = "results/cache_dose_response"


def load_split(name: str):
    if name in COSMIC:
        return load_cosmic(COSMIC[name], name, "roberta1")
    path, ds = MMDFN[name]
    return load_mmdfn(path, ds, ("text", "audio", "visual"))


def endpoint_from_cache(name: str) -> dict | None:
    """Read the k=0 (gru:none) and k=infinity (gru_leaky) endpoints from whichever cache has them."""
    for path, k0_key, kinf_key in [
        (f"results/cache/{name}.json", "gru:none", "gru_leaky:none"),
        (f"results/cache_capacity/{name}.json", "gru", "gru_leaky"),
    ]:
        if os.path.exists(path):
            r = json.load(open(path))
            rows = r.get("rows", r)
            k0 = rows.get(k0_key); kinf = rows.get(kinf_key)
            if k0 and kinf:
                k0v = k0["auc"] if isinstance(k0, dict) else k0
                kinfv = kinf["auc"] if isinstance(kinf, dict) else kinf
                trans = rows.get("baseline:speaker_transition", {}).get("auc", [None])[0] \
                    if "baseline:speaker_transition" in rows else None
                return {"k0": list(k0v), "kinf": list(kinfv), "transition": trans}
    return None


def run_dataset(name: str, seeds=(0, 1, 2), epochs=20) -> dict:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = f"{CACHE_DIR}/{name}.json"
    if os.path.exists(cache):
        print(f"[{name}] (cached)", flush=True)
        return json.load(open(cache))

    split = load_split(name)
    endpoints = endpoint_from_cache(name)
    if endpoints is None:
        raise RuntimeError(f"No cached k=0/k=infinity endpoints for {name} -- run "
                           f"src.experiments and src.capacity_control for this dataset first.")
    curve = {0: endpoints["k0"], "inf": endpoints["kinf"]}
    for k in K_VALUES:
        per_seed = []
        for s in seeds:
            m, _ = train_one(split, f"gru_look{k}", "focal", "none", seed=s, epochs=epochs)
            per_seed.append({key: v for key, v in m.items()
                             if isinstance(v, (int, float)) and not isinstance(v, bool)})
        curve[k] = list(summarize_seeds(per_seed)["shift_auc"])
        print(f"[{name}] k={k}  AUC={curve[k][0]:.3f}±{curve[k][1]:.3f}", flush=True)
    curve["transition"] = endpoints["transition"]
    json.dump(curve, open(cache, "w"), indent=2)
    return curve


def make_plot(all_curves: dict, path="results/figures/leakage_dose_response.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    x_labels = ["0", "1", "2", "4", "∞ (bidir)"]
    x_pos = [0, 1, 2, 4, 6]
    colors = plt.cm.tab10.colors
    for i, (ds, curve) in enumerate(all_curves.items()):
        ys = [curve[str(k)][0] if str(k) in curve else curve[k][0] for k in [0, 1, 2, 4]]
        es = [curve[str(k)][1] if str(k) in curve else curve[k][1] for k in [0, 1, 2, 4]]
        y_inf, e_inf = curve["inf"]
        ys.append(y_inf); es.append(e_inf)
        ax.errorbar(x_pos, ys, yerr=es, marker="o", label=ds, color=colors[i % 10], capsize=3)
        if curve.get("transition") is not None:
            ax.axhline(curve["transition"], color=colors[i % 10], linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xticks(x_pos); ax.set_xticklabels(x_labels)
    ax.set_xlabel("k = number of future utterances visible to the model")
    ax.set_ylabel("shift-AUC")
    ax.set_title("Leakage dose-response: each future utterance manufactures phantom AUC\n"
                "(dotted lines = speaker-transition-matrix baseline per dataset)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Wrote {path}")


def write_table(all_curves: dict, path="results/dose_response.md"):
    lines = ["# Leakage dose-response", "",
             "Shift-AUC as a function of k, the number of future utterances visible to the "
             "model. k=0 and k=infinity are read from existing results; k in {1, 2, 4} use "
             "LookaheadGRU, a causal GRU with a mean-pooled peek at the next k utterances.",
             "", "![dose-response](figures/leakage_dose_response.png)", "",
             "| dataset | transition | k=0 | k=1 | k=2 | k=4 | k=infinity (bidirectional) |",
             "|---|---|---|---|---|---|---|"]
    for ds, c in all_curves.items():
        def fmt(k):
            v = c[str(k)] if str(k) in c else c[k]
            return f"{v[0]:.3f}±{v[1]:.3f}"
        trans = f"{c['transition']:.3f}" if c.get("transition") is not None else "n/a"
        lines.append(f"| {ds} | {trans} | {fmt(0)} | {fmt(1)} | {fmt(2)} | {fmt(4)} | {fmt('inf')} |")
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    all_curves = {}
    for ds in ALL_DATASETS:
        all_curves[ds] = run_dataset(ds)
    json.dump(all_curves, open("results/dose_response.json", "w"), indent=2)
    write_table(all_curves)
    make_plot(all_curves)
    print("Wrote results/dose_response.md, results/figures/leakage_dose_response.png")


if __name__ == "__main__":
    main()
