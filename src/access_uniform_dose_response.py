"""Leakage dose-response with one fixed Transformer architecture.

Only the attention mask changes across k in {0,1,2,4,all}; parameters,
positional encoding, optimization, pooling, and checkpoint rule are identical.

Single-layer by construction (``layers=1``): a stacked multi-layer encoder using
the same window mask at every layer lets information propagate transitively --
layer 2's query at position i can read layer 1's output at i+k, which already
mixed in x_{i+2k}, giving an effective receptive field of ~L*k for L layers
rather than k. One layer is required for k to mean what the sweep claims it
means (exactly k future positions visible).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
# Type 3 (bitmap) fonts embedded by the matplotlib default can trip PDF preflight
# checks; 42 embeds real (Type 42/TrueType) outlines instead.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np

from .access_revision_experiments import jsonable, load_revision_split
from .experiments import ALL_KEYS
from .train import train_one

WINDOWS = [0, 1, 2, 4, "all"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=ALL_KEYS, choices=ALL_KEYS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--out-json", default="results/access_uniform_dose_response.json")
    ap.add_argument("--out-md", default="results/access_uniform_dose_response.md")
    ap.add_argument("--figure-stem", default="access_uniform_dose_response")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cache = Path("results/cache_access_uniform_dose")
    cache.mkdir(parents=True, exist_ok=True)
    results = {}
    for ds in args.datasets:
        split = load_revision_split(ds)
        ds_rows = {}
        for k in WINDOWS:
            name = f"transformer_future{k}"
            per_seed = []
            for seed in args.seeds:
                folder = cache / ds; folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"k{k}_seed{seed}.json"
                if path.exists() and not args.force:
                    row = json.loads(path.read_text(encoding="utf-8"))
                else:
                    metrics, ex = train_one(
                        split, name, "focal", "none", seed=seed, epochs=args.epochs,
                        lr=1e-3, hidden=128, dropout=0.1, compute_ci=False,
                        early_stopping=True, patience=5, min_epochs=5, track_history=True,
                        model_kwargs={"layers": 1})
                    row = {"seed": seed, "metrics": metrics, "history": ex["history"],
                           "best_epoch": ex["best_epoch"],
                           "best_val_auc": ex["best_val_auc"]}
                    np.savez_compressed(folder / f"k{k}_seed{seed}.npz",
                                        scores=np.asarray(ex["scores"], dtype=np.float32),
                                        y=np.asarray(ex["y"], dtype=np.int8),
                                        dids=np.asarray(ex["dids"], dtype=str))
                    path.write_text(json.dumps(jsonable(row), indent=2), encoding="utf-8")
                per_seed.append(row)
            aucs = [r["metrics"]["shift_auc"] for r in per_seed]
            ds_rows[str(k)] = {"auc_mean": float(np.mean(aucs)),
                               "auc_std": float(np.std(aucs)), "runs": per_seed}
            print(ds, k, ds_rows[str(k)]["auc_mean"], flush=True)
        base = ds_rows["0"]["auc_mean"]
        for k in WINDOWS:
            ds_rows[str(k)]["delta_from_k0"] = ds_rows[str(k)]["auc_mean"] - base
        results[ds] = ds_rows

    Path(args.out_json).write_text(
        json.dumps(jsonable(results), indent=2), encoding="utf-8")
    lines = ["# Same-architecture leakage dose-response", "",
             "A single one-layer Transformer is used throughout; only the future-attention mask "
             "changes. One layer keeps the receptive field exactly k rather than the ~2k a stacked "
             "multi-layer encoder would give under a per-layer window mask.", "",
             "| dataset | k=0 | k=1 | k=2 | k=4 | full | full-k0 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for ds, row in results.items():
        vals = [row[str(k)]["auc_mean"] for k in WINDOWS]
        lines.append(f"| {ds} | " + " | ".join(f"{v:.3f}" for v in vals) +
                     f" | {vals[-1]-vals[0]:+.3f} |")
    Path(args.out_md).write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.6), sharex=True)
    labels = ["0", "1", "2", "4", "full"]
    for ax, ds in zip(axes.flat, args.datasets):
        means = [results[ds][str(k)]["auc_mean"] for k in WINDOWS]
        stds = [results[ds][str(k)]["auc_std"] for k in WINDOWS]
        ax.errorbar(range(5), means, yerr=stds, marker="o", lw=2.4, ms=6, capsize=4,
                    color="#1f5a99")
        ax.set_title(ds.replace("_", "-"), fontsize=12.5)
        ax.set_xticks(range(5), labels)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(axis="y", alpha=.25)
        ax.set_ylim(max(0.45, min(means) - .05), min(1.0, max(means) + .05))
    for ax in axes[:, 0]: ax.set_ylabel("Shift ROC-AUC", fontsize=12)
    for ax in axes[-1, :]: ax.set_xlabel("Visible future utterances", fontsize=12)
    fig.suptitle("Future-access dose response with architecture held fixed", fontsize=15)
    fig.tight_layout()
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    Path("paper/access_submission/manuscript/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(f"results/figures/{args.figure_stem}.png", dpi=220, bbox_inches="tight")
    if args.figure_stem == "access_uniform_dose_response":
        fig.savefig("paper/access_submission/manuscript/figures/leakage_uniform_transformer.pdf",
                    bbox_inches="tight")
    plt.close(fig)
    print("Wrote same-architecture dose-response results and figure")


if __name__ == "__main__":
    main()
