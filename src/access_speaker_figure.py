"""Create the vector speaker-composition figure used by the IEEE Access paper."""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np

DATASETS = ["iemocap", "meld", "emorynlp", "dailydialog"]
LABELS = ["IEMOCAP", "MELD", "EmoryNLP", "DailyDialog"]
BLUE = "#1479B8"
GOLD = "#D39A21"
INK = "#222222"
GRID = "#D8D8D8"


def main():
    with open("results/access_speaker_analysis.json", encoding="utf-8") as f:
        data = json.load(f)
    same_frac = np.array([data[d]["immediate"]["same_speaker"]["fraction"] for d in DATASETS])
    same_rate = np.array([
        np.nan if data[d]["immediate"]["same_speaker"]["shift_rate"] is None
        else data[d]["immediate"]["same_speaker"]["shift_rate"] for d in DATASETS
    ])
    switch_rate = np.array([
        data[d]["immediate"]["speaker_switch"]["shift_rate"] for d in DATASETS
    ])

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8.0, "axes.titlesize": 9.0,
        "axes.labelsize": 8.0, "xtick.labelsize": 7.5, "ytick.labelsize": 8.0,
        # Type 3 (bitmap) fonts embedded by the matplotlib default can trip PDF
        # preflight checks; 42 embeds real (Type 42/TrueType) outlines instead.
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.55))
    y = np.arange(len(LABELS))

    ax = axes[0]
    ax.barh(y, same_frac, color=BLUE, edgecolor=INK, linewidth=0.45, label="Same speaker")
    ax.barh(y, 1 - same_frac, left=same_frac, color="white", hatch="////",
            edgecolor=GOLD, linewidth=0.7, label="Speaker switch")
    for i, v in enumerate(same_frac):
        ax.text(max(v / 2, 0.04), i, f"{100*v:.0f}%", ha="center", va="center",
                color="white" if v > 0.12 else INK, fontweight="bold", fontsize=7)
        ax.text(v + (1-v)/2, i, f"{100*(1-v):.0f}%", ha="center", va="center",
                color=INK, fontsize=7)
    ax.set_yticks(y, LABELS)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100"])
    ax.set_xlabel("Share of adjacent test decisions (%)")
    ax.set_title("(a) Next-speaker composition", loc="left", fontweight="bold")

    ax = axes[1]
    h = 0.34
    ax.barh(y - h/2, same_rate, height=h, color=BLUE, edgecolor=INK, linewidth=0.45,
            label="Same speaker")
    ax.barh(y + h/2, switch_rate, height=h, facecolor="white", hatch="////",
            edgecolor=GOLD, linewidth=0.7, label="Speaker switch")
    for i, v in enumerate(same_rate):
        if np.isfinite(v):
            ax.text(v + .015, i - h/2, f"{v:.3f}", va="center", fontsize=7, color=INK)
    for i, v in enumerate(switch_rate):
        ax.text(v + .015, i + h/2, f"{v:.3f}", va="center", fontsize=7, color=INK)
    ax.set_yticks(y, LABELS)
    ax.invert_yaxis()
    ax.set_xlim(0, .86)
    ax.set_xlabel("Observed emotion-shift rate")
    ax.set_title("(b) Shift rate by speaker relation", loc="left", fontweight="bold")
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 1.01), fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=2.4)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("paper/access_submission/manuscript/figures", exist_ok=True)
    fig.savefig("results/figures/access_speaker_relation.png", dpi=300, bbox_inches="tight")
    fig.savefig("paper/access_submission/manuscript/figures/speaker_relation.pdf",
                bbox_inches="tight")
    plt.close(fig)
    print("Wrote results/figures/access_speaker_relation.png and manuscript vector PDF")


if __name__ == "__main__":
    main()
