"""
ERC emotion labels -> shift labels, with logging.

Rule:
    For each dialogue, over consecutive utterances u_1..u_T with emotion labels y_1..y_T,
        shift[n] = 1[y_{n+1} != y_n]   for n = 1..T-1
    The last utterance has no successor and is excluded from training/eval (IGNORE_INDEX).
    Speakers are retained so the speaker-specific transition-matrix baseline can condition
    on speaker; the shift label itself is over consecutive dialogue utterances (n+1 may be a
    different speaker — that is the realistic conversational setting).

Run:  python -m src.labels --features /path/to/iemocap_or_meld.pkl
Writes docs/label_conversion.md and prints per-split shift base rates.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .dataset import Dialogue, ShiftSplit, load_cosmic, shift_targets, IGNORE_INDEX


def split_stats(dialogues: list[Dialogue]) -> dict:
    n_utt = 0
    n_decision = 0          # positions with a valid successor
    n_shift = 0
    per_dialogue_len = []
    emo_counts: Counter = Counter()
    for d in dialogues:
        s = shift_targets(d.labels)
        valid = s != IGNORE_INDEX
        n_utt += len(d.labels)
        n_decision += int(valid.sum())
        n_shift += int((s[valid] == 1).sum())
        per_dialogue_len.append(len(d.labels))
        emo_counts.update(d.labels.tolist())
    return {
        "dialogues": len(dialogues),
        "utterances": n_utt,
        "decision_points": n_decision,
        "shift_positions": n_shift,
        "shift_base_rate": (n_shift / n_decision) if n_decision else float("nan"),
        "mean_dialogue_len": float(np.mean(per_dialogue_len)) if per_dialogue_len else 0.0,
        "emotion_distribution": dict(sorted(emo_counts.items())),
    }


def report(split: ShiftSplit) -> dict:
    return {name: split_stats(getattr(split, name)) for name in ("train", "val", "test")}


def write_doc(rep: dict, features_path: str, out: Path = Path("docs/label_conversion.md")):
    lines = [
        "# Label conversion log — ERC emotion → shift",
        "",
        f"Source features: `{features_path}`",
        "",
        "Rule: `shift[n] = 1[y_{n+1} != y_n]`; last utterance of each dialogue is IGNORE_INDEX.",
        "Shift label is over consecutive dialogue utterances; speakers retained for the",
        "speaker-specific transition-matrix baseline only.",
        "",
        "| split | dialogues | utterances | decision pts | shift pts | shift rate | mean len |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in rep.items():
        lines.append(f"| {name} | {s['dialogues']} | {s['utterances']} | {s['decision_points']} "
                     f"| {s['shift_positions']} | {s['shift_base_rate']:.3f} | {s['mean_dialogue_len']:.1f} |")
    lines += ["", "Per-split emotion distribution:", ""]
    for name, s in rep.items():
        lines.append(f"- **{name}**: {s['emotion_distribution']}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Build + log shift labels from cached ERC features.")
    ap.add_argument("--features", required=True, help="Path to the COSMIC roberta pickle.")
    ap.add_argument("--dataset", required=True, choices=["iemocap", "meld", "emorynlp", "dailydialog"])
    args = ap.parse_args()

    split = load_cosmic(args.features, args.dataset)
    rep = report(split)
    for name, s in rep.items():
        print(f"[{name}] dialogues={s['dialogues']} decision_pts={s['decision_points']} "
              f"shift_rate={s['shift_base_rate']:.3f}")
    write_doc(rep, args.features)
    print("Wrote docs/label_conversion.md")
    if any(rep[n]["shift_base_rate"] > 0.5 for n in rep):
        print("NOTE: shift is the majority class here — re-check class weighting in losses.")


if __name__ == "__main__":
    main()
