"""
Checks for exact-duplicate dialogues across train/test/val splits, and re-tests the
DailyDialog GRU-vs-transition-matrix result on a deduplicated split.

DailyDialog's official split turns out not to be fully disjoint at the text level, which
matters here because DailyDialog is the only text dataset where a safe model beats the
transition-matrix baseline -- if that win were partly explained by duplicate dialogues
leaking across the split, it wouldn't be a real result.

Run:    python -m src.check_duplicate_dialogues
Writes: results/duplicate_audit.md, results/dedup_dailydialog.json
"""
from __future__ import annotations

import json

from .baselines import collect_shift_arrays
from .dataset import find_exact_duplicate_dialogues, load_cosmic
from .evaluate import paired_bootstrap_auc, summarize_seeds
from .train import run_baselines, train_one

COSMIC = {
    "iemocap": "data/feat/iemocap_features_roberta.pkl",
    "meld": "data/feat/meld_features_roberta.pkl",
    "emorynlp": "data/feat/emorynlp_features_roberta.pkl",
    "dailydialog": "data/feat/dailydialog_features_roberta.pkl",
}


def write_duplicate_audit(path="results/duplicate_audit.md"):
    lines = ["# Duplicate-dialogue audit", "",
             "Checks whether any test/val dialogue's full utterance-text sequence exactly "
             "matches a train dialogue (`src/dataset.py::find_exact_duplicate_dialogues`).", "",
             "| dataset | test dialogues | test duplicates | test dup% | val dialogues | val duplicates | val dup% |",
             "|---|---|---|---|---|---|---|"]
    reports = {}
    for ds, path_ in COSMIC.items():
        r = find_exact_duplicate_dialogues(path_, ds)
        reports[ds] = r
        lines.append(f"| {ds} | {r['n_test']} | {len(r['test_dup_ids'])} | "
                     f"{r['test_dup_frac']*100:.1f}% | {r['n_val']} | {len(r['val_dup_ids'])} | "
                     f"{r['val_dup_frac']*100:.1f}% |")
    open(path, "w").write("\n".join(lines) + "\n")
    return reports


def rerun_dailydialog_decontaminated(seeds=(0, 1, 2), epochs=20):
    path = COSMIC["dailydialog"]
    results = {}
    for tag, decontam in [("original", False), ("decontaminated", True)]:
        split = load_cosmic(path, "dailydialog", "roberta1", decontaminate=decontam)
        test_y = collect_shift_arrays(split.test)[2]
        base_out, base_scores = run_baselines(split, return_scores=True)
        per_seed, seed0_scores, dids = [], None, None
        for s in seeds:
            m, ex = train_one(split, "gru", "focal", "none", seed=s, epochs=epochs)
            per_seed.append({k: v for k, v in m.items()
                             if isinstance(v, (int, float)) and not isinstance(v, bool)})
            if s == seeds[0]:
                seed0_scores, dids = ex["scores"], ex["dids"]
        agg = summarize_seeds(per_seed)
        paired = paired_bootstrap_auc(test_y, seed0_scores, base_scores["speaker_transition"], dids)
        results[tag] = {
            "n_test": len(split.test), "n_val": len(split.val),
            "transition_auc": base_out["speaker_transition"]["shift_auc"],
            "gru_auc": list(agg["shift_auc"]), "paired": paired,
        }
        print(f"[{tag}] n_test={len(split.test)} transition={base_out['speaker_transition']['shift_auc']:.3f} "
              f"GRU={agg['shift_auc'][0]:.3f}±{agg['shift_auc'][1]:.3f} "
              f"delta={paired['delta_auc']:+.3f} beats={paired['beats_baseline']}", flush=True)
    return results


def main():
    print("=== duplicate audit (four text datasets) ===")
    write_duplicate_audit()
    print("Wrote results/duplicate_audit.md")
    print("\n=== DailyDialog: original vs deduplicated ===")
    results = rerun_dailydialog_decontaminated()
    json.dump(results, open("results/dedup_dailydialog.json", "w"), indent=2)
    print("Wrote results/dedup_dailydialog.json")


if __name__ == "__main__":
    main()
