"""Data-quality and cross-source alignment audit for the IEEE Access revision.

Checks schema/length integrity, finite features, split overlap, exact dialogue
duplicates, label ranges, duplicate-text feature consistency, and alignment of the
COSMIC and MM-DFN versions of IEMOCAP/MELD. The audit never modifies source data.

Run: python -m src.access_data_quality
"""
from __future__ import annotations

import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .dataset import _COSMIC_IDX, _MMDFN_IDX
from .experiments import COSMIC, MMDFN


def norm_text(value) -> str:
    # Feature extraction is case- and whitespace-sensitive. Using a lowercased or
    # whitespace-normalized key creates false "same input" matches.
    return str(value)


def dialogue_key(sentences) -> str:
    joined = "\n".join(norm_text(s) for s in sentences)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def split_overlap(train, val, test):
    a, b, c = set(train), set(val), set(test)
    return {"train_val": len(a & b), "train_test": len(a & c), "val_test": len(b & c)}


def feature_profile(feature_dicts: list[dict], ids) -> dict:
    n_arrays = n_values = nonfinite = 0
    min_v, max_v = float("inf"), -float("inf")
    dims = set()
    for did in ids:
        for fd in feature_dicts:
            arr = np.asarray(fd[did])
            if arr.ndim != 2:
                raise AssertionError(f"{did}: expected 2-D feature array, got {arr.shape}")
            n_arrays += 1; n_values += arr.size
            nonfinite += int((~np.isfinite(arr)).sum())
            min_v = min(min_v, float(np.nanmin(arr)))
            max_v = max(max_v, float(np.nanmax(arr)))
            dims.add(int(arr.shape[1]))
    return {"arrays": n_arrays, "values": n_values, "nonfinite": nonfinite,
            "min": min_v, "max": max_v, "dims": sorted(dims)}


def duplicate_text_feature_check(sentences: dict, features: dict, ids) -> dict:
    tolerance = 1e-3  # allows harmless batch-shape floating-point variation
    first = {}
    repeat_pairs = inconsistent = 0
    max_abs_diff = 0.0
    for did in ids:
        text_rows = sentences[did]
        feat_rows = np.asarray(features[did])
        for text, vector in zip(text_rows, feat_rows):
            key = norm_text(text)
            if key in first:
                repeat_pairs += 1
                diff = float(np.max(np.abs(np.asarray(vector) - first[key])))
                max_abs_diff = max(max_abs_diff, diff)
                if diff > tolerance:
                    inconsistent += 1
            else:
                first[key] = np.asarray(vector).copy()
    return {"repeated_text_occurrences": repeat_pairs,
            "feature_inconsistent_occurrences": inconsistent,
            "max_abs_diff": max_abs_diff,
            "tolerance": tolerance}


def duplicate_dialogues(sentences: dict, train, val, test) -> dict:
    keys = {name: [dialogue_key(sentences[d]) for d in ids]
            for name, ids in (("train", train), ("val", val), ("test", test))}
    tr = set(keys["train"])
    return {
        "within_train_extra": sum(v - 1 for v in Counter(keys["train"]).values() if v > 1),
        "within_val_extra": sum(v - 1 for v in Counter(keys["val"]).values() if v > 1),
        "within_test_extra": sum(v - 1 for v in Counter(keys["test"]).values() if v > 1),
        "val_exact_train": sum(k in tr for k in keys["val"]),
        "test_exact_train": sum(k in tr for k in keys["test"]),
        "val_exact_train_rate": sum(k in tr for k in keys["val"]) / max(len(keys["val"]), 1),
        "test_exact_train_rate": sum(k in tr for k in keys["test"]) / max(len(keys["test"]), 1),
    }


def audit_cosmic(dataset: str, path: str) -> dict:
    obj = load_pickle(path); idx = _COSMIC_IDX[dataset]
    speakers, labels, sentences = obj[idx["speakers"]], obj[idx["labels"]], obj[idx["sent"]]
    train, val, test = list(obj[idx["train"]]), list(obj[idx["val"]]), list(obj[idx["test"]])
    feature_dicts = [obj[i] for i in idx["roberta"]]
    all_ids = train + val + test
    length_errors = []
    label_counts = Counter()
    for did in all_ids:
        lengths = [len(speakers[did]), len(labels[did]), len(sentences[did])]
        lengths += [len(fd[did]) for fd in feature_dicts]
        if len(set(lengths)) != 1:
            length_errors.append({"id": str(did), "lengths": lengths})
        label_counts.update(int(v) for v in labels[did])
    return {
        "source": "COSMIC", "dataset": dataset, "path": path,
        "dialogues": {"train": len(train), "val": len(val), "test": len(test)},
        "utterances": {"train": sum(len(labels[d]) for d in train),
                       "val": sum(len(labels[d]) for d in val),
                       "test": sum(len(labels[d]) for d in test)},
        "split_id_overlap": split_overlap(train, val, test),
        "length_error_count": len(length_errors), "length_error_examples": length_errors[:5],
        "label_counts": dict(sorted(label_counts.items())),
        "label_range_valid": set(label_counts).issubset(set(range(idx["n_emo"]))),
        "feature_profile": feature_profile(feature_dicts[:1], all_ids),
        "dialogue_duplicates": duplicate_dialogues(sentences, train, val, test),
        "duplicate_text_feature_consistency": duplicate_text_feature_check(
            sentences, feature_dicts[0], all_ids),
    }


def audit_mmdfn(key: str, path: str, dataset: str) -> dict:
    obj = load_pickle(path); idx = _MMDFN_IDX[dataset]
    speakers, labels, sentences = obj[idx["spk"]], obj[idx["label"]], obj[6]
    train, test = list(obj[idx["train"]]), list(obj[idx["test"]])
    feature_dicts = [obj[idx[m]] for m in ("text", "audio", "visual")]
    all_ids = train + test
    length_errors = []
    label_counts = Counter()
    for did in all_ids:
        lengths = [len(speakers[did]), len(labels[did]), len(sentences[did])]
        lengths += [len(fd[did]) for fd in feature_dicts]
        if len(set(lengths)) != 1:
            length_errors.append({"id": str(did), "lengths": lengths})
        label_counts.update(int(v) for v in labels[did])
    return {
        "source": "MM-DFN", "dataset": dataset, "configuration": key, "path": path,
        "dialogues": {"train": len(train), "val": 0, "test": len(test)},
        "utterances": {"train": sum(len(labels[d]) for d in train),
                       "val": 0, "test": sum(len(labels[d]) for d in test)},
        "split_id_overlap": {"train_val": 0, "train_test": len(set(train) & set(test)),
                             "val_test": 0},
        "length_error_count": len(length_errors), "length_error_examples": length_errors[:5],
        "label_counts": dict(sorted(label_counts.items())),
        "label_range_valid": set(label_counts).issubset(set(range(idx["n_emo"]))),
        "feature_profile": {m: feature_profile([obj[idx[m]]], all_ids)
                            for m in ("text", "audio", "visual")},
        "dialogue_duplicates": {
            "test_exact_train": sum(dialogue_key(sentences[d]) in
                                    {dialogue_key(sentences[t]) for t in train} for d in test),
        },
        "duplicate_text_feature_consistency": duplicate_text_feature_check(
            sentences, obj[idx["text"]], all_ids),
    }


def cross_source(dataset: str, cosmic_path: str, mmdfn_path: str) -> dict:
    co, mm = load_pickle(cosmic_path), load_pickle(mmdfn_path)
    ci, mi = _COSMIC_IDX[dataset], _MMDFN_IDX[dataset]
    co_train = set(co[ci["train"]]) | set(co[ci["val"]])
    co_test = set(co[ci["test"]]); mm_train = set(mm[mi["train"]]); mm_test = set(mm[mi["test"]])
    common = sorted((co_train | co_test) & (mm_train | mm_test), key=str)
    text_equal = label_equal = length_equal = 0
    mismatches = []
    for did in common:
        ct, mt = co[ci["sent"]][did], mm[6][did]
        cl, ml = list(co[ci["labels"]][did]), list(mm[mi["label"]][did])
        same_text = [norm_text(x) for x in ct] == [norm_text(x) for x in mt]
        same_labels = cl == ml
        same_len = len(ct) == len(mt)
        text_equal += int(same_text); label_equal += int(same_labels); length_equal += int(same_len)
        if not (same_text and same_labels and same_len) and len(mismatches) < 10:
            mismatches.append({"id": str(did), "text_equal": same_text,
                               "labels_equal": same_labels, "length_equal": same_len})
    return {
        "dataset": dataset,
        "train_ids_equal_after_cosmic_val_reunion": co_train == mm_train,
        "test_ids_equal": co_test == mm_test,
        "cosmic_train_union": len(co_train), "mmdfn_train": len(mm_train),
        "cosmic_test": len(co_test), "mmdfn_test": len(mm_test),
        "common_dialogues": len(common), "text_equal": text_equal,
        "labels_equal": label_equal, "length_equal": length_equal,
        "mismatch_examples": mismatches,
    }


def write_report(result: dict, md_path: str):
    lines = ["# Access feature and split data-quality audit", "",
             "Grain: one feature row, label, and speaker entry per utterance; one shift decision "
             "per non-terminal utterance. All checks are read-only.", "",
             "| source/config | split ID overlap | length errors | non-finite values | "
             "test=train exact dialogues | repeated-text feature inconsistencies | max repeated-text Δ |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for key, row in result["sources"].items():
        overlap = sum(row["split_id_overlap"].values())
        fp = row["feature_profile"]
        nonfinite = fp["nonfinite"] if "nonfinite" in fp else sum(v["nonfinite"] for v in fp.values())
        dup = row["dialogue_duplicates"].get("test_exact_train", 0)
        inconsistent = row["duplicate_text_feature_consistency"]["feature_inconsistent_occurrences"]
        max_diff = row["duplicate_text_feature_consistency"]["max_abs_diff"]
        lines.append(f"| {key} | {overlap} | {row['length_error_count']} | {nonfinite} | "
                     f"{dup} | {inconsistent} | {max_diff:.3g} |")
    lines += ["", "## Cross-source alignment", "",
              "| dataset | train IDs aligned | test IDs aligned | text aligned | labels aligned |",
              "|---|---|---|---:|---:|"]
    for ds, row in result["cross_source"].items():
        lines.append(f"| {ds} | {row['train_ids_equal_after_cosmic_val_reunion']} | "
                     f"{row['test_ids_equal']} | {row['text_equal']}/{row['common_dialogues']} | "
                     f"{row['labels_equal']}/{row['common_dialogues']} |")
    lines += ["", "## Interpretation", "",
              "- Zero length errors/non-finite values is required before any experiment is cited.",
              "- Split-ID overlap and exact train-test dialogue duplication are separate checks; "
              "DailyDialog duplicates remain visible in this raw-source audit but are removed from every formal Access experiment.",
              "- Repeated identical utterance text should map to an identical feature vector within "
              "a deterministic utterance-local encoder. This is supporting evidence, not proof of the "
              "entire upstream extraction pipeline.",
              "- COSMIC validation IDs are reunited with COSMIC training IDs before comparison with "
              "MM-DFN because the released MM-DFN files do not contain a validation split."]
    Path(md_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    sources = {ds: audit_cosmic(ds, path) for ds, path in COSMIC.items()}
    for key, (path, ds) in MMDFN.items():
        sources[key] = audit_mmdfn(key, path, ds)
    cross = {
        "iemocap": cross_source("iemocap", COSMIC["iemocap"], MMDFN["iemocap_mm"][0]),
        "meld": cross_source("meld", COSMIC["meld"], MMDFN["meld_mm"][0]),
    }
    result = {"sources": sources, "cross_source": cross}
    Path("results/access_data_quality.json").write_text(
        json.dumps(serial(result), indent=2), encoding="utf-8")
    write_report(result, "results/access_data_quality.md")
    print("Wrote results/access_data_quality.{json,md}")


if __name__ == "__main__":
    main()
