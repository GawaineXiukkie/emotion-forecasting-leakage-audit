"""
Shared training/evaluation loop: train_one() is used by every script in this repo, and this
module's own CLI is a convenience entry point for running a single model on a single dataset
without going through the full harness in src/experiments.py.

Causality is structural, not a training-time convention: causal models only ever see x_<=n.
The leaky (bidirectional) variant exists purely to measure leakage inflation and is never
reported as a legitimate result (see docs/leakage_checklist.md).

Run:
    python -m src.train --features data/feat/iemocap_features_roberta.pkl --dataset iemocap \
        --model gru --loss focal --settings none predicted oracle --seeds 0 1 2
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from .dataset import (IGNORE_INDEX, ShiftSplit, assemble_inputs, input_dim,
                      load_cosmic, load_mmdfn, shift_targets)
from .baselines import (BaseRateBaseline, NoChangeBaseline, SpeakerTransitionMatrix,
                        TextHistoryMLP, collect_shift_arrays, tune_threshold)
from .losses import make_loss
from .models import build_model
from .evaluate import shift_metrics, bootstrap_ci, summarize_seeds


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(s: int):
    np.random.seed(s); torch.manual_seed(s)


# --------------------------------------------------------------------------- #
def fill_predicted_current_emotion(split: ShiftSplit, seed: int = 0):
    """Cheap causal ERC: predict y_n from the CURRENT utterance features x_n (allowed input),
    train on train fold, fill .predicted_labels on every split. Enables the 'predicted' setting."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def feats(dialogues):
        return np.concatenate([np.concatenate(list(d.features.values()), axis=1) for d in dialogues], axis=0)

    Xtr = feats(split.train)
    ytr = np.concatenate([d.labels for d in split.train])
    if len(Xtr) > 20000:  # cap for speed/stability on large corpora (e.g. DailyDialog)
        sel = np.random.default_rng(seed).choice(len(Xtr), 20000, replace=False)
        Xtr, ytr = Xtr[sel], ytr[sel]
    # StandardScaler before LR: raw RoBERTa features overflow / converge slowly otherwise.
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=200, random_state=seed)).fit(Xtr, ytr)
    for part in (split.train, split.val, split.test):
        for d in part:
            X = np.concatenate(list(d.features.values()), axis=1)
            d.predicted_labels = clf.predict(X).astype(np.int64)


MAX_PARTIES = 9  # speaker-aware models index parties 0..MAX_PARTIES-1 (per-dialogue local)


def build_tensors(dialogues, split: ShiftSplit, current_emotion: str):
    """Return (X[T,D] float, shift[T] long, spk[T] long, did) per dialogue.
    spk = per-dialogue LOCAL speaker index (0-based, capped at MAX_PARTIES)."""
    items = []
    for d in dialogues:
        X = assemble_inputs(d, split.modalities, split.num_emotions, current_emotion)
        y = shift_targets(d.labels)
        order = {s: i for i, s in enumerate(dict.fromkeys(d.speakers.tolist()))}
        spk = np.array([order[s] % MAX_PARTIES for s in d.speakers.tolist()], dtype=np.int64)
        items.append((torch.tensor(X), torch.tensor(y), torch.tensor(spk), d.did))
    return items


def pos_weight_from(dialogues) -> float:
    _, _, shift = collect_shift_arrays(dialogues)
    pos = max(int(shift.sum()), 1)
    neg = max(len(shift) - pos, 1)
    return neg / pos


def train_one(split: ShiftSplit, model_name: str, loss_name: str, current_emotion: str,
              seed: int, epochs: int = 30, lr: float = 1e-3, batch: int = 32):
    set_seed(seed)
    dev = device()
    d_in = input_dim(split, current_emotion)
    model = build_model(model_name, d_in).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = make_loss(loss_name, pos_weight_from(split.train))

    train_items = build_tensors(split.train, split, current_emotion)
    val_items = build_tensors(split.val, split, current_emotion)
    test_items = build_tensors(split.test, split, current_emotion)

    def batches(items, shuffle):
        order = np.random.permutation(len(items)) if shuffle else np.arange(len(items))
        for i in range(0, len(items), batch):
            chunk = [items[j] for j in order[i:i + batch]]
            X = pad_sequence([c[0] for c in chunk], batch_first=True).to(dev)
            y = pad_sequence([c[1] for c in chunk], batch_first=True,
                             padding_value=IGNORE_INDEX).to(dev)
            spk = pad_sequence([c[2] for c in chunk], batch_first=True, padding_value=0).to(dev)
            yield X, y, spk, [c[3] for c in chunk]

    for _ in range(epochs):
        model.train()
        for X, y, spk, _ in batches(train_items, shuffle=True):
            opt.zero_grad()
            logits = model(X, spk)
            loss = loss_fn(logits, y)
            if hasattr(model, "aux_loss"):   # e.g. CausalPseudoFuture's future-embedding
                loss = loss + model.aux_loss()  # regression loss; no-op for every other model
            loss.backward(); opt.step()

    # gather scores at valid decision points
    def scores_for(items):
        model.eval(); sc, tr, did = [], [], []
        with torch.no_grad():
            for X, y, spk, dids in batches(items, shuffle=False):
                p = torch.sigmoid(model(X, spk)).cpu().numpy()
                yy = y.cpu().numpy()
                for b, d_id in enumerate(dids):
                    valid = np.where(yy[b] != IGNORE_INDEX)[0]
                    sc.extend(p[b, valid]); tr.extend(yy[b, valid]); did.extend([d_id] * len(valid))
        return np.array(sc), np.array(tr, dtype=np.int64), np.array(did, dtype=object)

    val_s, val_y, _ = scores_for(val_items)
    thr = tune_threshold(val_s, val_y)
    test_s, test_y, test_d = scores_for(test_items)
    m = shift_metrics(test_y, (test_s >= thr).astype(int), test_s)
    pt, lo, hi = bootstrap_ci(test_y, (test_s >= thr).astype(int), test_s, test_d, "shift_f1", seed=seed)
    m["shift_f1_ci"] = (pt, lo, hi)
    extras = {"scores": test_s, "y": test_y, "dids": test_d, "threshold": thr}
    return m, extras


def run_baselines(split: ShiftSplit, return_scores: bool = False):
    val_y = collect_shift_arrays(split.val)[2]
    test_y = collect_shift_arrays(split.test)[2]
    out, scores = {}, {}
    for name, model in [
        ("base_rate", BaseRateBaseline().fit(split.train)),
        ("no_change", NoChangeBaseline()),
        ("speaker_transition", SpeakerTransitionMatrix(split.num_emotions).fit(split.train)),
        ("text_history_mlp", TextHistoryMLP().fit(split.train)),
    ]:
        vs = model.predict_score(split.val)
        thr = tune_threshold(vs, val_y) if name != "no_change" else 0.5
        ts = model.predict_score(split.test)
        out[name] = shift_metrics(test_y, (ts >= thr).astype(int), ts)
        scores[name] = ts
    return (out, scores) if return_scores else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--dataset", required=True, choices=["iemocap", "meld", "emorynlp", "dailydialog"])
    ap.add_argument("--source", default="cosmic", choices=["cosmic", "mmdfn"])
    ap.add_argument("--feature", default="roberta1", choices=["roberta1", "roberta_all"])
    ap.add_argument("--modalities", nargs="+", default=["text", "audio", "visual"],
                    help="mmdfn only: subset of text audio visual")
    ap.add_argument("--model", default="gru", choices=["gru", "tcn", "transformer"])
    ap.add_argument("--loss", default="focal", choices=["focal", "cb_ce"])
    ap.add_argument("--settings", nargs="+", default=["none", "predicted", "oracle"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()

    if args.source == "mmdfn":
        split = load_mmdfn(args.features, args.dataset, tuple(args.modalities))
        print(f"[mmdfn] modalities={split.modalities} feat_dim={split.feature_dim} "
              f"train={len(split.train)} val={len(split.val)} test={len(split.test)}")
    else:
        split = load_cosmic(args.features, args.dataset, args.feature)
    if "predicted" in args.settings:
        fill_predicted_current_emotion(split)

    print("== Baselines (must beat: speaker_transition, text_history_mlp) ==")
    for name, m in run_baselines(split).items():
        print(f"  {name:20s} shift_f1={m['shift_f1']:.3f} auc={m['shift_auc']:.3f} wf1={m['weighted_f1']:.3f}")

    print(f"\n== {args.model} / {args.loss} ==")
    for setting in args.settings:
        per_seed = []
        for s in args.seeds:
            m, _ = train_one(split, args.model, args.loss, setting, seed=s)
            per_seed.append({k: v for k, v in m.items()
                             if isinstance(v, (int, float)) and not isinstance(v, bool)})
        agg = summarize_seeds(per_seed)
        tag = setting + ("  [HEADLINE]" if setting in ("none", "predicted") else "  [upper-bound only]")
        print(f"  {tag}")
        for k in ("shift_f1", "shift_recall", "shift_auc", "weighted_f1", "macro_f1"):
            mean, std = agg[k]
            print(f"      {k:14s} {mean:.3f} ± {std:.3f}")


if __name__ == "__main__":
    main()
