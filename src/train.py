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
import copy

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from .dataset import (IGNORE_INDEX, ShiftSplit, assemble_inputs, input_dim,
                      load_cosmic, load_mmdfn, targets_for_dialogue)
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
def fill_predicted_current_emotion(split: ShiftSplit, seed: int = 0,
                                   tune_c: bool = False) -> dict:
    """Train-only linear ERC for the information-matched deployable baseline.

    The classifier predicts y_n from the current utterance row x_n. Ridge is used
    because it fits all emotion classes jointly in seconds; the earlier one-vs-rest
    liblinear implementation required dozens of serial binary fits per seed without
    changing what information the baseline receives. ``tune_c`` is retained as a
    backward-compatible flag and selects a validation-tuned ridge-alpha grid.
    """
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler

    def feats(dialogues):
        return np.concatenate([
            np.concatenate(list(d.features.values()), axis=1) for d in dialogues
        ], axis=0).astype(np.float64, copy=False)

    Xtr = feats(split.train)
    ytr = np.concatenate([d.labels for d in split.train])
    if len(Xtr) > 20000:  # cap for speed/stability on large corpora (e.g. DailyDialog)
        sel = np.random.default_rng(seed).choice(len(Xtr), 20000, replace=False)
        Xtr, ytr = Xtr[sel], ytr[sel]
    # StandardScaler is fitted on training utterances only.
    alpha_grid = (0.1, 1.0, 10.0, 100.0, 1000.0) if tune_c else (10.0,)
    Xva = feats(split.val)
    yva = np.concatenate([d.labels for d in split.val])
    scaler = StandardScaler().fit(Xtr)
    Xtr_scaled = scaler.transform(Xtr)
    Xva_scaled = scaler.transform(Xva)

    def stable_predict(clf, X):
        """Avoid a macOS Accelerate warning in sklearn's 2-D decision matmul."""
        coefs = np.atleast_2d(clf.coef_)
        intercepts = np.atleast_1d(clf.intercept_)
        decision = np.einsum("ij,kj->ik", X, coefs, optimize=False) + intercepts
        if len(clf.classes_) == 2:
            return clf.classes_[(decision[:, 0] > 0).astype(np.int64)]
        return clf.classes_[np.argmax(decision, axis=1)]

    best = None
    for alpha in alpha_grid:
        candidate = RidgeClassifier(alpha=alpha, solver="lsqr").fit(Xtr_scaled, ytr)
        pred = stable_predict(candidate, Xva_scaled)
        from sklearn.metrics import accuracy_score, f1_score
        score = f1_score(yva, pred, average="macro", zero_division=0)
        row = (float(score), -float(alpha), candidate,
               float(accuracy_score(yva, pred)), float(alpha))
        if best is None or row[:2] > best[:2]:
            best = row
    assert best is not None
    _, _, clf, val_acc, selected_alpha = best
    for part in (split.train, split.val, split.test):
        for d in part:
            X = np.concatenate(list(d.features.values()), axis=1).astype(np.float64, copy=False)
            d.predicted_labels = stable_predict(clf, scaler.transform(X)).astype(np.int64)
    test_y = np.concatenate([d.labels for d in split.test])
    test_pred = np.concatenate([d.predicted_labels for d in split.test])
    from sklearn.metrics import accuracy_score, f1_score
    return {
        "method": "train-only StandardScaler + RidgeClassifier(lsqr)",
        "selected_alpha": selected_alpha,
        "training_seed": int(seed),
        "val_macro_f1": float(best[0]),
        "val_accuracy": val_acc,
        "test_macro_f1": float(f1_score(test_y, test_pred, average="macro", zero_division=0)),
        "test_accuracy": float(accuracy_score(test_y, test_pred)),
    }


MAX_PARTIES = 9  # speaker-aware models index parties 0..MAX_PARTIES-1 (per-dialogue local)


def build_tensors(dialogues, split: ShiftSplit, current_emotion: str):
    """Return (X[T,D] float, shift[T] long, spk[T] long, did) per dialogue.
    spk = per-dialogue LOCAL speaker index (0-based, capped at MAX_PARTIES)."""
    items = []
    for d in dialogues:
        X = assemble_inputs(d, split.modalities, split.num_emotions, current_emotion)
        y = targets_for_dialogue(d)
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
              seed: int, epochs: int = 30, lr: float = 1e-3, batch: int = 32,
              compute_ci: bool = True, hidden: int = 128, dropout: float = 0.1,
              early_stopping: bool = False, patience: int = 5, min_epochs: int = 3,
              track_history: bool = False, model_kwargs: dict | None = None,
              evaluate_test: bool = True):
    set_seed(seed)
    dev = device()
    d_in = input_dim(split, current_emotion)
    model = build_model(model_name, d_in, hidden=hidden, dropout=dropout,
                        **(model_kwargs or {})).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = make_loss(loss_name, pos_weight_from(split.train))

    train_items = build_tensors(split.train, split, current_emotion)
    val_items = build_tensors(split.val, split, current_emotion)
    test_items = (build_tensors(split.test, split, current_emotion)
                  if evaluate_test else None)

    def batches(items, shuffle):
        order = np.random.permutation(len(items)) if shuffle else np.arange(len(items))
        for i in range(0, len(items), batch):
            chunk = [items[j] for j in order[i:i + batch]]
            X = pad_sequence([c[0] for c in chunk], batch_first=True).to(dev)
            y = pad_sequence([c[1] for c in chunk], batch_first=True,
                             padding_value=IGNORE_INDEX).to(dev)
            spk = pad_sequence([c[2] for c in chunk], batch_first=True, padding_value=0).to(dev)
            yield X, y, spk, [c[3] for c in chunk]

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

    history = []
    best_state, best_val_auc, best_epoch = None, -float("inf"), 0
    stale = 0
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for X, y, spk, _ in batches(train_items, shuffle=True):
            opt.zero_grad()
            logits = model(X, spk)
            loss = loss_fn(logits, y)
            if hasattr(model, "aux_loss"):   # e.g. CausalPseudoFuture's future-embedding
                # y[:,:-1] aligns with the auxiliary t->t+1 embedding targets.
                # The fixed variant masks padded dialogue tails; legacy experiments
                # intentionally retain their historical behavior for reproducibility.
                loss = loss + model.aux_loss(y[:, :-1] != IGNORE_INDEX)
            loss.backward(); opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        if early_stopping or track_history:
            val_s_epoch, val_y_epoch, _ = scores_for(val_items)
            from sklearn.metrics import roc_auc_score
            val_auc = (float(roc_auc_score(val_y_epoch, val_s_epoch))
                       if len(np.unique(val_y_epoch)) > 1 else float("nan"))
            history.append({"epoch": epoch + 1,
                            "train_loss": float(np.mean(epoch_losses)),
                            "val_auc": val_auc})
            if not np.isnan(val_auc) and val_auc > best_val_auc + 1e-5:
                best_val_auc, best_epoch = val_auc, epoch + 1
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
                stale = 0
            else:
                stale += 1
            if early_stopping and epoch + 1 >= min_epochs and stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_s, val_y, _ = scores_for(val_items)
    thr = tune_threshold(val_s, val_y)
    common = {"threshold": thr, "history": history,
              "best_val_auc": float(best_val_auc), "best_epoch": int(best_epoch),
              "epochs_ran": len(history) if history else epochs}
    if not evaluate_test:
        return {}, common
    test_s, test_y, test_d = scores_for(test_items)
    m = shift_metrics(test_y, (test_s >= thr).astype(int), test_s)
    if compute_ci:
        pt, lo, hi = bootstrap_ci(test_y, (test_s >= thr).astype(int), test_s, test_d,
                                  "shift_f1", seed=seed)
        m["shift_f1_ci"] = (pt, lo, hi)
    extras = {**common, "scores": test_s, "y": test_y, "dids": test_d}
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
