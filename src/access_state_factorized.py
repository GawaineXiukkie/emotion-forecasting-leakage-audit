"""Causal state-factorized forecasting experiment for the Access revision.

Motivation: next-turn emotion inference work models the future emotion state,
while shift-only supervision collapses every directed KxK transition to one bit.
This model jointly predicts current emotion, future emotion, and the binary shift.
No gold emotion label or future utterance is used at inference. The blend between
direct and factorized shift probabilities is selected on validation data only.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.nn.utils.rnn import pad_sequence

from .access_revision_experiments import (jsonable, load_revision_split,
                                          point_index)
from .baselines import PredictedLabelTransitionMatrix, collect_shift_arrays, tune_threshold
from .dataset import (IGNORE_INDEX, assemble_inputs, target_index_for_dialogue,
                      targets_for_dialogue)
from .evaluate import (hierarchical_bootstrap_auc,
                       paired_cluster_permutation_auc, shift_metrics)
from .experiments import COSMIC
from .holm_correction import holm_bonferroni
from .losses import make_loss
from .models import CausalStateFactorizedForecaster
from .train import device, fill_predicted_current_emotion, pos_weight_from, set_seed


SEARCH = [
    {"id": "h128_l3e4_aux25", "hidden": 128, "lr": 3e-4, "dropout": .1,
     "lambda_current": .25, "lambda_future": 1.0},
    {"id": "h128_l1e3_aux25", "hidden": 128, "lr": 1e-3, "dropout": .1,
     "lambda_current": .25, "lambda_future": 1.0},
    {"id": "h256_l3e4_aux25", "hidden": 256, "lr": 3e-4, "dropout": .3,
     "lambda_current": .25, "lambda_future": 1.0},
    {"id": "h128_l1e3_aux100", "hidden": 128, "lr": 1e-3, "dropout": .1,
     "lambda_current": 1.0, "lambda_future": 1.0},
]
BLENDS = (0.0, 0.25, 0.5, 0.75, 1.0)  # 0=factorized, 1=direct


def make_items(dialogues, split):
    items = []
    for d in dialogues:
        x = assemble_inputs(d, split.modalities, split.num_emotions, "none")
        shift = targets_for_dialogue(d)
        future = np.full(len(d.labels), IGNORE_INDEX, dtype=np.int64)
        for n in np.where(shift != IGNORE_INDEX)[0]:
            future[n] = d.labels[target_index_for_dialogue(d, int(n))]
        items.append((torch.tensor(x), torch.tensor(shift),
                      torch.tensor(d.labels), torch.tensor(future), d.did))
    return items


def train_state_model(split, config, seed, epochs=30, batch_size=32,
                      evaluate_test=True):
    set_seed(seed)
    dev = device()
    model = CausalStateFactorizedForecaster(
        split.feature_dim, split.num_emotions, config["hidden"], config["dropout"]).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    shift_loss = make_loss("focal", pos_weight_from(split.train))
    train_items = make_items(split.train, split)
    val_items = make_items(split.val, split)
    test_items = make_items(split.test, split)

    def batches(items, shuffle):
        order = np.random.permutation(len(items)) if shuffle else np.arange(len(items))
        for start in range(0, len(items), batch_size):
            chunk = [items[i] for i in order[start:start + batch_size]]
            x = pad_sequence([r[0] for r in chunk], batch_first=True).to(dev)
            shift = pad_sequence([r[1] for r in chunk], batch_first=True,
                                 padding_value=IGNORE_INDEX).to(dev)
            current = pad_sequence([r[2] for r in chunk], batch_first=True,
                                   padding_value=IGNORE_INDEX).to(dev)
            future = pad_sequence([r[3] for r in chunk], batch_first=True,
                                  padding_value=IGNORE_INDEX).to(dev)
            yield x, shift, current, future, [r[4] for r in chunk]

    def predict(items):
        model.eval()
        direct, factored, target, dids = [], [], [], []
        with torch.no_grad():
            for x, shift, current, future, batch_dids in batches(items, False):
                shift_logits, current_logits, future_logits = model.forward_all(x)
                direct_p = torch.sigmoid(shift_logits)
                current_p = torch.softmax(current_logits, dim=-1)
                future_p = torch.softmax(future_logits, dim=-1)
                factor_p = 1.0 - (current_p * future_p).sum(dim=-1)
                yy = shift.cpu().numpy()
                for i, did in enumerate(batch_dids):
                    valid = np.where(yy[i] != IGNORE_INDEX)[0]
                    direct.extend(direct_p[i, valid].cpu().numpy())
                    factored.extend(factor_p[i, valid].cpu().numpy())
                    target.extend(yy[i, valid]); dids.extend([did] * len(valid))
        return (np.asarray(direct), np.asarray(factored),
                np.asarray(target, dtype=np.int64), np.asarray(dids, dtype=object))

    best_state, best_auc, best_epoch, best_blend = None, -np.inf, 0, 1.0
    history, stale = [], 0
    for epoch in range(epochs):
        model.train(); losses = []
        for x, shift, current, future, _ in batches(train_items, True):
            optimizer.zero_grad()
            shift_logits, current_logits, future_logits = model.forward_all(x)
            valid = shift != IGNORE_INDEX
            loss = shift_loss(shift_logits, shift)
            loss = loss + config["lambda_current"] * F.cross_entropy(
                current_logits[valid], current[valid])
            loss = loss + config["lambda_future"] * F.cross_entropy(
                future_logits[valid], future[valid])
            loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        direct, factor, yv, _ = predict(val_items)
        candidates = [(roc_auc_score(yv, a * direct + (1 - a) * factor), a)
                      for a in BLENDS]
        val_auc, blend = max(candidates, key=lambda r: (r[0], -abs(r[1] - .5)))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "val_auc": float(val_auc), "blend_direct": float(blend)})
        if val_auc > best_auc + 1e-5:
            best_auc, best_epoch, best_blend = float(val_auc), epoch + 1, float(blend)
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
        if epoch + 1 >= 5 and stale >= 5:
            break
    model.load_state_dict(best_state)
    vd, vf, vy, _ = predict(val_items)
    val_score = best_blend * vd + (1 - best_blend) * vf
    threshold = tune_threshold(val_score, vy)
    common = {"threshold": threshold, "best_val_auc": best_auc,
              "best_epoch": best_epoch, "blend_direct": best_blend,
              "history": history}
    if not evaluate_test:
        return None, common
    td, tf, ty, dids = predict(test_items)
    score = best_blend * td + (1 - best_blend) * tf
    metrics = shift_metrics(ty, (score >= threshold).astype(int), score)
    return metrics, {**common, "scores": score, "direct_scores": td,
                     "factor_scores": tf, "y": ty, "dids": dids}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(COSMIC), choices=list(COSMIC))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--search-limit", type=int, default=len(SEARCH),
                        choices=range(1, len(SEARCH) + 1))
    parser.add_argument("--n-boot", type=int, default=1999)
    parser.add_argument("--n-perm", type=int, default=4999)
    parser.add_argument("--cache-dir", default="results/cache_access_state_factorized")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path(args.cache_dir); root.mkdir(parents=True, exist_ok=True)
    results = {}
    for ds in args.datasets:
        split = load_revision_split(ds)
        y, dids = point_index(split.test)
        val_y = collect_shift_arrays(split.val)[2]
        folder = root / ds; folder.mkdir(parents=True, exist_ok=True)

        search = []
        for config in SEARCH[:args.search_limit]:
            path = folder / f"search_{config['id']}.json"
            if path.exists() and not args.force:
                row = json.loads(path.read_text(encoding="utf-8"))
                if row.get("max_epochs") != args.epochs or row.get("config") != config:
                    row = None
            else:
                row = None
            if row is None:
                _, extra = train_state_model(split, config, 0, args.epochs,
                                             evaluate_test=False)
                row = {"config": config, "best_val_auc": extra["best_val_auc"],
                       "best_epoch": extra["best_epoch"],
                       "blend_direct": extra["blend_direct"], "history": extra["history"],
                       "max_epochs": args.epochs}
                path.write_text(json.dumps(jsonable(row), indent=2), encoding="utf-8")
            search.append(row)
            print(ds, config["id"], row["best_val_auc"], flush=True)
        selected = max(search, key=lambda r: (r["best_val_auc"],
                                               -r["config"]["hidden"]))["config"]

        scores, metric_rows, run_rows = [], [], []
        baseline_scores, erc_rows, baseline_metrics = [], [], []
        for seed in args.seeds:
            meta_path, npz_path = folder / f"seed{seed}.json", folder / f"seed{seed}.npz"
            if meta_path.exists() and npz_path.exists() and not args.force:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("max_epochs") == args.epochs and meta.get("config") == selected:
                    with np.load(npz_path, allow_pickle=False) as z:
                        score = z["scores"]
                else:
                    meta = None
            else:
                meta = None
            if meta is None:
                metrics, extra = train_state_model(split, selected, seed, args.epochs)
                score = extra["scores"]
                meta = {"seed": seed, "config": selected, "metrics": metrics,
                        "best_val_auc": extra["best_val_auc"],
                        "best_epoch": extra["best_epoch"],
                        "blend_direct": extra["blend_direct"],
                        "max_epochs": args.epochs,
                        "threshold": extra["threshold"], "history": extra["history"]}
                np.savez_compressed(npz_path, scores=score.astype(np.float32),
                                    direct_scores=extra["direct_scores"].astype(np.float32),
                                    factor_scores=extra["factor_scores"].astype(np.float32),
                                    y=y.astype(np.int8), dids=dids)
                meta_path.write_text(json.dumps(jsonable(meta), indent=2), encoding="utf-8")
            scores.append(score); metric_rows.append(meta["metrics"]); run_rows.append(meta)

            erc_rows.append(fill_predicted_current_emotion(split, seed=seed, tune_c=True))
            baseline = PredictedLabelTransitionMatrix(split.num_emotions).fit(split.train)
            val_score = baseline.predict_score(split.val)
            threshold = tune_threshold(val_score, val_y)
            bs = baseline.predict_score(split.test)
            baseline_scores.append(bs)
            baseline_metrics.append(shift_metrics(y, (bs >= threshold).astype(int), bs))

        a, b = np.stack(scores), np.stack(baseline_scores)
        inference = {**hierarchical_bootstrap_auc(y, a, b, dids, args.n_boot, 5901),
                     **paired_cluster_permutation_auc(y, a, b, dids, args.n_perm, 5901)}
        results[ds] = {
            "selected": selected, "search": search,
            "auc": {"mean": float(np.mean([m["shift_auc"] for m in metric_rows])),
                    "std": float(np.std([m["shift_auc"] for m in metric_rows]))},
            "pr_auc": {"mean": float(np.mean([m["shift_pr_auc"] for m in metric_rows])),
                       "std": float(np.std([m["shift_pr_auc"] for m in metric_rows]))},
            "predicted_transition_auc": {
                "mean": float(np.mean([m["shift_auc"] for m in baseline_metrics])),
                "std": float(np.std([m["shift_auc"] for m in baseline_metrics]))},
            "erc": erc_rows, "inference": inference, "runs": run_rows,
        }
        print(ds, results[ds]["auc"], inference, flush=True)

    corrected = holm_bonferroni({ds: row["inference"]["p_value"]
                                 for ds, row in results.items()})
    for ds, value in corrected.items():
        results[ds]["inference"]["holm"] = value
    Path("results/access_state_factorized.json").write_text(
        json.dumps(jsonable(results), indent=2), encoding="utf-8")
    lines = ["# Causal state-factorized forecaster", "",
             "No gold emotion label or future utterance is used at inference. DailyDialog is decontaminated.", "",
             "| dataset | state-factorized AUC | predicted-transition AUC | ΔAUC [95% CI] | p | Holm p |",
             "|---|---:|---:|---:|---:|---:|"]
    for ds, row in results.items():
        a, b, r = row["auc"], row["predicted_transition_auc"], row["inference"]
        lines.append(f"| {ds} | {a['mean']:.3f}±{a['std']:.3f} | "
                     f"{b['mean']:.3f}±{b['std']:.3f} | {r['delta_auc']:+.3f} "
                     f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | "
                     f"{r['p_value']:.4g} | {r['holm']['p_holm']:.4g} |")
    Path("results/access_state_factorized.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
