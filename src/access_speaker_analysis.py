"""Speaker-relation audit for the IEEE Access extension.

The original benchmark defines an immediate-turn target

    z_n = 1[y_{n+1} != y_n].

In multi-party dialogue, n+1 is often spoken by another person.  This script
separates same-speaker from speaker-switch decision points and also constructs a
distinct self-shift target: the emotion change at the current speaker's next
utterance, skipping intervening turns by other speakers.  It uses labels and
train-fold transition counts only; no test labels are used for fitting.

Run:
    python -m src.access_speaker_analysis
"""
from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

from .dataset import Dialogue
from .experiments import ALL_KEYS, load_split


def safe_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, score))


def immediate_points(dialogues: list[Dialogue]):
    """Yield (current emotion, next emotion, current speaker, same speaker, gap)."""
    for d in dialogues:
        for n in range(max(len(d.labels) - 1, 0)):
            yield (int(d.labels[n]), int(d.labels[n + 1]), d.speakers[n],
                   bool(d.speakers[n] == d.speakers[n + 1]), 1)


def self_shift_points(dialogues: list[Dialogue]):
    """Yield transitions to each speaker's next own utterance.

    At n, the destination is the smallest m>n with speaker_m == speaker_n.
    The gap is m-n and can exceed one when other speakers intervene.
    """
    for d in dialogues:
        next_pos: dict[object, int] = {}
        rows = []
        for n in range(len(d.labels) - 1, -1, -1):
            spk = d.speakers[n]
            if spk in next_pos:
                m = next_pos[spk]
                rows.append((int(d.labels[n]), int(d.labels[m]), spk, True, m - n))
            next_pos[spk] = n
        yield from reversed(rows)


@dataclass
class TransitionScorer:
    num_emotions: int
    alpha: float = 1.0
    min_count: int = 20

    def fit(self, points):
        self.global_counts = np.zeros((self.num_emotions, self.num_emotions), dtype=np.float64)
        self.speaker_counts: dict[object, np.ndarray] = {}
        for cur, nxt, spk, _, _ in points:
            self.global_counts[cur, nxt] += 1
            self.speaker_counts.setdefault(
                spk, np.zeros_like(self.global_counts)
            )[cur, nxt] += 1
        return self

    def _shift_score(self, counts: np.ndarray, current: int) -> float:
        row = counts[current] + self.alpha
        return float(1.0 - row[current] / row.sum())

    def score(self, points, speaker_conditioned: bool) -> np.ndarray:
        out = []
        for cur, _, spk, _, _ in points:
            counts = self.global_counts
            if speaker_conditioned:
                local = self.speaker_counts.get(spk)
                if local is not None and local[cur].sum() >= self.min_count:
                    counts = local
            out.append(self._shift_score(counts, cur))
        return np.asarray(out, dtype=np.float64)


def summarize(train_dialogues: list[Dialogue], test_dialogues: list[Dialogue],
              num_emotions: int, point_fn) -> dict:
    train = list(point_fn(train_dialogues))
    test = list(point_fn(test_dialogues))
    scorer = TransitionScorer(num_emotions).fit(train)
    y = np.asarray([int(nxt != cur) for cur, nxt, _, _, _ in test], dtype=np.int64)
    same = np.asarray([same for _, _, _, same, _ in test], dtype=bool)
    gaps = np.asarray([gap for _, _, _, _, gap in test], dtype=np.int64)
    global_score = scorer.score(test, speaker_conditioned=False)
    speaker_score = scorer.score(test, speaker_conditioned=True)

    def group(mask: np.ndarray) -> dict:
        yy = y[mask]
        return {
            "n": int(mask.sum()),
            "fraction": float(mask.mean()) if len(mask) else 0.0,
            "shift_rate": float(yy.mean()) if len(yy) else None,
            "global_transition_auc": safe_auc(yy, global_score[mask]),
            "speaker_transition_auc": safe_auc(yy, speaker_score[mask]),
        }

    horizon = {}
    for name, mask in {
        "gap_1": gaps == 1,
        "gap_2": gaps == 2,
        "gap_3_plus": gaps >= 3,
    }.items():
        horizon[name] = group(mask)

    return {
        "n": len(test),
        "shift_rate": float(y.mean()) if len(y) else None,
        "global_transition_auc": safe_auc(y, global_score),
        "speaker_transition_auc": safe_auc(y, speaker_score),
        "same_speaker": group(same),
        "speaker_switch": group(~same),
        "gap_mean": float(gaps.mean()) if len(gaps) else None,
        "gap_median": float(np.median(gaps)) if len(gaps) else None,
        "gap_p90": float(np.quantile(gaps, 0.9)) if len(gaps) else None,
        "horizon": horizon,
    }


def switch_rate_bootstrap(dialogues: list[Dialogue], n_boot: int = 5000,
                          seed: int = 0) -> dict | None:
    """Dialogue-cluster bootstrap for shift-rate(switch) - shift-rate(same)."""
    rows = []
    for d in dialogues:
        same_pos = same_n = switch_pos = switch_n = 0
        for cur, nxt, _, same, _ in immediate_points([d]):
            shifted = int(nxt != cur)
            if same:
                same_pos += shifted; same_n += 1
            else:
                switch_pos += shifted; switch_n += 1
        rows.append((same_pos, same_n, switch_pos, switch_n))
    arr = np.asarray(rows, dtype=np.float64)
    total = arr.sum(axis=0)
    if total[1] == 0 or total[3] == 0:
        return None
    observed = total[2] / total[3] - total[0] / total[1]
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        sampled = arr[rng.integers(0, len(arr), len(arr))].sum(axis=0)
        if sampled[1] and sampled[3]:
            boot.append(sampled[2] / sampled[3] - sampled[0] / sampled[1])
    boot = np.asarray(boot)
    # Null-centered bootstrap test: the resampling distribution is centered on the
    # *observed* statistic (sampling variability), not on 0, so testing against 0
    # directly (e.g. 2*min(P(boot<=0), P(boot>=0))) is not a valid p-value except by
    # coincidence. Recentering at 0 and comparing to |observed| treats the bootstrap
    # deviations as a stand-in for the null sampling distribution instead.
    deviations = np.abs(boot - observed)
    p = min(1.0, float(np.mean(deviations >= abs(observed))))
    return {
        "difference": float(observed),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "p_value": p,
        "bootstrap_dialogue_clusters": int(n_boot),
    }


def md_value(x, digits=3):
    return "--" if x is None else f"{x:.{digits}f}"


def write_markdown(results: dict, path: str):
    lines = [
        "# Speaker-relation audit",
        "",
        "All transition probabilities are fit on the training fold only. AUC is evaluated "
        "on the original test fold. `Immediate` is the paper's adjacent-turn target; "
        "`self-shift` predicts the current speaker's emotion at their next own utterance.",
        "",
        "## Immediate-turn target",
        "",
        "| dataset | test points | next speaker same | shift rate: same | shift rate: switch | global-trans AUC | speaker-trans AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ds, r in results.items():
        a = r["immediate"]
        lines.append(
            f"| {ds} | {a['n']:,} | {a['same_speaker']['fraction']:.3f} | "
            f"{md_value(a['same_speaker']['shift_rate'])} | "
            f"{md_value(a['speaker_switch']['shift_rate'])} | "
            f"{md_value(a['global_transition_auc'])} | {md_value(a['speaker_transition_auc'])} |"
        )
    lines += [
        "",
        "## Current speaker's next-own-utterance target",
        "",
        "| dataset | test points | shift rate | mean gap (turns) | p90 gap | global-trans AUC | speaker-trans AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ds, r in results.items():
        a = r["self_shift"]
        lines.append(
            f"| {ds} | {a['n']:,} | {md_value(a['shift_rate'])} | "
            f"{md_value(a['gap_mean'], 2)} | {md_value(a['gap_p90'], 1)} | "
            f"{md_value(a['global_transition_auc'])} | {md_value(a['speaker_transition_auc'])} |"
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    results = {}
    for key in ALL_KEYS:
        print(f"[{key}] loading", flush=True)
        split = load_split(key)
        results[key] = {
            "immediate": summarize(split.train, split.test, split.num_emotions, immediate_points),
            "self_shift": summarize(split.train, split.test, split.num_emotions, self_shift_points),
        }
        results[key]["immediate"]["switch_minus_same"] = switch_rate_bootstrap(split.test)
        a = results[key]["immediate"]
        print(f"[{key}] same-next={a['same_speaker']['fraction']:.3f} "
              f"shift(same/switch)={md_value(a['same_speaker']['shift_rate'])}/"
              f"{md_value(a['speaker_switch']['shift_rate'])}", flush=True)
        del split
        gc.collect()

    os.makedirs("results", exist_ok=True)
    with open("results/access_speaker_analysis.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    write_markdown(results, "results/access_speaker_analysis.md")
    print("Wrote results/access_speaker_analysis.{json,md}")


if __name__ == "__main__":
    main()
