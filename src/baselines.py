"""
Baselines the models need to beat.

  - no-change / last-label : always predict "no shift" (shift=0). For this task these
                             coincide: predicting y_{n+1}=y_n means shift=0 everywhere.
  - speaker-specific transition matrix : P(y_{n+1} | y_n, speaker), estimated on the train
                             fold only. Test speakers are unseen (e.g. IEMOCAP's session
                             split), so unseen speakers back off to the global train
                             transition matrix. This is the baseline the paper's models are
                             measured against.
  - text-history MLP       : sklearn MLP over mean-pooled text history x_<=n.

All operate on decision points n (0 to T-2). Causal: only x_<=n and y_n are used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import (Dialogue, IGNORE_INDEX, target_index_for_dialogue,
                      targets_for_dialogue)


# --------------------------------------------------------------------------- #
def iter_decision_points(dialogues: list[Dialogue]):
    """Yield (dialogue, n) for every valid decision point (has a successor)."""
    for d in dialogues:
        s = targets_for_dialogue(d)
        for n in np.where(s != IGNORE_INDEX)[0]:
            yield d, int(n)


def collect_shift_arrays(dialogues: list[Dialogue]):
    """Return aligned arrays over all decision points: y_n, speaker_n, shift."""
    y_n, spk_n, shift = [], [], []
    for d, n in iter_decision_points(dialogues):
        y_n.append(int(d.labels[n]))
        spk_n.append(d.speakers[n])
        shift.append(int(targets_for_dialogue(d)[n]))
    return np.asarray(y_n), np.asarray(spk_n, dtype=object), np.asarray(shift, dtype=np.int64)


# --------------------------------------------------------------------------- #
class NoChangeBaseline:
    """Always predict no shift (== last-label / inertia). Score = 0 for shift."""
    def predict_score(self, dialogues: list[Dialogue]) -> np.ndarray:
        _, _, shift = collect_shift_arrays(dialogues)
        return np.zeros_like(shift, dtype=np.float64)


class BaseRateBaseline:
    """Predict the train-majority class (constant). Score = train shift base rate -> threshold
    tuning collapses it to always-majority; AUC = 0.5. The 'is there any signal at all' floor."""
    def fit(self, train_dialogues: list[Dialogue]):
        _, _, shift = collect_shift_arrays(train_dialogues)
        self.rate = float(shift.mean())
        return self

    def predict_score(self, dialogues: list[Dialogue]) -> np.ndarray:
        _, _, shift = collect_shift_arrays(dialogues)
        return np.full_like(shift, self.rate, dtype=np.float64)


class SpeakerTransitionMatrix:
    """P(y_{n+1} | y_n, speaker) with Laplace smoothing; global backoff for unseen speakers.

    Conditions on the CURRENT speaker (speaker of utterance n) — known at decision time,
    fully causal. shift_score = 1 - P(y_{n+1}=y_n | y_n, speaker)."""
    def __init__(self, num_emotions: int, alpha: float = 1.0, min_count: int = 20):
        self.K = num_emotions
        self.alpha = alpha
        self.min_count = min_count
        self.per_speaker: dict[object, np.ndarray] = {}  # speaker -> [K, K] counts
        self.global_counts = np.zeros((self.K, self.K), dtype=np.float64)

    def fit(self, train_dialogues: list[Dialogue]):
        for d, n in iter_decision_points(train_dialogues):
            target_idx = target_index_for_dialogue(d, n)
            j, k, spk = int(d.labels[n]), int(d.labels[target_idx]), d.speakers[n]
            self.per_speaker.setdefault(spk, np.zeros((self.K, self.K)))[j, k] += 1
            self.global_counts[j, k] += 1
        return self

    def _row_prob(self, counts: np.ndarray, j: int) -> np.ndarray:
        row = counts[j] + self.alpha
        return row / row.sum()

    def _shift_prob_for(self, j: int, spk) -> float:
        spk_counts = self.per_speaker.get(spk)
        if spk_counts is None or spk_counts[j].sum() < self.min_count:
            counts = self.global_counts            # backoff: unseen / sparse speaker
        else:
            counts = spk_counts
        p = self._row_prob(counts, j)
        return float(1.0 - p[j])                    # P(next != current)

    def predict_score(self, dialogues: list[Dialogue], use_predicted_labels: bool = False) -> np.ndarray:
        y_n, spk_n, _ = collect_shift_arrays(dialogues)
        if use_predicted_labels:
            rows = []
            for d, n in iter_decision_points(dialogues):
                if d.predicted_labels is None:
                    raise ValueError(f"{d.did}: predicted labels are required")
                rows.append(int(d.predicted_labels[n]))
            y_n = np.asarray(rows, dtype=np.int64)
        return np.array([self._shift_prob_for(int(j), s) for j, s in zip(y_n, spk_n)])


class PredictedLabelTransitionMatrix(SpeakerTransitionMatrix):
    """Deployable transition baseline using train-only ERC predictions at test time."""
    def predict_score(self, dialogues: list[Dialogue]) -> np.ndarray:
        return super().predict_score(dialogues, use_predicted_labels=True)


# --------------------------------------------------------------------------- #
def _pooled_text_history(dialogues: list[Dialogue], modality: str = "text") -> np.ndarray:
    """Mean-pool features of utterances 0..n (inclusive) at each decision point."""
    rows = []
    for d in dialogues:
        s = targets_for_dialogue(d)
        X = d.features.get(modality)
        if X is None:  # fallback: concat all modalities
            X = np.concatenate(list(d.features.values()), axis=1)
        csum = np.cumsum(X, axis=0)
        for n in np.where(s != IGNORE_INDEX)[0]:
            rows.append(csum[n] / (n + 1))
    return np.asarray(rows, dtype=np.float32)


@dataclass
class TextHistoryMLP:
    """sklearn MLP over mean-pooled text history. A genuine (non-trivial) baseline."""
    hidden: tuple = (128,)
    seed: int = 0

    max_fit: int = 20000  # cap training rows for speed/stability on large corpora (logged)

    def fit(self, train_dialogues: list[Dialogue]):
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        X = _pooled_text_history(train_dialogues)
        _, _, y = collect_shift_arrays(train_dialogues)
        if len(X) > self.max_fit:
            rng = np.random.default_rng(self.seed)
            sel = rng.choice(len(X), self.max_fit, replace=False)
            X, y = X[sel], y[sel]
        # StandardScaler is essential: raw RoBERTa features are unscaled -> matmul overflow
        # and very slow / non-converging MLP. early_stopping keeps it fast.
        self.clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=self.hidden, max_iter=100,
                          early_stopping=True, n_iter_no_change=5, random_state=self.seed),
        ).fit(X, y)
        return self

    def predict_score(self, dialogues: list[Dialogue]) -> np.ndarray:
        X = _pooled_text_history(dialogues)
        return self.clf.predict_proba(X)[:, 1]


def tune_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    """Pick the threshold maximizing shift-F1 on a (validation) set."""
    from sklearn.metrics import f1_score
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(np.concatenate([[0.0, 1.0], scores])):
        f1 = f1_score(y, (scores >= t).astype(int), pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t
