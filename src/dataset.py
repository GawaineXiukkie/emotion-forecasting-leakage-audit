"""
Cached-feature loading + target-absent windowing + leakage truncation.

We use a causal sequence formulation: the model consumes a whole dialogue once
(unidirectional GRU / causal TCN), and at each position n emits a shift prediction
for the next utterance, i.e. shift[n] = 1[y_{n+1} != y_n]. Position T-1 has no n+1
and is masked out. Causality is structural: the model at n cannot see x_{>n}, so the
target utterance n+1 is never an input.

Off-the-shelf ERC features (MM-DFN / conv-emotion) are per-utterance, so a feature
vector at index i summarizes only utterance i -- no window crosses into n+1.

Current-emotion setting (see README): controls whether the current emotion label is
appended to x_n:  'oracle' (gold y_n) | 'predicted' (ŷ_n provided) | 'none'.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

IGNORE_INDEX = -100  # masked positions (last utterance of each dialogue)


@dataclass
class Dialogue:
    did: str
    features: dict[str, np.ndarray]   # modality -> [T, d_modality]
    labels: np.ndarray                # [T] int emotion class
    speakers: np.ndarray              # [T] speaker id (string/int); used by transition baseline
    predicted_labels: np.ndarray | None = None  # [T] ŷ_n for the 'predicted' setting
    custom_shift_targets: np.ndarray | None = None  # optional [T] binary/IGNORE target
    custom_target_indices: np.ndarray | None = None  # optional [T] future label index

    def __post_init__(self):
        T = len(self.labels)
        for m, arr in self.features.items():
            assert arr.shape[0] == T, f"{self.did}: modality {m} has {arr.shape[0]} rows, expected {T}"
        assert len(self.speakers) == T
        if self.custom_shift_targets is not None:
            assert len(self.custom_shift_targets) == T
        if self.custom_target_indices is not None:
            assert len(self.custom_target_indices) == T


@dataclass
class ShiftSplit:
    train: list[Dialogue]
    val: list[Dialogue]
    test: list[Dialogue]
    modalities: tuple[str, ...] = ()
    num_emotions: int = 0
    feature_dim: int = 0  # concatenated modality dim (before current-emotion append)


# --------------------------------------------------------------------------- #
# Loading — declare-lab/conv-emotion COSMIC RoBERTa pickles (real schema, verified).
#
# IEMOCAP list[10]: [0]speakers('M'/'F') [1]labels(0-5) [2..5]roberta1-4[T,1024]
#                   [6]sentences [7]trainIds [8]testIds [9]validIds   (split by session;
#                   train=Ses01-04, test=Ses05 -> speaker-independent).
# MELD list[11]:    [0]speakers(one-hot[T,9]) [1]emotion(0-6) [2]sentiment
#                   [3..6]roberta1-4 [7]sentences [8]trainIds [9]testIds [10]validIds.
#
# Feature used: roberta1 (1024-d per utterance, independent per utterance -> no future leak).
# IEMOCAP speaker id is made global (session+gender => the real 10 actors) so the
# speaker-transition baseline is person-level and the unseen-test-speaker backoff is real.
# Non-IEMOCAP speaker IDs are dialogue-qualified local roles. This preserves
# within-dialogue equality for speaker-aware models while preventing role 0 in
# unrelated dialogues from being treated as one global person by a baseline.
# --------------------------------------------------------------------------- #
_COSMIC_IDX = {
    "iemocap":     dict(speakers=0, labels=1, roberta=(2, 3, 4, 5), sent=6, train=7, test=8, val=9, n_emo=6),
    "meld":        dict(speakers=0, labels=1, roberta=(3, 4, 5, 6), sent=7, train=8, test=9, val=10, n_emo=7),
    "emorynlp":    dict(speakers=0, labels=1, roberta=(2, 3, 4, 5), sent=6, train=7, test=8, val=9, n_emo=7),
    "dailydialog": dict(speakers=0, labels=1, roberta=(2, 3, 4, 5), sent=6, train=7, test=8, val=9, n_emo=7),
}


def find_exact_duplicate_dialogues(path: str | Path, dataset: str) -> dict:
    """Cross-split contamination check: dialogues whose full utterance-text sequence exactly
    matches some train dialogue. Returns counts and the offending test/val dialogue ids
    (dropped by load_cosmic when decontaminate=True)."""
    dataset = dataset.lower()
    idx = _COSMIC_IDX[dataset]
    with open(path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")
    sent = obj[idx["sent"]]
    # Some upstream pickles store split IDs as sets. Sorting makes cache order
    # independent of PYTHONHASHSEED and therefore stable across worker processes.
    train_ids = sorted(obj[idx["train"]], key=str)
    test_ids = sorted(obj[idx["test"]], key=str)
    val_ids = sorted(obj[idx["val"]], key=str)

    def key(vid):
        return " | ".join(sent[vid])

    train_keys = {key(v) for v in train_ids}
    test_dup = [v for v in test_ids if key(v) in train_keys]
    val_dup = [v for v in val_ids if key(v) in train_keys]
    return {"dataset": dataset, "n_train": len(train_ids), "n_test": len(test_ids),
            "n_val": len(val_ids), "test_dup_ids": test_dup, "val_dup_ids": val_dup,
            "test_dup_frac": len(test_dup) / max(len(test_ids), 1),
            "val_dup_frac": len(val_dup) / max(len(val_ids), 1)}


def _iemocap_speaker(vid: str, raw) -> str:
    """Global actor id = session prefix + gender, e.g. 'Ses03'+'F' -> 'Ses03_F'."""
    return f"{str(vid)[:5]}_{raw}"


def load_cosmic(path: str | Path, dataset: str, feature: str = "roberta1",
                decontaminate: bool = False) -> ShiftSplit:
    """feature: 'roberta1' (1024-d) or 'roberta_all' (concat roberta1-4, 4096-d).
    decontaminate=True: drop test/val dialogues that exactly duplicate a train dialogue's
    text (DailyDialog is affected -- 13.4% of its test dialogues are duplicates). Logged."""
    dataset = dataset.lower()
    idx = _COSMIC_IDX[dataset]
    with open(path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")
    r_idx = idx["roberta"] if feature == "roberta_all" else idx["roberta"][:1]
    speakers, labels = obj[idx["speakers"]], obj[idx["labels"]]
    feat_dicts = [obj[i] for i in r_idx]
    # Upstream split containers are not uniformly ordered (some are sets).
    # A stable order is required for portable, self-indexed prediction caches.
    train_ids = sorted(obj[idx["train"]], key=str)
    test_ids = sorted(obj[idx["test"]], key=str)
    val_ids = sorted(obj[idx["val"]], key=str)

    if decontaminate:
        sent = obj[idx["sent"]]
        key = lambda vid: " | ".join(sent[vid])
        train_keys = {key(v) for v in train_ids}
        n_test_before, n_val_before = len(test_ids), len(val_ids)
        test_ids = [v for v in test_ids if key(v) not in train_keys]
        val_ids = [v for v in val_ids if key(v) not in train_keys]
        print(f"[decontaminate] {dataset}: dropped {n_test_before - len(test_ids)}/{n_test_before} "
              f"test dialogues, {n_val_before - len(val_ids)}/{n_val_before} val dialogues "
              f"(exact duplicate of a train dialogue).")

    def build(ids) -> list[Dialogue]:
        out = []
        for vid in ids:
            X = np.concatenate([np.asarray(fd[vid], dtype=np.float32) for fd in feat_dicts], axis=1)
            arr = np.asarray(speakers[vid])
            if dataset == "iemocap":                    # 'M'/'F' -> global actor (session+gender)
                spk = np.array([_iemocap_speaker(vid, s) for s in speakers[vid]], dtype=object)
            elif arr.ndim == 2:                         # one-hot [T,n] -> dialogue-local role
                local = arr.argmax(axis=1)
                spk = np.array([f"{vid}::role_{int(s)}" for s in local], dtype=object)
            else:                                       # scalar dialogue-local role (DailyDialog)
                spk = np.array([f"{vid}::role_{s}" for s in arr], dtype=object)
            out.append(Dialogue(did=str(vid), features={"text": X},
                                labels=np.asarray(labels[vid], dtype=np.int64), speakers=spk))
        return out

    train, val, test = build(train_ids), build(val_ids), build(test_ids)
    return ShiftSplit(train, val, test, ("text",), idx["n_emo"], X_dim(train))


def X_dim(dialogues: list[Dialogue]) -> int:
    return sum(arr.shape[1] for arr in dialogues[0].features.values())


# --------------------------------------------------------------------------- #
# Multimodal features (MM-DFN / DialogueRNN): text + audio (OpenSmile) + visual.
# IEMOCAP_features.pkl  len9 : [1]spk('M'/'F') [2]label(0-5) [3]text100 [4]audio1582
#                              [5]visual342 [7]trainVid(120) [8]testVid(31)  no val
# MELD_features_raw1.pkl len10: [1]spk[T,9] [2]emotion(0-6) [3]text600 [4]audio300
#                              [5]visual342 [7]trainVid(1152) [8]testVid(280) no val
# Neither has a val split -> carve a deterministic slice of train as val.
# --------------------------------------------------------------------------- #
_MMDFN_IDX = {
    "iemocap": dict(spk=1, label=2, text=3, audio=4, visual=5, train=7, test=8, n_emo=6),
    "meld":    dict(spk=1, label=2, text=3, audio=4, visual=5, train=7, test=8, n_emo=7),
}


def load_mmdfn(path: str | Path, dataset: str,
               modalities: tuple[str, ...] = ("text", "audio", "visual"),
               val_frac: float = 0.1, seed: int = 0) -> ShiftSplit:
    dataset = dataset.lower()
    idx = _MMDFN_IDX[dataset]
    with open(path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")
    speakers, labels = obj[idx["spk"]], obj[idx["label"]]
    mod_dicts = {m: obj[idx[m]] for m in modalities}
    train_ids = sorted(obj[idx["train"]], key=str)
    test_ids = sorted(obj[idx["test"]], key=str)

    # deterministic val carve from train
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_ids))
    n_val = max(1, int(len(train_ids) * val_frac))
    val_set = {train_ids[i] for i in perm[:n_val]}
    tr_ids = [v for v in train_ids if v not in val_set]
    va_ids = [v for v in train_ids if v in val_set]

    def build(ids) -> list[Dialogue]:
        out = []
        for vid in ids:
            feats = {m: np.asarray(mod_dicts[m][vid], dtype=np.float32) for m in modalities}
            raw_spk = speakers[vid]
            if dataset == "iemocap":
                spk = np.array([_iemocap_speaker(vid, s) for s in raw_spk], dtype=object)
            else:
                local = np.asarray(raw_spk).argmax(axis=1)
                spk = np.array([f"{vid}::role_{int(s)}" for s in local], dtype=object)
            out.append(Dialogue(did=str(vid), features=feats,
                                labels=np.asarray(labels[vid], dtype=np.int64), speakers=spk))
        return out

    train, val, test = build(tr_ids), build(va_ids), build(test_ids)
    return ShiftSplit(train, val, test, tuple(modalities), idx["n_emo"], X_dim(train))


# --------------------------------------------------------------------------- #
# Shift labels + feature assembly
# --------------------------------------------------------------------------- #
def shift_targets(labels: np.ndarray) -> np.ndarray:
    """shift[n] = 1[y_{n+1} != y_n] for n<T-1; shift[T-1] = IGNORE_INDEX."""
    T = len(labels)
    s = np.full(T, IGNORE_INDEX, dtype=np.int64)
    if T >= 2:
        s[:-1] = (labels[1:] != labels[:-1]).astype(np.int64)
    return s


def targets_for_dialogue(d: Dialogue) -> np.ndarray:
    return (d.custom_shift_targets if d.custom_shift_targets is not None
            else shift_targets(d.labels))


def target_index_for_dialogue(d: Dialogue, n: int) -> int:
    if d.custom_target_indices is not None:
        idx = int(d.custom_target_indices[n])
        if idx < 0:
            raise ValueError(f"{d.did}:{n} has no valid custom target index")
        return idx
    return n + 1


def apply_self_shift_target(dialogues: list[Dialogue]) -> None:
    """Set n -> current speaker's next-own-utterance target in place."""
    for d in dialogues:
        T = len(d.labels)
        targets = np.full(T, IGNORE_INDEX, dtype=np.int64)
        indices = np.full(T, -1, dtype=np.int64)
        next_for = {}
        for n in range(T - 1, -1, -1):
            speaker = d.speakers[n]
            if speaker in next_for:
                j = next_for[speaker]
                indices[n] = j
                targets[n] = int(d.labels[j] != d.labels[n])
            next_for[speaker] = n
        d.custom_shift_targets = targets
        d.custom_target_indices = indices


def assemble_inputs(d: Dialogue, modalities: tuple[str, ...], num_emotions: int,
                    current_emotion: str = "none") -> np.ndarray:
    """Concatenate modality features (early fusion) and optionally append the current
    emotion one-hot. current_emotion ∈ {'none','oracle','predicted'}."""
    X = np.concatenate([d.features[m] for m in modalities], axis=1)  # [T, D]
    if current_emotion == "none":
        return X
    if current_emotion == "oracle":
        idx = d.labels
    elif current_emotion == "predicted":
        if d.predicted_labels is None:
            raise ValueError(f"{d.did}: 'predicted' setting needs predicted_labels (run an ERC pass first)")
        idx = d.predicted_labels
    else:
        raise ValueError(current_emotion)
    onehot = np.eye(num_emotions, dtype=np.float32)[idx]  # [T, num_emotions]
    return np.concatenate([X, onehot], axis=1)


def input_dim(split: ShiftSplit, current_emotion: str) -> int:
    return split.feature_dim + (split.num_emotions if current_emotion != "none" else 0)
