"""Strictly utterance-local text features fitted on training utterances only."""
from __future__ import annotations

import pickle

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer

from .dataset import _COSMIC_IDX, load_cosmic


def load_tfidf_svd(path: str, dataset: str, max_features: int = 20000,
                   n_components: int = 128, decontaminate: bool = False):
    """Return a ShiftSplit whose features depend on one utterance at a time.

    Vocabulary, IDF, and SVD are fitted on training utterances only. Dialogue order,
    speaker IDs, labels, and splits come from the same audited COSMIC pickle used by
    the main experiment. No context, neighbouring utterance, or outcome is supplied
    to the feature transform.
    """
    split = load_cosmic(path, dataset, "roberta1", decontaminate=decontaminate)
    with open(path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")
    idx = _COSMIC_IDX[dataset]
    sentences = obj[idx["sent"]]

    def sentence_rows(did):
        """Resolve release-specific key types without relying on implicit coercion."""
        if did in sentences:
            return sentences[did]
        try:
            numeric = int(did)
        except (TypeError, ValueError):
            numeric = None
        if numeric is not None and numeric in sentences:
            return sentences[numeric]
        raise KeyError(f"No sentence entry aligned to dialogue id {did!r}")

    train_text = [str(s) for d in split.train for s in sentence_rows(d.did)]
    vectorizer = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=2, max_features=max_features,
        sublinear_tf=True, strip_accents="unicode", dtype=np.float64,
    )
    train_sparse = vectorizer.fit_transform(train_text)
    n_comp = min(n_components, max(2, train_sparse.shape[1] - 1))
    # ARPACK avoids unstable randomized power iterations observed with Apple's
    # Accelerate BLAS. It is an officially supported TruncatedSVD solver and is
    # deterministic here because the sparse matrix and requested rank are fixed.
    svd = TruncatedSVD(n_components=n_comp, algorithm="arpack", tol=0.0,
                       random_state=0)
    svd.fit(train_sparse)
    normalizer = Normalizer(copy=False)

    def replace(dialogues):
        for d in dialogues:
            sparse = vectorizer.transform([str(s) for s in sentence_rows(d.did)])
            dense = normalizer.transform(svd.transform(sparse)).astype(np.float32)
            if not np.isfinite(dense).all():
                raise FloatingPointError(f"Non-finite local features for dialogue {d.did!r}")
            d.features = {"text": dense}

    for part in (split.train, split.val, split.test):
        replace(part)
    split.feature_dim = n_comp
    metadata = {
        "name": "train-only utterance-local TF-IDF(1,2)+TruncatedSVD",
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "components": int(n_comp),
        "svd_solver": "arpack",
        "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
        "fit_utterances": int(len(train_text)),
        "decontaminated": bool(decontaminate),
    }
    return split, metadata
