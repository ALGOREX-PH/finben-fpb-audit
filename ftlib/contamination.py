"""Test-set contamination detectors: is a benchmark sentence (or a near copy) in my training data?

Three detectors, from strict to loose. Report all three -- they catch different things:
  exact     same text after normalisation (case, punctuation, spacing, encoding junk removed)
  near_dup  character n-gram TF-IDF cosine >= threshold (catches small edits, re-tokenisation)
  ngram     shares at least one word n-gram of length n (the GPT-3 paper's approach, with n=13
            for long documents; short sentences need a smaller n). LOOSE: on formulaic text (newswire
            boilerplate like "Oyj said ... in the first nine months of") it flags different sentences
            that share a template phrase. Inspect its extra hits by eye; don't count them blindly.
Columns: contaminated = exact | near_dup (the primary definition); loose = contaminated | ngram.

Usage:
    from ftlib.contamination import normalize, contamination_table
    table = contamination_table(test_texts, train_texts, near_dup_threshold=0.9, ngram_n=8)
"""
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def normalize(text: str) -> str:
    """Letters and digits only, lower case. Makes 'per+ñmeri' and 'per+æmeri' (encoding junk) match."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split())


def _ngrams(text: str, n: int) -> set:
    words = normalize(text).split()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def contamination_table(test_texts, train_texts, near_dup_threshold=0.9, ngram_n=8, chunk=512):
    """One row per test text: which detectors fire, the closest training text, and its similarity.

    Returns a DataFrame with columns:
      exact, near_dup, ngram, contaminated (= exact | near_dup), loose (= any of the three),
      max_cosine (float), nearest_train (int index into train_texts)
    """
    test_texts, train_texts = list(test_texts), list(train_texts)
    train_norm = [normalize(t) for t in train_texts]
    test_norm = [normalize(t) for t in test_texts]
    train_set = set(train_norm)

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(train_norm + test_norm)
    X_train, X_test = vec.transform(train_norm), vec.transform(test_norm)
    best_sim, best_idx = np.zeros(len(test_texts)), np.zeros(len(test_texts), dtype=int)
    for s in range(0, len(test_texts), chunk):
        sims = (X_test[s:s + chunk] @ X_train.T).toarray()
        best_idx[s:s + chunk] = sims.argmax(1)
        best_sim[s:s + chunk] = sims.max(1)

    train_grams = set().union(*(_ngrams(t, ngram_n) for t in train_texts)) if train_texts else set()
    table = pd.DataFrame({
        "exact": [t in train_set for t in test_norm],
        "near_dup": best_sim >= near_dup_threshold,
        "ngram": [bool(_ngrams(t, ngram_n) & train_grams) for t in test_texts],
        "max_cosine": best_sim.round(4),
        "nearest_train": best_idx,
    })
    table["contaminated"] = table.exact | table.near_dup
    table["loose"] = table.contaminated | table.ngram
    return table
