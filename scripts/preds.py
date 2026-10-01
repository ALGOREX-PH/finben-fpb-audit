"""Prediction files with their gold labels attached. No GPU.

Shipped prediction CSVs hold ids, raw replies, parses and probabilities only: FinBen FPB is gated and the
fresh sentences are copyrighted. Every report reads them through here, which joins the gold labels back
from the local data (data/finben_*.csv, data/fresh/labels.csv).

  import preds
  df = preds.read(path)                         # FinBen split (validation, test, train_sample)
  df = preds.read(path, preds.fresh_gold())     # fresh 2026 set, gold = data/fresh/labels.csv
"""
from functools import cache

import pandas as pd

import config
from ftlib.finben import attach_gold


@cache
def finben_gold():
    """id -> gold label for every FinBen FPB sentence (train, validation and test ids are disjoint)."""
    frames = [pd.read_csv(config.DATA_DIR / f, usecols=["id", "answer"]) for f in config.SPLIT_FILES.values()]
    g = pd.concat(frames, ignore_index=True)
    assert g.id.is_unique, "FinBen ids repeat across splits"
    return g.set_index("id").answer.str.strip().str.lower()


def fresh_gold(labels=None):
    """id -> gold label for the fresh set: the adjudicated 'final' label if set, else the blind 'human' one.
    Excluded or unlabelled sentences come back as '' (callers drop anything outside config.LABELS)."""
    lab = pd.read_csv(labels or config.DATA_DIR / "fresh" / "labels.csv", keep_default_na=False, dtype=str)
    return lab.set_index("id").final.where(lambda s: s != "", lab.set_index("id").human)


def read(path, gold=None):
    """Prediction CSV at `path` with a `gold` column, or None if the file doesn't exist."""
    if not path.exists():
        return None
    return attach_gold(pd.read_csv(path, keep_default_na=False), finben_gold() if gold is None else gold)
