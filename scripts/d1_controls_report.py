"""Part D report: results/partD/REPORT.md -- does training on the test set, or training harder, reproduce
FinMA-7B's drop from FinBen's test to the fresh 2026 sentences? No GPU, seconds.

  uv run python scripts/d1_controls_report.py
  uv run python scripts/d1_controls_report.py --labels data/fresh/labels_human.csv --out results/partD/REPORT_human.md

Pre-registered in ANALYSIS.md, Part D (git tag prereg-partD), before any control model was trained:
  cells     A clean 2 epochs (the Part B final models)   B contaminated 2 epochs (train + validation + TEST)
            C clean 15 epochs                             D contaminated 15 epochs
  drop(X)   fresh wF1 - FinBen test wF1
  DiD15     drop(C) - drop(D)    DiD2  drop(A) - drop(B)    positive = training on the test set inflates FinBen
  verdict   on seed 3407: MEMORISATION if DiD15 >= 0.03 with CI excluding 0, drop(D) FinMA-sized and drop(C) not;
            OVER-SPECIALISATION if DiD15's CI includes 0 and both drops are FinMA-sized; else INCONCLUSIVE.
            FinMA-sized = drop <= -0.117 (within 3 points of FinMA's -0.147, or worse).
CIs: 2,000 bootstrap draws; the same resampled indices for every system within a test set (paired), FinBen test
and fresh set resampled independently of each other (as c4_report.py does).
Missing control predictions are reported as missing: the report can be run at any point of the Part D runs.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config_b as config
import preds
from config_b import LABELS
from ftlib.finben import AGREEMENT_LEVELS, accuracy_by_agreement

N_BOOT = 2000
PRIMARY_SEED = 3407
MIN_DID = 0.03
FINMA_SIZED = -0.117
CELLS = {"A": (2, "tv", "clean, 2 epochs"), "B": (2, "tvt", "contaminated, 2 epochs"),
         "C": (15, "tv", "clean, 15 epochs"), "D": (15, "tvt", "contaminated, 15 epochs")}
SPLITS = ["test", "train_sample", "fresh"]
PART_C_PRED = config.ROOT / "results" / "partC" / "predictions"


def run_name(cell, seed, sel):
    epochs, data, _ = CELLS[cell]
    return f"e{epochs}_lr{sel['lr']:g}_r{sel['rank']}_s{seed}_{data}"


def locate(cell, seed, split, sel, pred_root):
    """Prediction file of one cell / seed / split. Cell A's test and fresh files are the shipped Part B/C results."""
    name = run_name(cell, seed, sel)
    if cell == "A" and split == "test":
        return config.PRED_DIR / "test" / f"{name}.csv"
    if cell == "A" and split == "fresh":
        return PART_C_PRED / f"{name}.csv"
    return pred_root / split / f"{name}.csv"


CODE = {label: i for i, label in enumerate(LABELS)}          # anything else ('missing', '') -> len(LABELS)
K = len(LABELS) + 1


def encode(labels):
    return np.array([CODE.get(x, len(LABELS)) for x in labels], dtype=np.int64)


def wf1(g, p):
    """Weighted F1 on int-coded labels, identical to sklearn's f1_score(average='weighted', zero_division=0)."""
    cm = np.bincount(g * K + p, minlength=K * K).reshape(K, K)
    tp, support, predicted = np.diag(cm), cm.sum(1), cm.sum(0)
    prec = np.divide(tp, predicted, out=np.zeros(K), where=predicted > 0)
    rec = np.divide(tp, support, out=np.zeros(K), where=support > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(K), where=prec + rec > 0)
    return float((f1 * support).sum() / support.sum())


class Scored:
    """One system's FinBen test and fresh predictions, with bootstrap replicates on the shared indices."""

    def __init__(self, test_g, test_p, fresh_g, fresh_p, idx_test, idx_fresh):
        self.test, self.fresh = wf1(test_g, test_p), wf1(fresh_g, fresh_p)
        self.test_boot = np.array([wf1(test_g[i], test_p[i]) for i in idx_test])
        self.fresh_boot = np.array([wf1(fresh_g[j], fresh_p[j]) for j in idx_fresh])

    @property
    def drop(self):
        return self.fresh - self.test

    @property
    def drop_boot(self):
        return self.fresh_boot - self.test_boot


def ci(values):
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(lo), float(hi)


def fmt_ci(point, values):
    lo, hi = ci(values)
    return f"{point:.4f} [{lo:.3f}, {hi:.3f}]"
