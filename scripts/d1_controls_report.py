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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path,
                    help="fresh-set label file (default data/fresh/labels.csv, AI-made); pass human labels later")
    ap.add_argument("--out", type=Path, default=config.PARTD_DIR / "REPORT.md")
    ap.add_argument("--pred-root", type=Path, default=config.PARTD_DIR / "predictions",
                    help="where the control predictions live (<pred-root>/<split>/<run>.csv)")
    args = ap.parse_args()

    pred_root, out = args.pred_root, args.out
    sel = json.loads(config.SELECTED.read_text())
    labels_file = args.labels or config.DATA_DIR / "fresh" / "labels.csv"
    lab = pd.read_csv(labels_file, keep_default_na=False, dtype=str)
    ai_labels = "labeled_by" in lab and lab.labeled_by.str.contains("claude", case=False).any()
    fresh_gold = preds.fresh_gold(labels_file)

    # ---- data: FinBen test gold + agreement, fresh gold, train sample
    test = pd.read_csv(config.DATA_DIR / config.SPLIT_FILES["test"])
    test_ids, test_g = test.id.values, encode(test.answer.str.strip().str.lower())
    cand = pd.read_csv(config.DATA_DIR / "fresh" / "candidates.csv")
    fresh_label = fresh_gold.reindex(cand.id).fillna("").values
    keep = np.isin(fresh_label, LABELS)
    fresh_ids, fresh_g = cand.id.values[keep], encode(fresh_label[keep])
    rng = np.random.default_rng(0)
    idx_test = rng.integers(0, len(test_g), (N_BOOT, len(test_g)))
    idx_fresh = rng.integers(0, len(fresh_g), (N_BOOT, len(fresh_g)))

    def load(path, gold=None, ids=None):
        d = preds.read(path, gold)
        if d is None:
            return None
        if ids is not None:
            d = d.set_index("id").reindex(ids)
            assert d.pred.notna().all(), f"{path} is missing sentences"
            d = d.reset_index()
        return d

    # ---- every system with all three splits: cells x seeds, plus FinMA
    systems, missing = {}, []
    for cell in CELLS:
        for seed in config.SEEDS:
            files = {s: locate(cell, seed, s, sel, pred_root) for s in SPLITS}
            have = {s: f for s, f in files.items() if f.exists()}
            if len(have) < len(SPLITS):
                missing.append((cell, seed, [s for s in SPLITS if s not in have]))
                continue
            systems[(cell, seed)] = {"test": load(files["test"], ids=test_ids), "train_sample": load(files["train_sample"]),
                                     "fresh": load(files["fresh"], fresh_gold, ids=fresh_ids)}
    finma = {"test": load(config.PRED_DIR / "test" / "finma_7b.csv", ids=test_ids),
             "train_sample": load(config.PRED_DIR / "train_sample" / "finma_7b.csv"),
             "fresh": load(PART_C_PRED / "finma_7b.csv", fresh_gold, ids=fresh_ids)}

    def score(s):
        return Scored(test_g, encode(s["test"].pred), fresh_g, encode(s["fresh"].pred), idx_test, idx_fresh)

    scored = {k: score(s) for k, s in systems.items()}
    finma_scored = score(finma)

    lines = ["# Part D: saw the test set, or trained harder? (generated by `d1_controls_report.py`)", "",
             f"Pre-registered in ANALYSIS.md, Part D (git tag `prereg-partD`). Fresh-set labels: `{labels_file.name}`"
             + (" -- **AI-made (Claude); human spot-check pending, so every verdict below is preliminary.**" if ai_labels
                else "."), "",
             f"Fresh sentences used: {keep.sum()} of {len(cand)} (the rest are excluded or unlabelled). "
             f"FinBen test: {len(test_g)}. Recipe for every cell: lr {sel['lr']:g}, rank {sel['rank']}.", ""]
    if missing:
        lines += ["**Not run yet:** " + "; ".join(f"{c} seed {s} ({', '.join(m)})" for c, s, m in missing), ""]

    # ---- 1. per system
    def acc(d):
        return float((d.pred.values == d.gold.values).mean())

    lines += ["## 1. Scores per cell and seed", "",
              "| Cell | Seed | FinBen test wF1 [95% CI] | Train-sample acc | Test acc | Gap (train − test) | "
              "Fresh wF1 [95% CI] | Drop (fresh − FinBen) [95% CI] |", "|---|---|---|---|---|---|---|---|"]

    def row(label, seed, s, sc):
        a_tr, a_te = acc(s["train_sample"]), acc(s["test"])
        return (f"| {label} | {seed} | {fmt_ci(sc.test, sc.test_boot)} | {a_tr:.4f} | {a_te:.4f} | {a_tr - a_te:+.4f} | "
                f"{fmt_ci(sc.fresh, sc.fresh_boot)} | {sc.drop:+.4f} [{ci(sc.drop_boot)[0]:+.3f}, {ci(sc.drop_boot)[1]:+.3f}] |")

    for (cell, seed), s in systems.items():
        lines.append(row(f"**{cell}** {CELLS[cell][2]}", seed, s, scored[(cell, seed)]))
    lines += [row("FinMA-7B (official prompt)", "—", finma, finma_scored), "",
              "*Every model trained on the train sample, so its gap is train-vs-test familiarity. Contaminated cells "
              "(B, D) also trained on the test sentences: their gap should be near 0, like a model that saw both.*", ""]

    # ---- 2. accuracy by annotator agreement (FinBen test)
    agree = test.set_index("id").agreement.reindex(test_ids).values
    lines += ["## 2. FinBen test accuracy by annotator agreement", "",
              "| Cell | Seed | " + " | ".join(AGREEMENT_LEVELS) + " |", "|---|---|" + "---|" * len(AGREEMENT_LEVELS)]

    def agree_row(label, seed, d):
        by = accuracy_by_agreement(d.pred, d.gold, agree)
        return f"| {label} | {seed} | " + " | ".join(f"{by[lv][0]:.3f}" for lv in AGREEMENT_LEVELS) + " |"

    for (cell, seed), s in systems.items():
        lines.append(agree_row(f"**{cell}**", seed, s["test"]))
    n_by = accuracy_by_agreement(finma["test"].pred, finma["test"].gold, agree)
    lines += [agree_row("FinMA-7B", "—", finma["test"]), "",
              "*n per level: " + ", ".join(f"{lv} {n_by[lv][1]}" for lv in AGREEMENT_LEVELS) + ". On 50Agree the "
              "annotators split, so a high score there means the model knows the label rather than reads it off.*", ""]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
