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

    # ---- 3. contrasts, seed-matched
    def contrast(x, y, seeds=None):
        """mean over common seeds of drop(x) - drop(y), its bootstrap CI and the seeds used (None if no common seed)."""
        common = [s for s in config.SEEDS if (x, s) in scored and (y, s) in scored and (seeds is None or s in seeds)]
        if not common:
            return None
        point = np.mean([scored[(x, s)].drop - scored[(y, s)].drop for s in common])
        boot = np.mean([scored[(x, s)].drop_boot - scored[(y, s)].drop_boot for s in common], axis=0)
        return float(point), ci(boot), common

    contrasts = [("DiD15 = drop(C) − drop(D)", "C", "D", "contamination effect at 15 epochs (primary)"),
                 ("DiD2 = drop(A) − drop(B)", "A", "B", "contamination effect at 2 epochs (secondary)"),
                 ("Intensity = drop(A) − drop(C)", "A", "C", "cost of training 15 vs 2 epochs, clean models")]
    lines += ["## 3. Contrasts (positive DiD = training on the test set inflated the FinBen score)", "",
              "| Contrast | Meaning | Seed 3407 [95% CI] | Seed-matched mean [95% CI] (seeds) |", "|---|---|---|---|"]
    results = {}
    for label, x, y, meaning in contrasts:
        primary, matched = contrast(x, y, [PRIMARY_SEED]), contrast(x, y)
        results[label] = primary
        show = lambda r: "*not run yet*" if r is None else f"{r[0]:+.4f} [{r[1][0]:+.3f}, {r[1][1]:+.3f}]"
        seeds = "" if matched is None else f" ({', '.join(map(str, matched[2]))})"
        lines.append(f"| {label} | {meaning} | {show(primary)} | {show(matched)}{seeds} |")
    lines += ["", f"*FinMA's own drop: {finma_scored.drop:+.4f}. FinMA-sized = drop ≤ {FINMA_SIZED}. Bootstrap CIs "
                  "cover sentence sampling only, not seed-to-seed variation (cell A's seeds alone span several points).*", ""]

    # ---- 4. the pre-registered predictions, checked
    g = lambda cell: scored.get((cell, PRIMARY_SEED))
    gap = lambda cell: acc(systems[(cell, PRIMARY_SEED)]["train_sample"]) - acc(systems[(cell, PRIMARY_SEED)]["test"])
    a50 = lambda cell: accuracy_by_agreement(systems[(cell, PRIMARY_SEED)]["test"].pred,
                                             systems[(cell, PRIMARY_SEED)]["test"].gold, agree)["50Agree"][0]
    did15, did2 = results["DiD15 = drop(C) − drop(D)"], results["DiD2 = drop(A) − drop(B)"]
    checks = [
        ("D1", "D test wF1 ≥ 0.97", ["D"], lambda: (g("D").test >= 0.97, f"{g('D').test:.4f}")),
        ("D2", "B test wF1 0.96 ± 0.02, and A < B ≤ D", ["A", "B", "D"],
         lambda: (abs(g("B").test - 0.96) <= 0.02 and g("A").test < g("B").test <= g("D").test,
                  f"A {g('A').test:.4f}, B {g('B').test:.4f}, D {g('D').test:.4f}")),
        ("D3", "C within 0.02 of A (test wF1)", ["A", "C"],
         lambda: (abs(g("C").test - g("A").test) <= 0.02, f"{g('C').test - g('A').test:+.4f}")),
        ("D4", "gap B and D within ±0.02 of 0; gap C ≥ gap A", ["A", "B", "C", "D"],
         lambda: (abs(gap("B")) <= 0.02 and abs(gap("D")) <= 0.02 and gap("C") >= gap("A"),
                  f"A {gap('A'):+.3f}, B {gap('B'):+.3f}, C {gap('C'):+.3f}, D {gap('D'):+.3f}")),
        ("D5", "50Agree: D ≥ 0.90; C within 0.07 of A", ["A", "C", "D"],
         lambda: (a50("D") >= 0.90 and abs(a50("C") - a50("A")) <= 0.07,
                  f"A {a50('A'):.3f}, C {a50('C'):.3f}, D {a50('D'):.3f}")),
        ("D6", "drop(C) = −0.08 ± 0.04", ["C"], lambda: (abs(g("C").drop + 0.08) <= 0.04, f"{g('C').drop:+.4f}")),
        ("D7", "DiD15 ≥ +0.05 and DiD2 ≥ +0.04, both CIs excluding 0", ["A", "B", "C", "D"],
         lambda: (did15[0] >= 0.05 and did15[1][0] > 0 and did2[0] >= 0.04 and did2[1][0] > 0,
                  f"DiD15 {did15[0]:+.4f} [{did15[1][0]:+.3f}, {did15[1][1]:+.3f}], "
                  f"DiD2 {did2[0]:+.4f} [{did2[1][0]:+.3f}, {did2[1][1]:+.3f}]")),
    ]
    lines += ["## 4. Pre-registered predictions (seed 3407)", "", "| # | Prediction | Observed | Outcome |", "|---|---|---|---|"]
    for key, text, needs, check in checks:
        if all(g(c) is not None for c in needs):
            ok, observed = check()
            lines.append(f"| {key} | {text} | {observed} | {'**confirmed**' if ok else '**failed**'} |")
        else:
            lines.append(f"| {key} | {text} | — | *not run yet* |")
    lines.append("")

    # ---- 5. verdict
    prelim = " (preliminary: AI labels)" if ai_labels else ""
    if g("C") is None or g("D") is None:
        verdict, detail = "NOT READY: cells C and D (seed 3407) are needed", []
    else:
        d_c, d_d, (did, (lo, hi), _) = g("C").drop, g("D").drop, did15
        if did >= MIN_DID and lo > 0 and d_d <= FINMA_SIZED and d_c > FINMA_SIZED:
            verdict = "MEMORISATION explains the pattern" + prelim
        elif lo <= 0 <= hi and d_c <= FINMA_SIZED and d_d <= FINMA_SIZED:
            verdict = "OVER-SPECIALISATION explains the pattern" + prelim
        else:
            verdict = "INCONCLUSIVE" + prelim
        detail = [f"- drop(C) {d_c:+.4f}, drop(D) {d_d:+.4f}, FinMA {finma_scored.drop:+.4f} (FinMA-sized: ≤ {FINMA_SIZED})",
                  f"- DiD15 {did:+.4f}, 95% CI [{lo:+.3f}, {hi:+.3f}]", ""]
    lines += ["## 5. Verdict (rule fixed before the runs)", "", *detail, f"**{verdict}.**", "",
              "*Memorisation: DiD15 ≥ 0.03 with CI excluding 0, drop(D) FinMA-sized, drop(C) not. Over-specialisation: "
              "DiD15's CI includes 0 and both drops FinMA-sized. Anything else: inconclusive. Part D shows what "
              "memorisation and over-training do to OUR model; it can show memorisation can produce FinMA's pattern, "
              "not that it did for FinMA.*", ""]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
