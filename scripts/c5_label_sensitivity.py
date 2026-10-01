"""Part C, step 5: how sensitive is the Part C verdict to the (AI-made) fresh-set labels? No GPU, seconds.

  uv run python scripts/c5_label_sensitivity.py
  uv run python scripts/c5_label_sensitivity.py --labels data/fresh/labels_human.csv   # same analysis, other labels

THIS IS A SENSITIVITY ANALYSIS, NOT A RE-LABELLING. The gold labels stay exactly as they are; the script asks
how much the conclusions move if the labels or the reference numbers were different. The fresh-set gold
labels are AI-made (Claude) and the human spot-check is still pending.

Definitions are c4_report.py's, replicated exactly: weighted F1 (zero_division=0), "ours" = the frozen 3-seed
ensemble (mean label probabilities -> argmax), sentences labelled outside negative/neutral/positive dropped,
MIN_SHRINK = 0.03, 2000 bootstrap replicates from numpy default_rng(0).
Output: results/partC/SENSITIVITY.md (numbers and sentence ids only, never sentence text).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, f1_score

import config_b as config
import preds
from config_b import LABELS

C_DIR = config.ROOT / "results" / "partC"
MIN_SHRINK = 0.03          # c4_report.py's decision threshold
N_BOOT, BOOT_SEED = 2000, 0   # c4_report.py's bootstrap settings
N_ORDERS = 50              # random relabelling orders for the tipping-point CI check
CLASSES = LABELS + ["missing"]   # FinBen's parser outputs 'missing' when no label is found


def wf1(g, p):
    return f1_score(g, p, average="weighted", zero_division=0)


def ensemble(dfs):
    return np.array(LABELS)[np.mean([d[[f"p_{l}" for l in LABELS]].values for d in dfs], axis=0).argmax(1)]


def mcnemar(g, a, b):
    """c4_report.py's exact McNemar: binomial test on the sentences exactly one of the two systems gets right."""
    a, b = np.asarray(a) == np.asarray(g), np.asarray(b) == np.asarray(g)
    x, y = int(np.sum(a & ~b)), int(np.sum(~a & b))
    return (binomtest(x, x + y, 0.5).pvalue if x + y else 1.0), x, y


def codes(labels):
    return np.array([CLASSES.index(x) for x in labels])


def boot_wf1(g, p, idx):
    """Weighted F1 for every bootstrap replicate at once. g, p: class codes; idx: (replicates, n) indices.
    Same numbers as sklearn's f1_score(average='weighted', zero_division=0), vectorised."""
    k = len(CLASSES)
    gi, pi = g[idx], p[idx]
    off = (np.arange(len(idx))[:, None] * k * k)
    cm = np.bincount((off + gi * k + pi).ravel(), minlength=len(idx) * k * k).reshape(len(idx), k, k)
    tp = np.diagonal(cm, axis1=1, axis2=2)
    support, predicted = cm.sum(2), cm.sum(1)
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros(tp.shape), where=denom > 0)
    return (f1 * support).sum(1) / support.sum(1)


def load(labels):
    """Fresh-set and FinBen-test predictions of every system, gold attached, aligned by sentence id."""
    sel = json.loads(config.SELECTED.read_text())
    final = [f"e{sel['epochs']}_lr{sel['lr']:g}_r{sel['rank']}_s{s}_tv" for s in config.SEEDS]
    gold = preds.fresh_gold(labels)
    fresh = {n: preds.read(C_DIR / "predictions" / f"{n}.csv", gold) for n in ["e4b_zeroshot", "finma_7b", *final]}
    test = {n: preds.read(config.PRED_DIR / "test" / f"{n}.csv") for n in ["e4b_zeroshot", "finma_7b", *final]}
    for split in (fresh, test):
        ids = next(iter(split.values())).id.values
        assert all((d.id.values == ids).all() for d in split.values()), "prediction files are not aligned by id"
    keep = np.isin(fresh["finma_7b"].gold.values, LABELS)

    def systems(split, mask):
        out = {"E4B-it zero-shot": split["e4b_zeroshot"].pred.values[mask],
               **{f"Ours, seed {s}": split[n].pred.values[mask] for s, n in zip(config.SEEDS, final)},
               "Ours, 3-seed ensemble": ensemble([split[n] for n in final])[mask],
               "FinMA-7B": split["finma_7b"].pred.values[mask]}
        return out

    g = fresh["finma_7b"].gold.values[keep]
    ids = fresh["finma_7b"].id.values[keep]
    fg = test["finma_7b"].gold.values
    return g, ids, systems(fresh, keep), fg, systems(test, np.ones(len(fg), bool))


def resample(n_fin, n_fresh):
    """c4_report.py's index stream: per replicate, FinBen indices first, then fresh indices, one rng."""
    rng = np.random.default_rng(BOOT_SEED)
    i, j = [], []
    for _ in range(N_BOOT):
        i.append(rng.integers(0, n_fin, n_fin))
        j.append(rng.integers(0, n_fresh, n_fresh))
    return np.array(i), np.array(j)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=config.DATA_DIR / "fresh" / "labels.csv")
    ap.add_argument("--out", type=Path, default=C_DIR / "SENSITIVITY.md")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")   # Windows consoles can't print every character in the report

    g, ids, fresh, fg, fin = load(args.labels)
    G = codes(g)
    P = {k: codes(v) for k, v in fresh.items()}
    for k in P:   # the vectorised F1 must equal sklearn's
        assert abs(boot_wf1(G, P[k], np.arange(len(G))[None])[0] - wf1(g, fresh[k])) < 1e-12, k

    lines = ["# Part C label sensitivity (generated by `c5_label_sensitivity.py`)", "",
             "> **This is a sensitivity analysis, not a re-labelling.** The fresh-set answer key is unchanged. It was "
             "made by an LLM (Claude), and the human spot-check is still pending. Each section asks how far a "
             "conclusion would move if labels or reference numbers were different.", "",
             f"Labels: `{(args.labels.relative_to(config.ROOT) if args.labels.is_relative_to(config.ROOT) else args.labels).as_posix()}`; "
             f"{len(g)} labelled sentences (negative {np.sum(g == 'negative')}, neutral {np.sum(g == 'neutral')}, "
             f"positive {np.sum(g == 'positive')}). \"Ours\" = the frozen 3-seed ensemble unless a seed is named.", ""]

    # ---- a) FinMA's lead on the fresh set alone, paired bootstrap
    ours, finma = P["Ours, 3-seed ensemble"], P["FinMA-7B"]
    lead = wf1(g, fresh["FinMA-7B"]) - wf1(g, fresh["Ours, 3-seed ensemble"])
    rng = np.random.default_rng(BOOT_SEED)
    J_own = np.array([rng.integers(0, len(g), len(g)) for _ in range(N_BOOT)])
    I_c4, J_c4 = resample(len(fg), len(g))
    lead_ci = lambda J: np.quantile(boot_wf1(G, finma, J) - boot_wf1(G, ours, J), [0.025, 0.975])
    (lo, hi), (lo4, hi4) = lead_ci(J_own), lead_ci(J_c4)
    lines += ["## a) FinMA's lead on the fresh set alone", "",
              f"FinMA − ours, weighted F1: **{lead:+.4f}**, 95% CI **[{lo:+.3f}, {hi:+.3f}]**. "
              + ("The CI includes 0: on fresh data alone, neither system is ahead." if lo <= 0 <= hi else
                 f"The CI excludes 0: on fresh data alone, {'FinMA' if lo > 0 else 'our ensemble'} is ahead."), "",
              f"*Paired bootstrap: each of {N_BOOT} replicates resamples the {len(g)} fresh sentences with replacement "
              f"and scores both systems on the same draw; percentile interval; numpy `default_rng({BOOT_SEED})` drawing "
              f"fresh indices only. Drawing them as the fresh half of c4_report.py's two-set stream instead gives "
              f"[{lo4:+.3f}, {hi4:+.3f}]: the endpoints carry about ±0.002 of Monte Carlo noise.*", ""]

    # ---- b) each of our seeds vs FinMA, McNemar on the same fresh sentences
    lines += ["## b) Each system vs FinMA-7B on the fresh set (McNemar)", "",
              "| System | Fresh wF1 | wF1 − FinMA | Only this right / only FinMA right | McNemar p (exact) |",
              "|---|---|---|---|---|"]
    for k, p in fresh.items():
        if k == "FinMA-7B":
            continue
        pv, x, y = mcnemar(g, p, fresh["FinMA-7B"])
        lines.append(f"| {k} | {wf1(g, p):.4f} | {wf1(g, p) - wf1(g, fresh['FinMA-7B']):+.4f} | {x} / {y} | {pv:.2g} |")
    ours_p = {k: mcnemar(g, p, fresh["FinMA-7B"])[0] for k, p in fresh.items() if k.startswith("Ours")}
    seeds_w = [wf1(g, p) for k, p in fresh.items() if k.startswith("Ours, seed")]
    lines += ["", f"*{'None' if min(ours_p.values()) >= 0.05 else 'Not all'} of our models "
                  f"differ significantly from FinMA on the fresh set (smallest p = {min(ours_p.values()):.2g}). Our seeds "
                  f"span {max(seeds_w) - min(seeds_w):.3f} wF1 among themselves, about the size of the FinMA gap.*", ""]

    # ---- c) tipping point: how many label changes would flip c4_report.py's verdict?
    FG = codes(fg)
    fin_f, fin_o = codes(fin["FinMA-7B"]), codes(fin["Ours, 3-seed ensemble"])
    lead_fin = wf1(fg, fin["FinMA-7B"]) - wf1(fg, fin["Ours, 3-seed ensemble"])
    lead_fin_b = boot_wf1(FG, fin_f, I_c4) - boot_wf1(FG, fin_o, I_c4)    # FinBen half of c4's stream
    lab = pd.read_csv(args.labels, keep_default_na=False, dtype=str)       # c4_report.py's AI-label test and wording
    ai_gold = "labeled_by" in lab and lab.labeled_by.str.contains("claude", case=False).any()
    consistent = "memorisation CONSISTENT (preliminary: AI labels)" if ai_gold else "memorisation CONSISTENT"

    def shrink_at(Gk):
        """c4_report.py's shrink, its 95% CI and its verdict, with fresh gold labels Gk."""
        lead_fresh = boot_wf1(Gk, finma, np.arange(len(Gk))[None])[0] - boot_wf1(Gk, ours, np.arange(len(Gk))[None])[0]
        sh_b = lead_fin_b - (boot_wf1(Gk, finma, J_c4) - boot_wf1(Gk, ours, J_c4))
        lo_, hi_ = np.quantile(sh_b, [0.025, 0.975])
        gk = np.array(CLASSES)[Gk]
        p_, _, _ = mcnemar(gk, fresh["FinMA-7B"], fresh["Ours, 3-seed ensemble"])
        shrink = lead_fin - lead_fresh
        verdict = (consistent if shrink >= MIN_SHRINK and lo_ > 0 else
                   "memorisation NOT supported" if lead_fresh >= MIN_SHRINK and p_ < 0.05 else "INCONCLUSIVE")
        return lead_fresh, shrink, lo_, hi_, verdict

    flip = np.flatnonzero((g == "neutral") & (fresh["FinMA-7B"] == "positive") & (fresh["Ours, 3-seed ensemble"] == "neutral"))
    flip = flip[np.argsort(ids[flip])]       # primary order: by sentence id (deterministic, content-blind)
    POS = CLASSES.index("positive")
    rows, first_rule, first_ci = [], None, None
    for k in range(len(flip) + 1):
        Gk = G.copy()
        Gk[flip[:k]] = POS
        lead_fresh, shrink, lo_, hi_, verdict = shrink_at(Gk)
        rows.append(f"| {k} | {lead_fresh:+.4f} | **{shrink:+.4f}** | [{lo_:+.3f}, {hi_:+.3f}] | {verdict} |")
        first_rule = k if first_rule is None and shrink < MIN_SHRINK else first_rule
        first_ci = k if first_ci is None and lo_ <= 0 else first_ci

    # the point estimates don't depend on WHICH k sentences are relabelled (they share gold, FinMA and ensemble
    # labels, so every choice gives the same confusion matrices); only the bootstrap CI can. Check over random orders.
    rng_o = np.random.default_rng(1)
    lo_range = {}
    for _ in range(N_ORDERS):
        order = rng_o.permutation(flip)
        for k in range(1, len(flip) + 1):
            Gk = G.copy()
            Gk[order[:k]] = POS
            sh_b = lead_fin_b - (boot_wf1(Gk, finma, J_c4) - boot_wf1(Gk, ours, J_c4))
            lo_range.setdefault(k, []).append(np.quantile(sh_b, 0.025))
    ci_first = [min(k for k in lo_range if lo_range[k][r] <= 0) if any(lo_range[k][r] <= 0 for k in lo_range) else None
                for r in range(N_ORDERS)]
    ci_first_known = [c for c in ci_first if c is not None]

    lines += ["## c) Tipping point: label changes needed to flip the Part C verdict", "",
              f"Candidates: the **{len(flip)}** fresh sentences labelled *neutral* that FinMA calls *positive* and our "
              "ensemble calls *neutral*. Relabelling one of them *positive* moves one sentence from \"only ours right\" "
              "to \"only FinMA right\", the change most favourable to FinMA a single label edit can make. "
              f"Sentence ids, in the order used: {', '.join(f'`{i}`' for i in ids[flip])}.", "",
              "| k relabelled | FinMA lead, fresh | Shrink | 95% CI | c4 verdict |", "|---|---|---|---|---|", *rows, "",
              f"- The shrink falls below the {MIN_SHRINK} rule at **k = {first_rule}**"
              f"{'' if first_rule is not None else ' (never, within the candidates)'}.",
              f"- The CI first includes 0 at **k = {first_ci}** in id order"
              + (f"; over {N_ORDERS} random orders, at k = {min(ci_first_known)}–{max(ci_first_known)}"
                 if ci_first_known else "") + ".",
              "- Which sentences are relabelled doesn't change the point estimates: all candidates share the same gold, "
              "FinMA and ensemble labels, so any k of them give the same confusion matrices. Only the bootstrap CI depends "
              "on the choice, through which sentences each replicate happens to draw.", "",
              f"*k = 0 reproduces c4_report.py exactly (same bootstrap stream). {len(flip)} candidates out of "
              f"{int(np.sum(g == 'neutral'))} neutral-labelled sentences; a human disagreeing with the AI on "
              f"{first_rule} of them, all in FinMA's favour, would make the result inconclusive.*", ""]

    # ---- d) FinMA's drop against its PUBLISHED FinBen score instead of our re-run
    pub = config.FINMA_PUBLISHED_F1
    w = {k: wf1(g, fresh[k]) for k in ["FinMA-7B", "Ours, 3-seed ensemble", "E4B-it zero-shot"]}
    wf = {k: wf1(fg, fin[k]) for k in w}
    lead_pub = pub - wf["Ours, 3-seed ensemble"]
    shrink_pub = lead_pub - lead
    sh_b = (pub - boot_wf1(FG, fin_o, I_c4)) - (boot_wf1(G, finma, J_c4) - boot_wf1(G, ours, J_c4))
    lo_p, hi_p = np.quantile(sh_b, [0.025, 0.975])
    lines += ["## d) FinMA's drop measured from its published score", "",
              f"Part C measures FinMA's drop from our re-run (FinBen test {wf['FinMA-7B']:.3f}). FinMA's own paper "
              f"reports {pub:.2f}. Measured from that, the drop is much smaller:", "",
              "| System | FinBen FPB reference | Fresh wF1 | Drop |", "|---|---|---|---|",
              f"| FinMA-7B, from our re-run | {wf['FinMA-7B']:.4f} | {w['FinMA-7B']:.4f} | {w['FinMA-7B'] - wf['FinMA-7B']:+.4f} |",
              f"| FinMA-7B, from the published score | {pub:.2f} | {w['FinMA-7B']:.4f} | **{w['FinMA-7B'] - pub:+.4f}** |",
              f"| Ours, 3-seed ensemble | {wf['Ours, 3-seed ensemble']:.4f} | {w['Ours, 3-seed ensemble']:.4f} | "
              f"**{w['Ours, 3-seed ensemble'] - wf['Ours, 3-seed ensemble']:+.4f}** |",
              f"| E4B-it zero-shot | {wf['E4B-it zero-shot']:.4f} | {w['E4B-it zero-shot']:.4f} | "
              f"{w['E4B-it zero-shot'] - wf['E4B-it zero-shot']:+.4f} |", "",
              f"With the published number as FinMA's FinBen score, FinMA's FinBen lead is {lead_pub:+.4f} and the shrink is "
              f"**{shrink_pub:+.4f}**, 95% CI [{lo_p:+.3f}, {hi_p:+.3f}] (published score held fixed, everything else "
              f"resampled as in c4_report.py): "
              + (f"meets the {MIN_SHRINK} rule with a CI excluding 0, so the verdict holds on this reference too. "
                 if shrink_pub >= MIN_SHRINK and lo_p > 0 else
                 f"by c4_report.py's rule that is **inconclusive** on this reference. The Part C verdict therefore "
                 f"depends on our re-run's {wf['FinMA-7B']:.3f} as FinMA's FinBen score. ")
              + f"Our re-run's {wf['FinMA-7B']:.3f} is itself unexplained "
              f"({100 * (wf['FinMA-7B'] - pub):.1f} points above the published score; see ANALYSIS.md, Part B).", ""]

    # ---- e) confusion matrices on the fresh set
    lines += ["## e) Confusion matrices on the fresh set", "", "Rows: gold label (AI-made). Columns: the model's parsed answer.", ""]
    for k in ["FinMA-7B", "Ours, 3-seed ensemble", "E4B-it zero-shot"]:
        cm = confusion_matrix(g, fresh[k], labels=CLASSES)[:len(LABELS)]
        lines += [f"**{k}** (wF1 {wf1(g, fresh[k]):.4f})", "",
                  "| gold \\ predicted | " + " | ".join(CLASSES) + " | total |", "|---" * (len(CLASSES) + 2) + "|",
                  *[f"| {lab} | " + " | ".join(str(v) for v in row) + f" | {row.sum()} |" for lab, row in zip(LABELS, cm)],
                  f"| total | " + " | ".join(str(v) for v in cm.sum(0)) + f" | {cm.sum()} |", ""]
    cell = lambda k, gl, pl: int(np.sum((g == gl) & (fresh[k] == pl)))
    lines += [f"*Neutral sentences called positive: FinMA {cell('FinMA-7B', 'neutral', 'positive')}, ours "
              f"{cell('Ours, 3-seed ensemble', 'neutral', 'positive')}, zero-shot {cell('E4B-it zero-shot', 'neutral', 'positive')}. "
              f"Positive sentences called neutral: FinMA {cell('FinMA-7B', 'positive', 'neutral')}, ours "
              f"{cell('Ours, 3-seed ensemble', 'positive', 'neutral')}, zero-shot {cell('E4B-it zero-shot', 'positive', 'neutral')}. "
              "The neutral/positive border is where the labels matter most: whether a mildly upbeat corporate statement "
              "counts as neutral (PhraseBank's convention) or positive decides much of the FinMA gap (section c).*", ""]

    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
