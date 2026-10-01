"""Part C, step 2: label the fresh sentences -- BLIND, by PhraseBank's rule. No GPU.

  uv run python scripts/c2_label.py                # label (resumes where you stopped)
  uv run python scripts/c2_label.py --compare      # agreement with the second annotator
  uv run python scripts/c2_label.py --adjudicate   # settle each disagreement yourself
  uv run python scripts/c2_label.py --spotcheck 40 # human check of the shipped (AI-made) gold labels

THE RULE (Financial PhraseBank's annotation guideline, Malo et al. 2014):
  Read the sentence as an INVESTOR. Would this news, on its own, likely make the company's share
  price go UP (positive), DOWN (negative), or have no clear effect (neutral)?
  - Facts with no clear good/bad direction (dates, appointments, routine procedures) -> neutral
  - Judge the sentence alone; don't look anything up.

Blind: you see only the sentence -- no model output, no AI suggestion -- so your labels can't be
anchored by what the models say. Saved after every answer to data/fresh/labels.csv.
The second annotator (data/fresh/labels_claude.csv) is an independent AI pass, compared only AFTER you
finish; disagreements are then decided by you (--adjudicate). Your final label is the gold label.

AS SHIPPED, the gold labels were NOT made this way: data/fresh/labels.csv was filled by an AI (Claude) at
the user's request, following the rule above (its labeled_by column says so). No human has labelled the
set yet; --spotcheck N measures how often a blind human agrees with that AI answer key.
"""
import argparse

import config  # noqa: F401
import pandas as pd

from config import DATA_DIR

FRESH = DATA_DIR / "fresh"
CAND, LAB, CLAUDE = FRESH / "candidates.csv", FRESH / "labels.csv", FRESH / "labels_claude.csv"
KEYS = {"1": "negative", "2": "neutral", "3": "positive"}


def load():
    cand = pd.read_csv(CAND)
    if LAB.exists():
        lab = pd.read_csv(LAB, keep_default_na=False, dtype=str)
    else:
        lab = pd.DataFrame({"id": cand.id, "human": "", "final": "", "note": ""})
    return cand, lab


def save(lab):
    lab.to_csv(LAB, index=False, encoding="utf-8")


def label(cand, lab):
    todo = [i for i in lab.index if not lab.at[i, "human"]]
    print(__doc__.split("THE RULE")[1].split("Blind:")[0].rstrip())
    print(f"\n{len(lab) - len(todo)} of {len(lab)} done. Keys: 1 = negative   2 = neutral   3 = positive   "
          "x = exclude (not a usable sentence)   s = skip   b = back   q = save and quit")
    k = 0
    while k < len(todo):
        i = todo[k]
        print(f"\n[{len(lab) - len(todo) + k + 1}/{len(lab)}]  {cand.text[i]}")
        a = input("  1/2/3/x/s/b/q > ").strip().lower()
        if a == "q":
            break
        if a == "b":
            k = max(0, k - 1)
            continue
        if a == "s":
            k += 1
            continue
        if a in KEYS or a == "x":
            lab.at[i, "human"] = KEYS.get(a, "excluded")
            save(lab)
            k += 1
        else:
            print("  ? type 1, 2, 3, x, s, b or q")
    done = lab.human.ne("").sum()
    print(f"\nSaved. {done} of {len(lab)} labeled. {lab.human.value_counts().to_dict()}")


def compare(cand, lab):
    if not CLAUDE.exists():
        raise SystemExit(f"{CLAUDE.name} not found -- the second annotator pass hasn't been added yet.")
    cl = pd.read_csv(CLAUDE, keep_default_na=False, dtype=str).set_index("id")
    both = lab[lab.human.isin(KEYS.values())].copy()
    both["claude"] = cl.loc[both.id, "label"].values
    both = both[both.claude.isin(KEYS.values())]
    agree = (both.human == both.claude).mean()
    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(both.human, both.claude)
    print(f"{len(both)} sentences labeled by both: raw agreement {agree:.1%}, Cohen's kappa {kappa:.3f}")
    print("(PhraseBank's own annotators: all 16 agreed on only ~47% of sentences; kappa around 0.6-0.8 is typical for this task)")
    print(pd.crosstab(both.human, both.claude, rownames=["you"], colnames=["second annotator"]))
    return both


def adjudicate(cand, lab):
    both = compare(cand, lab)
    cl = pd.read_csv(CLAUDE, keep_default_na=False, dtype=str).set_index("id")
    dis = both[(both.human != both.claude) & (lab.loc[both.index, "final"] == "")]
    print(f"\n{len(dis)} disagreements to settle. Keys: 1/2/3 = final label, Enter = keep yours, q = quit")
    for i in dis.index:
        print(f"\n  {cand.text[i]}\n  you: {lab.at[i, 'human']}   second annotator: {both.at[i, 'claude']}  "
              f"({cl.at[lab.at[i, 'id'], 'reason'] if 'reason' in cl else ''})")
        a = input("  final > ").strip().lower()
        if a == "q":
            break
        lab.at[i, "final"] = KEYS.get(a, lab.at[i, "human"])
        save(lab)
    print("Saved.")


def spotcheck(cand, lab, n):
    """Label a random sample BLIND, then measure agreement with the gold labels (e.g. when Claude made them)."""
    from sklearn.metrics import cohen_kappa_score
    f = FRESH / "spotcheck.csv"
    usable = lab[lab.human.isin(KEYS.values())]
    ids = usable.sample(min(n, len(usable)), random_state=7).id.tolist()
    sc = pd.read_csv(f, keep_default_na=False, dtype=str) if f.exists() else pd.DataFrame({"id": ids, "label": ""})
    todo = [i for i in sc.index if not sc.at[i, "label"]]
    text = cand.set_index("id").text
    print(__doc__.split("THE RULE")[1].split("Blind:")[0].rstrip())
    print(f"\nSpot-check: {len(sc) - len(todo)} of {len(sc)} done. Keys: 1 = negative  2 = neutral  3 = positive  q = save and quit")
    for i in todo:
        print(f"\n[{i + 1}/{len(sc)}]  {text[sc.at[i, 'id']]}")
        a = input("  1/2/3/q > ").strip().lower()
        while a not in KEYS and a != "q":
            a = input("  ? 1, 2, 3 or q > ").strip().lower()
        if a == "q":
            break
        sc.at[i, "label"] = KEYS[a]
        sc.to_csv(f, index=False, encoding="utf-8")
    done = sc[sc.label != ""]
    if len(done):
        gold = lab.set_index("id").loc[done.id, "human"].values
        print(f"\nYour blind labels vs the gold labels on {len(done)} sentences: agreement {(done.label.values == gold).mean():.1%}, "
              f"Cohen's kappa {cohen_kappa_score(done.label.values, gold):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--adjudicate", action="store_true")
    ap.add_argument("--spotcheck", type=int, metavar="N", help="blind-label N random sentences to check the gold labels")
    args = ap.parse_args()
    cand, lab = load()
    if args.spotcheck:
        spotcheck(cand, lab, args.spotcheck)
    elif args.compare:
        compare(cand, lab)
    elif args.adjudicate:
        adjudicate(cand, lab)
    else:
        label(cand, lab)
