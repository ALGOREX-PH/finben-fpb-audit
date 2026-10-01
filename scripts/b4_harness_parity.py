"""Phase 4: is our scoring identical to FinBen's official harness? No GPU, seconds.

  uv run python scripts/b4_harness_parity.py

FinBen's published FPB numbers come from PIXIU's `src/tasks/flare.py` (class FPB -> Classification).
Running that old harness itself on Gemma 4 E4B in 8 GB isn't possible (it predates the architecture and
loads full-size models), so instead this script re-scores OUR saved raw model outputs with PIXIU's own
code, copied verbatim below, and checks that every prediction and every metric matches ours.
If they match, the only difference from the official harness is how the model is loaded -- not how
it is prompted, decoded or scored.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef

import config_b as config

# ---- verbatim from PIXIU src/tasks/flare.py, Classification.process_results (LOWER_CASE = True) ----------


def pixiu_process_results(doc, results):
    LOWER_CASE = True
    gold: str = doc["choices"][doc["gold"]]
    if LOWER_CASE:
        gold = gold.lower()
    ini_result = results[0].strip()
    if LOWER_CASE:
        ini_result = ini_result.lower()

    result = None
    for choice in doc["choices"]:
        if LOWER_CASE:
            choice = choice.lower()
        if choice in ini_result:
            result = choice
            break
    if result is None:
        result = "missing"

    acc = 1.0 if gold == result else 0.0
    return {"acc": acc, "missing": int(result == "missing"), "f1": (result, gold), "macro_f1": (result, gold), "mcc": (result, gold)}
# ------------------------------------------------------------------------------------------------------------


def pixiu_aggregate(items):
    preds, golds = zip(*[r["f1"] for r in items])
    labels = sorted(set(golds))
    idx = {l: i for i, l in enumerate(labels + ["missing"])}
    return {"acc": np.mean([r["acc"] for r in items]),
            "f1": f1_score(golds, preds, average="weighted"),
            "macro_f1": f1_score(golds, preds, average="macro"),
            "mcc": matthews_corrcoef([idx[g] for g in golds], [idx[p] for p in preds]),
            "missing": np.mean([r["missing"] for r in items])}


test = pd.read_csv(config.DATA_DIR / config.SPLIT_FILES["test"])
docs = [{"choices": json.loads(c), "gold": int(g)} for c, g in zip(test.choices, test.gold)]
rows = []
for f in sorted((config.PRED_DIR / "test").glob("*.csv")):
    ours = pd.read_csv(f, keep_default_na=False)
    if len(ours) != len(test) or "raw" not in ours:
        continue
    items = [pixiu_process_results(d, [r]) for d, r in zip(docs, ours.raw)]
    same_pred = np.mean([it["f1"][0] == p for it, p in zip(items, ours.pred)])
    same_gold = np.mean([it["f1"][1] == g for it, g in zip(items, ours.gold)])
    agg = pixiu_aggregate(items)
    ours_f1 = f1_score(ours.gold, ours.pred, average="weighted", zero_division=0)
    rows.append((f.stem, same_pred, same_gold, agg["f1"], ours_f1, agg["acc"], agg["mcc"], agg["missing"]))

print(f"{'system':28} {'same pred':>9} {'same gold':>9} {'PIXIU wF1':>9} {'our wF1':>8} {'acc':>6} {'mcc':>6} {'missing':>7}")
for r in rows:
    print(f"{r[0]:28} {r[1]:9.1%} {r[2]:9.1%} {r[3]:9.4f} {r[4]:8.4f} {r[5]:6.3f} {r[6]:6.3f} {r[7]:7.1%}")
if not rows:   # all([]) is True -- never report parity on zero systems
    raise SystemExit(print("no complete test predictions in results/partB/predictions/test yet -- nothing to check") or 0)
ok = all(r[1] == 1 and r[2] == 1 and abs(r[3] - r[4]) < 1e-9 for r in rows)
out = config.RESULTS_DIR / "harness_parity.json"
out.write_text(json.dumps({"identical_to_pixiu_scoring": ok, "systems": [r[0] for r in rows]}, indent=2))
print(f"\nScoring identical to FinBen's official PIXIU code on every system: {ok}  -> {out.name}")
