"""Choosing the configuration -- from VALIDATION predictions only. Run it any time to see the table:

  uv run python scripts/b3_select.py

Rules:
  * score = weighted F1 on validation (FinBen's metric), mean over the seeds available for a config
  * TIE_TOLERANCE: a config within 0.003 of the best (~2 of 776 sentences) counts as tied, and the
    CHEAPER one wins (fewer epochs, then lower rank, then learning rate closest to the baseline).
    Picking the top number among ties would just be fitting noise in the validation set.
  * decoding (generated reply vs label-probability argmax) and seed-ensembling are chosen the same way.
freeze() writes results/selected_config.json -- after that, the test split may be scored once.
"""
import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import config_b as config
import preds
from config_b import LABELS

TIE_TOLERANCE = 0.003
RUN = re.compile(r"^e(?P<epochs>\d+)_lr(?P<lr>[\d.e-]+)_r(?P<rank>\d+)_s(?P<seed>\d+)$")


def wf1(gold, pred):
    return f1_score(gold, pred, average="weighted", zero_division=0)


def validation_runs():
    """{(epochs, lr, rank): {seed: DataFrame}} for every train-only adapter scored on validation."""
    runs = defaultdict(dict)
    for f in (config.PRED_DIR / "validation").glob("e*_lr*_r*_s*.csv"):
        m = RUN.match(f.stem)
        if m:
            runs[(int(m["epochs"]), float(m["lr"]), int(m["rank"]))][int(m["seed"])] = preds.read(f)
    return runs


def score_table():
    rows = []
    for cfg, seeds in validation_runs().items():
        dfs = list(seeds.values())
        gold = dfs[0].gold
        gen = [wf1(d.gold, d.pred) for d in dfs]
        arg = [wf1(d.gold, d.pred_argmax) for d in dfs]
        probs = np.mean([d[[f"p_{l}" for l in LABELS]].values for d in dfs], axis=0)
        ens = wf1(gold, np.array(LABELS)[probs.argmax(1)]) if len(dfs) > 1 else np.nan
        rows.append({"epochs": cfg[0], "lr": cfg[1], "rank": cfg[2], "seeds": len(dfs),
                     "wf1_generated": np.mean(gen), "std": np.std(gen, ddof=1) if len(gen) > 1 else np.nan,
                     "wf1_argmax": np.mean(arg), "wf1_ensemble": ens})
    return pd.DataFrame(rows)


def cost(row):
    return (row["epochs"], row["rank"], abs(np.log(row["lr"] / config.BASE["lr"])))


def pick(table, column="wf1_generated"):
    """Best row, with ties (within TIE_TOLERANCE) broken towards the cheaper configuration."""
    best = table[column].max()
    tied = table[table[column] >= best - TIE_TOLERANCE]
    return tied.loc[min(tied.index, key=lambda i: cost(tied.loc[i]))]


def freeze():
    table = score_table()
    full = table[table.seeds >= len(config.SEEDS)]
    if full.empty:
        raise SystemExit("no configuration has all seeds on validation yet -- run the finalist stage first")
    row = pick(full)
    single = max(row.wf1_generated, row.wf1_argmax)
    decoding = "argmax" if row.wf1_argmax > row.wf1_generated + TIE_TOLERANCE else "generated"
    ensemble = bool(row.wf1_ensemble > single + TIE_TOLERANCE)
    chosen = {"epochs": int(row.epochs), "lr": float(row.lr), "rank": int(row["rank"]), "decoding": decoding,
              "ensemble": ensemble, "validation": {k: (None if pd.isna(row[k]) else round(float(row[k]), 4))
                                                   for k in ["wf1_generated", "std", "wf1_argmax", "wf1_ensemble"]},
              "rule": f"max validation weighted F1 over {len(config.SEEDS)} seeds; ties within {TIE_TOLERANCE} -> cheaper config"}
    config.SELECTED.parent.mkdir(parents=True, exist_ok=True)
    config.SELECTED.write_text(json.dumps(chosen, indent=2))
    return chosen


if __name__ == "__main__":
    t = score_table()
    if t.empty:
        print("no validation predictions yet")
    else:
        print(t.sort_values("wf1_generated", ascending=False).round(4).to_string(index=False))
        if config.SELECTED.exists():
            print("\nFROZEN:", config.SELECTED.read_text())
