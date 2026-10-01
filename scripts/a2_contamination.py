"""Step 2: does FinBen's OWN training split leak into its test split? No GPU, seconds.

  uv run python scripts/a2_contamination.py

Benchmarks are usually split once, at random, from one source. Financial PhraseBank has many
near-identical sentences (same company template, different numbers), so a random split can put a
sentence in train and its near-copy in test. A model fine-tuned on train then gets partial credit
for memory, not skill. This script measures that, using ftlib.contamination:
  contaminated = the test sentence (or a near-copy, cosine >= 0.90) appears in TRAIN (validation overlap
                 is reported separately: it's watched during training, never trained on)
  clean        = it doesn't -> the honest part of the test set
  loose        = also counts 8-gram overlap: over-fires on newswire boilerplate, so its extra hits are
                 printed for you to judge by eye
It also writes a DECONTAMINATED training set (train minus every sentence that near-copies a test
sentence), so 03_train_qlora.py --decontaminate can train a model that never saw them.
Outputs: data/contamination.csv (per test sentence), data/finben_train_decontam.csv
"""
import config_a as config  # noqa: F401
import pandas as pd

from config_a import DATA_DIR, NEAR_DUP_THRESHOLD, NGRAM_N
from ftlib.contamination import contamination_table

need = [DATA_DIR / f"finben_{s}.csv" for s in ("train", "test")]
if not all(p.exists() for p in need):
    raise SystemExit("FinBen splits missing -- run 01_download.py first")
train, test = pd.read_csv(need[0]), pd.read_csv(need[1])
valid_file = DATA_DIR / config.VALID_FILE
valid = pd.read_csv(valid_file) if valid_file.exists() else train.head(0)
# Contamination = what the adapter's WEIGHTS trained on = the train split. The validation split is only
# watched during training (never trained on), so its overlap is reported separately.
seen = train.assign(split="train")

# 1. test sentences that appear (near-)verbatim in train
tab = contamination_table(test.text, seen.text, NEAR_DUP_THRESHOLD, NGRAM_N)
tab.insert(0, "id", test.id)
tab["gold"] = test.answer.str.lower().values
tab["nearest_seen_text"] = seen.text.values[tab.nearest_train]
tab["nearest_seen_split"] = seen.split.values[tab.nearest_train]
tab["nearest_seen_label"] = seen.answer.str.lower().values[tab.nearest_train]
tab["same_label_as_seen"] = tab["contaminated"] & (tab.nearest_seen_label == tab.gold)
tab["in_validation"] = contamination_table(test.text, valid.text, NEAR_DUP_THRESHOLD, NGRAM_N)["contaminated"].values if len(valid) else False
if "agreement" in test:
    tab["agreement"] = test.agreement.values
tab.drop(columns="nearest_train").to_csv(DATA_DIR / "contamination.csv", index=False)

# 2. the other direction: training sentences that near-copy a test sentence -> drop them
rev = contamination_table(train.text, test.text, NEAR_DUP_THRESHOLD, NGRAM_N)
clean_train = train[~rev["contaminated"].values]
clean_train.to_csv(DATA_DIR / "finben_train_decontam.csv", index=False)

n = len(tab)
print(f"FinBen FPB: test {n} vs train {len(train)}  (validation {len(valid)}: overlap reported separately)")
for det in ["exact", "near_dup", "contaminated", "ngram", "loose"]:
    print(f"  {det:12} {int(tab[det].sum()):5}  ({tab[det].mean():.1%} of test)")
c = tab[tab["contaminated"]]
print(f"clean test sentences: {int((~tab['contaminated']).sum())} ({(~tab['contaminated']).mean():.1%})")
if len(c):
    print(f"contaminated test sentences whose train/valid copy has the SAME label: {c.same_label_as_seen.mean():.1%} "
          f"({int((~c.same_label_as_seen).sum())} disagree: memorising those would COST points)")
    print(f"nearest copy found in: {c.nearest_seen_split.value_counts().to_dict()}")
    for i in c.index[:3]:
        print(f"  test : {test.text[i][:120]}\n  seen : {tab.nearest_seen_text[i][:120]}  (cosine {tab.max_cosine[i]:.2f})")
print(f"test sentences with a near-copy in VALIDATION (never trained on): {int(tab.in_validation.sum())}")
print(f"decontaminated train: {len(clean_train)} of {len(train)} kept ({len(train) - len(clean_train)} removed)")

only_ngram = tab.ngram & ~tab.contaminated
if only_ngram.any():
    print(f"\n{int(only_ngram.sum())} test sentences flagged ONLY by the {NGRAM_N}-gram detector -- judge them yourself "
          "(same sentence reworded = contamination; shared boilerplate = false alarm):")
    for i in tab.index[only_ngram][:5]:
        print(f"  test : {test.text[i][:120]}\n  seen : {tab.nearest_seen_text[i][:120]}  (cosine {tab.max_cosine[i]:.2f})")
if "agreement" in tab:
    print("\ncontamination by PhraseBank agreement level:")
    print(tab.groupby("agreement")["contaminated"].agg(["size", "mean"]).rename(columns={"size": "n", "mean": "contaminated"}).round(3).to_string())
