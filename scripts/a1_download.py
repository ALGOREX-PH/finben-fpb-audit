"""Step 1: download the official FinBen FPB test set and find out which PhraseBank version it is.

  uv run python scripts/a1_download.py

The dataset is GATED. One-time setup:
  1. Open https://huggingface.co/datasets/TheFinAI/flare-fpb (log in) and accept the access conditions.
  2. In a terminal:  uv run hf auth login   (paste a Read token from huggingface.co/settings/tokens)

Besides saving the data, this checks each test sentence against the four PhraseBank agreement levels
(50% / 66% / 75% / 100% of annotators agreed). That tells you which version FinBen used -- papers often
don't say, and scores on AllAgree are far higher than on 50Agree.
"""
import glob
import json

import config_a as config  # noqa: F401  (first: cache paths)
import kagglehub
import pandas as pd
from datasets import load_dataset

from config_a import DATA_DIR, FINBEN_DATASET, FINBEN_URL
from ftlib.contamination import normalize

try:
    ds = load_dataset(FINBEN_DATASET)
except Exception as e:   # gated / not logged in / terms not accepted
    raise SystemExit(
        f"Could not download {FINBEN_DATASET}: {type(e).__name__}: {str(e)[:200]}\n\n"
        f"Fix (one time):\n  1. Open {FINBEN_URL} , log in, and accept the access conditions.\n"
        f"  2. Run:  uv run hf auth login   (paste a Read token from https://huggingface.co/settings/tokens)\n"
        f"  3. Check: uv run hf auth whoami\nThen run this script again.")

DATA_DIR.mkdir(parents=True, exist_ok=True)
for split, part in ds.items():
    df = part.to_pandas()
    df["choices"] = df["choices"].map(lambda c: json.dumps(list(c)))
    df.to_csv(DATA_DIR / f"finben_{split}.csv", index=False)
    print(f"{split}: {len(df)} rows, answers {df.answer.value_counts().to_dict()}")

test = pd.read_csv(DATA_DIR / "finben_test.csv")
print("\nexample query (the exact prompt FinBen sends):\n---\n" + test["query"].iloc[0] + "\n---")
print(f"choices: {test['choices'].iloc[0]}   (first match in THIS order wins when parsing)")

# Which PhraseBank version? Tag each sentence with the strictest agreement level containing it.
glob_pb = lambda: glob.glob(str(config.ROOT / ".cache" / "kagglehub" / "datasets" / "ankurzing" / "*" / "versions" / "*" / "FinancialPhraseBank"))
if not glob_pb():
    kagglehub.dataset_download("ankurzing/sentiment-analysis-for-financial-news")
pb_dir = glob_pb()[0]
levels = {}
for level in ["AllAgree", "75Agree", "66Agree", "50Agree"]:
    with open(f"{pb_dir}/Sentences_{level}.txt", encoding="latin-1") as fh:
        levels[level] = {normalize(line.rsplit("@", 1)[0]): line.rsplit("@", 1)[1].strip() for line in fh if "@" in line}
print("\nstrictest PhraseBank agreement level per sentence (tells you which PhraseBank version FinBen used):")
for split in ds:
    df = pd.read_csv(DATA_DIR / f"finben_{split}.csv")
    key = df.text.map(normalize)
    df["agreement"] = [next((lv for lv in levels if k in levels[lv]), "not_in_phrasebank") for k in key]
    df["pb_label"] = [levels["50Agree"].get(k, "") for k in key]
    df.to_csv(DATA_DIR / f"finben_{split}.csv", index=False)
    found = df.pb_label.ne("")
    print(f"  {split:6} {df.agreement.value_counts().to_dict()}  "
          f"| FinBen answer == PhraseBank label: {(df.answer[found].str.lower() == df.pb_label[found]).mean():.1%}")
