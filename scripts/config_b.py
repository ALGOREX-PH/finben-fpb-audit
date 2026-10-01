"""Part B settings: improve the fine-tune to beat FinMA-7B -- choosing ONLY on validation.

The rules (they are what make a "we beat X" claim mean something):
  1. Every choice (epochs, learning rate, rank, decoding, ensembling) is made on the VALIDATION split.
  2. The TEST split is scored ONCE, with the final configuration, after the choice is frozen
     (results/partB/selected_config.json). No re-tuning after seeing test numbers.
  3. Claims use 3 seeds (mean +- std) and McNemar tests on the same test sentences.
  4. FinMA is re-run by us with the same parser and metric and its own official prompt wrapper.
"""
from config import *  # noqa: F401,F403

RESULTS_DIR = ROOT / "results" / "partB"
PRED_DIR = RESULTS_DIR / "predictions"            # predictions/<split>/<system>.csv
SELECTED = RESULTS_DIR / "selected_config.json"   # written by b3_select.freeze(); freezes the final choice

BATCH_SIZE = 4               # 4 x 4 for EVERY run: rank 32 doesn't fit 8 GB at 8, and identical runs keep comparisons fair
GRAD_ACCUM = 4               # effective batch 16

# Baseline = Part A's setting. The search changes one thing at a time from here.
BASE = {"epochs": 1, "lr": 2e-4, "rank": 16}
SEARCH = {
    "epochs": [1, 2, 3],
    "lr": [1e-4, 2e-4, 4e-4],
    "rank": [16, 32],
}
SEARCH_SEED = 3407           # one seed per candidate during the search...
SEEDS = [3407, 42, 7]        # ...then 3 seeds for the finalists and the final model
FINALISTS = 2

# The model to beat (PIXIU / FinBen). Published FinBen FPB weighted F1 = 0.88 (FinBen paper, Table 3).
FINMA_MODEL = "ChanceFocus/finma-7b-nlp"          # same weights as TheFinAI/finma-7b-nlp, not gated, MIT
FINMA_PUBLISHED_F1 = 0.88
MEMO_SAMPLE = 970            # train sentences sampled for the memorisation check
