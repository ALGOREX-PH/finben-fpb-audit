"""Shared settings: FinBen FPB with base Gemma 4 E4B-it.

  Part A (a*.py): how much of FinBen's test set leaks from its own train split, and does it inflate scores?
  Part B (b*.py): improve the fine-tune to beat FinMA-7B -- choosing only on validation, testing once.
  Part C (c*.py): a fresh 2026 test set no model can have seen -- does FinMA's lead survive?
  Part D (d*.py): contamination x training-intensity controls for Part C's memorisation reading.
Part-specific settings live in config_a.py / config_b.py (both start from this file).
"""
from pathlib import Path

from ftlib.paths import MODELS_DIR, ROOT  # noqa: F401  (also points HF/Kaggle caches at .cache/)

TASK_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"                     # FinBen FPB splits + contamination.csv (git-ignored, gated data)
ADAPTERS_DIR = MODELS_DIR / "adapters"             # adapters of both parts

# The benchmark (FinBen / PIXIU "FLARE" FPB task)
FINBEN_DATASET = "TheFinAI/flare-fpb"            # gated: accept terms on its Hugging Face page + `hf auth login`
FINBEN_URL = "https://huggingface.co/datasets/TheFinAI/flare-fpb"
SPLIT_FILES = {"train": "finben_train.csv", "validation": "finben_validation.csv", "test": "finben_test.csv"}
VALID_FILE = SPLIT_FILES["validation"]           # FinBen names its dev split "validation"
LABELS = ["negative", "neutral", "positive"]     # fixed order for probability columns

# Model: base Gemma 4 E4B-it with fresh LoRA adapters
MODEL_NAME = "unsloth/gemma-4-E4B-it"
CHAT_TEMPLATE = "gemma-4"
MAX_SEQ_LEN = 384            # FinBen's query wraps each sentence in an instruction

# Contamination detectors (ftlib.contamination)
NEAR_DUP_THRESHOLD = 0.90
NGRAM_N = 8                  # loose sensitivity check only (over-fires on newswire boilerplate)
