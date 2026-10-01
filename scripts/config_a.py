"""Part A settings (contamination study). Everything shared comes from config.py."""
from config import *  # noqa: F401,F403

RESULTS_DIR = ROOT / "results" / "partA"
PRED_DIR = RESULTS_DIR / "predictions"

# The Part A fine-tune (as run on 2026-09-30): 1 epoch, lr 2e-4, rank 16, micro-batch 8 x 2
LORA_R = 16
LORA_ALPHA = 16
LEARNING_RATE = 2e-4
EPOCHS = 1
BATCH_SIZE = 8
GRAD_ACCUM = 2
SEEDS = [3407, 42, 7]
