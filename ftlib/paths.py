"""Workspace paths + environment setup. Import this FIRST in every script (before torch,
transformers, datasets or kagglehub) -- most of these settings are read once, at import time."""
import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# Big downloads (models, datasets) go into the repo's .cache/ folder.
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("KAGGLEHUB_CACHE", str(ROOT / ".cache" / "kagglehub"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ...but the Hugging Face login token stays in its standard place (~/.cache/huggingface/token),
# so a plain `hf auth login` in any terminal is picked up by every script here.
os.environ.setdefault("HF_TOKEN_PATH", str(Path.home() / ".cache" / "huggingface" / "token"))

# PyTorch prints "NOTE: Redirects are currently not supported in Windows or MacOs." on every
# import on Windows. It's about a multi-machine training feature we don't use.
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
