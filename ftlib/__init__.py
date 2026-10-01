"""ftlib: small shared helpers used by scripts/.

  ftlib.paths          repo root; keeps Hugging Face / Kaggle caches inside the repo (.cache/)
  ftlib.model_loader   load Gemma 4 E-series in 4-bit on an 8 GB GPU (per-layer embedding table kept in CPU RAM)
  ftlib.metrics        classification metrics, bootstrap CIs, McNemar's test, calibration (ECE)
  ftlib.contamination  exact / near-duplicate / n-gram train-test overlap detectors
  ftlib.finben         FinBen/PIXIU's FPB answer parser; re-attaching gold labels to shipped predictions (no torch)
  ftlib.llm_classify   batched LLM classification: generated reply + label probabilities
"""
