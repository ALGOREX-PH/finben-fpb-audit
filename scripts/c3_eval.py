"""Part C, step 3: run the models on the fresh sentences, with FinBen's exact prompt and parser. GPU.

  uv run python scripts/c3_eval.py --system zeroshot
  uv run python scripts/c3_eval.py --system ours --seed 3407     # the frozen Part B final model
  uv run python scripts/c3_eval.py --system finma                # official 'Human:/Assistant:' wrapper
  uv run python scripts/c3_eval.py --system control --adapter e15_lr0.0004_r16_s3407_tvt \
      --adapter-dir models/controls --pred-dir results/partD/predictions/fresh   # Part D control
  add --limit 8 for a smoke test

Predictions don't depend on the labels, so this can run before or after labelling.
Each sentence goes into FinBen's single FPB template, verbatim:
  "Analyze the sentiment of this statement extracted from a financial news article. Provide your answer
   as either negative, positive, or neutral.\\nText: {text}\\nAnswer:"
and replies are parsed with FinBen's rule (first choice found, choices order positive/neutral/negative).
Output: results/partC/predictions/<system>.csv
"""
import argparse
import importlib.util
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import config_b as config  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config_b import LABELS  # noqa: E402

PRED = config.ROOT / "results" / "partC" / "predictions"
CHOICES = ["positive", "neutral", "negative"]          # FinBen FPB's choice order (first match wins)

_spec = importlib.util.spec_from_file_location("b2_eval", config.TASK_DIR / "b2_eval.py")   # Part B's tested code
b2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b2)


def finben_template():
    test = pd.read_csv(config.DATA_DIR / config.SPLIT_FILES["test"])
    templates = {q.replace(t, "{text}") for q, t in zip(test["query"], test.text)}
    assert len(templates) == 1, f"expected ONE FinBen template, found {len(templates)}"
    return templates.pop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["zeroshot", "ours", "finma", "control"])
    ap.add_argument("--adapter", help="--system control: run name under --adapter-dir, e.g. e15_lr0.0004_r16_s3407_tvt")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--adapter-dir", type=Path, default=config.ADAPTERS_DIR,
                    help="folder holding the adapter (Part D controls: models/controls)")
    ap.add_argument("--pred-dir", type=Path, default=PRED,
                    help="output folder, <pred-dir>/<system>.csv (Part D: results/partD/predictions/fresh)")
    args = ap.parse_args()

    sel = json.loads(config.SELECTED.read_text())
    adapter = f"e{sel['epochs']}_lr{sel['lr']:g}_r{sel['rank']}_s{args.seed}_tv"
    if args.system == "control" and not args.adapter:
        ap.error("--system control needs --adapter")
    name = {"zeroshot": "e4b_zeroshot", "finma": "finma_7b", "ours": adapter, "control": args.adapter}[args.system]
    cand = pd.read_csv(config.DATA_DIR / "fresh" / "candidates.csv")
    out = args.pred_dir / f"{name}.csv"
    if args.skip_existing and out.exists() and len(pd.read_csv(out)) == len(cand):
        print(f"skip: {out.name} exists")
        raise SystemExit(0)
    frozen = out.resolve().is_relative_to(PRED.resolve())
    if frozen and args.adapter_dir.resolve() != config.ADAPTERS_DIR.resolve():
        raise SystemExit("control adapters must not write into results/partC -- pass --pred-dir results/partD/predictions/fresh")
    if frozen and out.exists() and not args.limit:
        raise SystemExit(f"{out} is a frozen Part C result -- not overwriting it")
    if args.limit:
        cand = cand.head(args.limit)

    tmpl = finben_template()
    queries = [tmpl.replace("{text}", t) for t in cand.text]
    t0 = time.time()
    if args.system == "finma":
        raws, probs = b2.run_finma(queries, wrap=True)
    else:
        raws, probs = b2.run_gemma(queries, {"ours": adapter, "control": args.adapter}.get(args.system),
                                   adapter_dir=args.adapter_dir)
    res = pd.DataFrame({"id": cand.id, "text": cand.text, "raw": raws,
                        "pred": [b2.finben_parse(r, CHOICES) for r in raws],
                        "pred_argmax": np.array(LABELS)[probs.argmax(1)]})
    for j, label in enumerate(LABELS):
        res[f"p_{label}"] = probs[:, j]
    if not args.limit:
        args.pred_dir.mkdir(parents=True, exist_ok=True)
        res.to_csv(out, index=False)
    print(f"{name} on {len(res)} fresh sentences ({time.time() - t0:.0f}s): predicted {res.pred.value_counts().to_dict()}")
