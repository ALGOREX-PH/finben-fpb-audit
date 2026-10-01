"""Step 4: score every system on the FinBen FPB test set with FinBen's settings. GPU, ~1-2 min each.

  uv run python scripts/a4_eval.py --system zeroshot
  uv run python scripts/a4_eval.py --system qlora --seed 3407
  uv run python scripts/a4_eval.py --system qlora --seed 3407 --decontaminate
  uv run python scripts/a4_eval.py --system finbert
  add --limit 40 for a smoke test

FinBen's evaluation settings (PIXIU src/tasks/flare.py):
  prompt  the dataset's `query` field, verbatim (sent as the user turn of Gemma's chat template)
  parse   lower-case the reply, take the FIRST choice (in the dataset's `choices` order) that appears
          anywhere in it, else "missing"
  metrics accuracy, weighted F1 (their headline "F1"), macro F1, MCC, missing rate   -> 05_report.py
Each prediction file also keeps a strict parse (reply is exactly one label) and the first-token
label probabilities, for calibration.
"""
import argparse
import json
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import config_a as config  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config_a import ADAPTERS_DIR, DATA_DIR, LABELS, PRED_DIR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--system", required=True, choices=["zeroshot", "qlora", "finbert"])
ap.add_argument("--seed", type=int, default=3407)
ap.add_argument("--decontaminate", action="store_true", help="use the adapter trained on decontaminated data")
ap.add_argument("--limit", type=int)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--skip-existing", action="store_true")
args = ap.parse_args()

run = f"fpb-seed{args.seed}" + ("-decontam" if args.decontaminate else "")
name = {"zeroshot": "e4b_zeroshot", "finbert": "finbert",
        "qlora": f"qlora_seed{args.seed}" + ("-decontam" if args.decontaminate else "")}[args.system]
test = pd.read_csv(DATA_DIR / "finben_test.csv")
out = PRED_DIR / f"{name}.csv"
if args.skip_existing and out.exists() and len(pd.read_csv(out)) == len(test):
    print(f"skip: {out.name} already has all {len(test)} predictions")
    raise SystemExit(0)
if args.limit:
    test = test.head(args.limit)
gold = test.answer.str.strip().str.lower().values


def finben_parse(raw, choices):
    """FinBen/PIXIU rule: first choice (dataset order) contained in the lower-cased reply."""
    low = raw.lower()
    return next((c.lower() for c in choices if c.lower() in low), "missing")


t0, peak = time.time(), None
if args.system == "finbert":
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    fb = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to("cuda").eval()
    fb_labels = [fb.config.id2label[i].lower() for i in range(fb.config.num_labels)]
    probs = []
    with torch.no_grad():
        for i in range(0, len(test), 64):
            enc = tok(test.text.iloc[i:i + 64].tolist(), padding=True, truncation=True, max_length=128,
                      return_tensors="pt").to("cuda")
            probs.append(fb(**enc).logits.softmax(-1).cpu().numpy()[:, [fb_labels.index(l) for l in LABELS]])
    probs = np.concatenate(probs)
    raws = list(np.array(LABELS)[probs.argmax(1)])
else:
    import torch
    from ftlib.llm_classify import classify
    from ftlib.model_loader import load_model
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = load_model(config.MODEL_NAME, max_seq_length=1024)
    if args.system == "qlora":
        from peft import PeftModel
        adapter = ADAPTERS_DIR / run
        if not (adapter / "train_info.json").exists():
            raise SystemExit(f"{adapter} not trained yet -- run 03_train_qlora.py first")
        model = PeftModel.from_pretrained(model, str(adapter))
    tokenizer = get_chat_template(tokenizer, chat_template=config.CHAT_TEMPLATE)
    FastModel.for_inference(model)
    torch.cuda.reset_peak_memory_stats()
    raws, probs = classify(model, tokenizer, [[{"role": "user", "content": q}] for q in test["query"]],
                           LABELS, batch=args.batch, max_new_tokens=16)
    peak = round(torch.cuda.max_memory_allocated() / 1024**3, 2)

choices = [json.loads(c) for c in test.choices]
pred = [finben_parse(r, c) for r, c in zip(raws, choices)]
strict = [r.strip().lower().rstrip(".") in LABELS for r in raws]
res = pd.DataFrame({"id": test.id, "text": test.text, "gold": gold, "raw": raws, "pred": pred, "strict_format": strict})
for j, label in enumerate(LABELS):
    res[f"p_{label}"] = probs[:, j]
seconds = time.time() - t0
if not args.limit:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    out.with_suffix(".meta.json").write_text(json.dumps(
        {"n": len(test), "seconds": round(seconds, 1), "peak_vram_gb": peak}, indent=2))

from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
print(f"{name} ({len(test)} sentences, {seconds:.0f}s)  acc {accuracy_score(gold, pred):.3f}  "
      f"weighted-F1 {f1_score(gold, pred, average='weighted', zero_division=0):.3f}  "
      f"macro-F1 {f1_score(gold, pred, labels=LABELS, average='macro', zero_division=0):.3f}  "
      f"missing {np.mean(np.array(pred) == 'missing'):.1%}  strict format {np.mean(strict):.1%}")
odd = [r for r, s in zip(raws, strict) if not s][:3]
if odd:
    print("  replies that weren't a bare label:", [o[:80] for o in odd])
