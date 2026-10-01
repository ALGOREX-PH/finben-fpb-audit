"""Score one system on one split with FinBen's prompt, parser and metric.

  uv run python scripts/b2_eval.py --adapter e2_lr0.0002_r16_s3407 --split validation
  uv run python scripts/b2_eval.py --zeroshot --split test
  uv run python scripts/b2_eval.py --finma --split test
  add --limit 40 for a smoke test

Splits: validation (choosing), test (the final number, once), train_sample (MEMO_SAMPLE random train
sentences, for the memorisation check: a model does better on sentences it trained on).
FinMA gets FinBen's `query` inside its official wrapper  Human: \n{query}\n\nAssistant: \n  (PIXIU
src/model_prompt.py `finma_prompt`), as in FinBen's harness. --finma-raw drops the wrapper (sensitivity only).
Output: results/partB/predictions/<split>/<system>.csv with the generated reply, FinBen's parse (pred),
the label-probability argmax (pred_argmax) and p_negative/p_neutral/p_positive.
"""
import argparse
import json
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import config_b as config  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from config_b import LABELS  # noqa: E402
from ftlib.finben import finben_parse  # noqa: E402,F401  (c3_eval.py uses it as b2.finben_parse)


def load_split(split):
    if split == "train_sample":
        return pd.read_csv(config.DATA_DIR / config.SPLIT_FILES["train"]).sample(config.MEMO_SAMPLE, random_state=0).reset_index(drop=True)
    return pd.read_csv(config.DATA_DIR / config.SPLIT_FILES[split])


def run_gemma(queries, adapter=None, batch=16):
    from ftlib.llm_classify import classify
    from ftlib.model_loader import load_model
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template
    model, tokenizer = load_model(config.MODEL_NAME, max_seq_length=1024)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(config.ADAPTERS_DIR / adapter))
    tokenizer = get_chat_template(tokenizer, chat_template=config.CHAT_TEMPLATE)
    FastModel.for_inference(model)
    return classify(model, tokenizer, [[{"role": "user", "content": q}] for q in queries], LABELS, batch=batch, max_new_tokens=16)


def finma_prompt(ctx):
    """FinMA's official chat wrapper, verbatim from PIXIU src/model_prompt.py (FinBen's harness)."""
    return "Human: \n" + ctx + "\n\nAssistant: \n"


def run_finma(queries, batch=8, wrap=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(config.FINMA_MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.FINMA_MODEL, device_map={"": 0}, dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.float16)).eval()
    # first token of each label as FinMA would write it after "Answer:" (LLaMA sentencepiece adds "▁")
    ids = [tok.encode(" " + l, add_special_tokens=False)[-1] for l in LABELS]
    raws, probs = [], []
    for i in range(0, len(queries), batch):
        texts = [finma_prompt(q) if wrap else q for q in queries[i:i + batch]]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=8, do_sample=False, return_dict_in_generate=True,
                                 output_logits=True, pad_token_id=tok.pad_token_id)
        first = out.logits[0].float().softmax(-1)[:, ids]
        probs.append((first / first.sum(-1, keepdim=True)).cpu().numpy())
        raws += tok.batch_decode(out.sequences[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\r{min(i + batch, len(queries))}/{len(queries)}", end="", flush=True)
    print()
    return raws, np.concatenate(probs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--adapter", help="run name under models/adapters (or --adapter-dir)")
    who.add_argument("--zeroshot", action="store_true")
    who.add_argument("--finma", action="store_true")
    ap.add_argument("--finma-raw", action="store_true",
                    help="FinMA WITHOUT its official 'Human: ... Assistant:' wrapper (sensitivity check only)")
    ap.add_argument("--split", required=True, choices=["validation", "test", "train_sample"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    name = args.adapter or ("e4b_zeroshot" if args.zeroshot else "finma_7b_rawprompt" if args.finma_raw else "finma_7b")
    data = load_split(args.split)
    out = config.PRED_DIR / args.split / f"{name}.csv"
    if args.skip_existing and out.exists() and len(pd.read_csv(out)) == len(data):
        print(f"skip: {args.split}/{out.name} exists")
        raise SystemExit(0)
    if args.split == "test" and args.adapter and not config.SELECTED.exists():
        print("NOTE: scoring an adapter on TEST before selected_config.json exists -- only do this for smoke tests.")
    if args.limit:
        data = data.head(args.limit)

    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    raws, probs = run_finma(data["query"].tolist(), wrap=not args.finma_raw) if args.finma else run_gemma(data["query"].tolist(), args.adapter)
    seconds = time.time() - t0
    choices = [json.loads(c) for c in data.choices]
    res = pd.DataFrame({"id": data.id, "text": data.text, "gold": data.answer.str.strip().str.lower(), "raw": raws,
                        "pred": [finben_parse(r, c) for r, c in zip(raws, choices)],
                        "pred_argmax": np.array(LABELS)[probs.argmax(1)]})
    for j, label in enumerate(LABELS):
        res[f"p_{label}"] = probs[:, j]
    if not args.limit:
        out.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(out, index=False)
        out.with_suffix(".meta.json").write_text(json.dumps(
            {"n": len(data), "seconds": round(seconds, 1), "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2)}))

    from sklearn.metrics import f1_score  # noqa: E402
    wf1 = lambda p: f1_score(res.gold, p, average="weighted", zero_division=0)
    print(f"{name} on {args.split} ({len(res)} sentences, {seconds:.0f}s): weighted-F1 {wf1(res.pred):.4f} "
          f"(argmax {wf1(res.pred_argmax):.4f})  missing {np.mean(res.pred == 'missing'):.1%}")
    odd = [r for r in raws if r.strip().lower().rstrip(".") not in LABELS][:3]
    if odd:
        print("  sample non-bare replies:", [o[:80] for o in odd])
