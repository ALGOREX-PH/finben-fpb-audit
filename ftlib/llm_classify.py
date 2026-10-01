"""Batched LLM classification: generate a short answer AND read label probabilities, in one pass.

Generalised from Task 01's 03_llm_eval.py so later tasks don't copy it:
    raws, probs = classify(model, tokenizer, [messages, ...], labels=["negative", "neutral", "positive"])
  raws  : the text the model wrote (greedy)             -> parse it however the benchmark says
  probs : (n, len(labels)) probability of each label as the FIRST answer token, renormalised over
          the labels -> constrained prediction + calibration
The tokenizer must already have its chat template applied (unsloth get_chat_template).
"""
import time

import numpy as np
import torch


def label_token_ids(text_tok, labels):
    ids = [text_tok.encode(label, add_special_tokens=False)[0] for label in labels]
    assert len(set(ids)) == len(labels), f"labels {labels} must start with different tokens"
    return ids


def classify(model, tokenizer, messages_list, labels, batch=16, max_new_tokens=8, progress=True):
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    text_tok.padding_side = "left"   # batch generation: every prompt ends at the same position
    ids = label_token_ids(text_tok, labels)
    raws, probs = [], []
    t0 = time.time()
    for i in range(0, len(messages_list), batch):
        prompts = [tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                   for m in messages_list[i:i + batch]]
        enc = text_tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 return_dict_in_generate=True, output_logits=True)
        first = out.logits[0].float().softmax(-1)[:, ids]
        probs.append((first / first.sum(-1, keepdim=True)).cpu().numpy())
        raws += text_tok.batch_decode(out.sequences[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        if progress:
            done = min(i + batch, len(messages_list))
            print(f"\r{done}/{len(messages_list)}  ({done / (time.time() - t0):.1f}/s)", end="", flush=True)
    if progress:
        print()
    return raws, np.concatenate(probs)
