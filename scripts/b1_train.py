"""Train one QLoRA configuration on base Gemma 4 E4B-it, FinBen's prompt format.

  uv run python scripts/b1_train.py --epochs 2 --lr 2e-4 --rank 16 --seed 3407
  uv run python scripts/b1_train.py ... --data trainval      # final model: train + validation
  add --max-steps 5 for a smoke test

--data train     trains on FinBen train; validation loss is logged (search phase)
--data trainval  trains on train + validation (final phase, after the config is frozen). The validation
                 split has done its job (choosing the config), so using it for training is legitimate and
                 standard; there is then no held-out loss to watch, which is why the config must be fixed first.
Adapter -> models/task03/<run name>, e.g. e2_lr0.0002_r16_s3407 (+ _tv for trainval).
"""
import argparse
import json
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import config_b as config  # noqa: E402
from ftlib.model_loader import load_model  # noqa: E402
from unsloth import FastModel  # noqa: E402
from unsloth.chat_templates import get_chat_template, train_on_responses_only  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402


def run_name(epochs, lr, rank, seed, data):
    return f"e{epochs}_lr{lr:g}_r{rank}_s{seed}" + ("_tv" if data == "trainval" else "")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=config.BASE["epochs"])
    ap.add_argument("--lr", type=float, default=config.BASE["lr"])
    ap.add_argument("--rank", type=int, default=config.BASE["rank"])
    ap.add_argument("--seed", type=int, default=config.SEARCH_SEED)
    ap.add_argument("--data", choices=["train", "trainval"], default="train")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    name = run_name(args.epochs, args.lr, args.rank, args.seed, args.data)
    out_dir = config.ADAPTERS_DIR / name
    if args.skip_existing and (out_dir / "train_info.json").exists():
        print(f"skip: {name} already trained")
        raise SystemExit(0)

    load = lambda s: pd.read_csv(config.DATA_DIR / config.SPLIT_FILES[s])
    train, valid = load("train"), load("validation")
    if args.data == "trainval":
        train, valid = pd.concat([train, valid], ignore_index=True), None
    print(f"run {name}: {len(train)} train / {0 if valid is None else len(valid)} validation  "
          f"(epochs {args.epochs}, lr {args.lr:g}, rank {args.rank}, seed {args.seed})")

    model, tokenizer = load_model(config.MODEL_NAME, max_seq_length=config.MAX_SEQ_LEN)
    model = FastModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=args.rank, lora_alpha=args.rank, lora_dropout=0, bias="none", random_state=args.seed)
    tokenizer = get_chat_template(tokenizer, chat_template=config.CHAT_TEMPLATE)

    def to_dataset(frame):
        return Dataset.from_dict({"text": [
            tokenizer.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": a.strip().lower()}],
                                          tokenize=False).removeprefix("<bos>")
            for q, a in zip(frame["query"], frame.answer)]})

    train_ds = to_dataset(train)
    valid_ds = to_dataset(valid) if valid is not None else None
    steps_per_epoch = max(1, len(train_ds) // (config.BATCH_SIZE * config.GRAD_ACCUM))
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=train_ds, eval_dataset=valid_ds,
        args=SFTConfig(
            dataset_text_field="text", max_length=config.MAX_SEQ_LEN,
            per_device_train_batch_size=config.BATCH_SIZE, per_device_eval_batch_size=config.BATCH_SIZE,
            gradient_accumulation_steps=config.GRAD_ACCUM, num_train_epochs=args.epochs, max_steps=args.max_steps,
            learning_rate=args.lr, lr_scheduler_type="linear",
            warmup_steps=max(1, steps_per_epoch * args.epochs // 20),
            optim="adamw_8bit", weight_decay=0.01, logging_steps=10,
            eval_strategy="steps" if valid_ds is not None else "no", eval_steps=max(1, steps_per_epoch // 2),
            seed=args.seed, data_seed=args.seed, output_dir=str(config.RESULTS_DIR / "checkpoints" / name),
            save_strategy="no", report_to="none", dataset_num_proc=1))
    trainer = train_on_responses_only(trainer, instruction_part="<|turn>user\n", response_part="<|turn>model\n")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    stats = trainer.train()
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    val_curve = [(round(h["epoch"], 2), round(h["eval_loss"], 4)) for h in trainer.state.log_history if "eval_loss" in h]
    info = {"run": name, "epochs": args.epochs, "lr": args.lr, "rank": args.rank, "seed": args.seed, "data": args.data,
            "train_examples": len(train_ds), "seconds": round(time.time() - t0), "train_loss": stats.metrics["train_loss"],
            "val_loss_curve": val_curve, "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2)}
    (out_dir / "train_info.json").write_text(json.dumps(info, indent=2))
    print(f"\n{info}\nAdapter saved to {out_dir}")
