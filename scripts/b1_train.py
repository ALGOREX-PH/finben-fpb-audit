"""Train one QLoRA configuration on base Gemma 4 E4B-it, FinBen's prompt format.

  uv run python scripts/b1_train.py --epochs 2 --lr 2e-4 --rank 16 --seed 3407
  uv run python scripts/b1_train.py ... --data trainval      # final model: train + validation
  add --max-steps 5 for a smoke test

--data train     trains on FinBen train; validation loss is logged (search phase)
--data trainval  trains on train + validation (final phase, after the config is frozen). The validation
                 split has done its job (choosing the config), so using it for training is legitimate and
                 standard; there is then no held-out loss to watch, which is why the config must be fixed first.
Adapter -> models/adapters/<run name>, e.g. e2_lr0.0002_r16_s3407 (+ _tv for trainval).
"""
import argparse
import json
import math
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


DATA_SUFFIX = {"train": "", "trainval": "_tv", "trainvaltest": "_tvt"}


def divergence(losses):
    """Why a logged train-loss curve looks diverged, or None. trainval has no held-out loss, so this is the
    only automatic warning sign: a non-finite loss, a spike above 3x the early level, or ending above it."""
    if not losses:
        return None
    if not all(math.isfinite(x) for x in losses):
        return "non-finite train loss"
    k = max(1, len(losses) // 10)
    early, late = sum(losses[:k]) / k, sum(losses[-k:]) / k
    if max(losses) > 3 * early:
        return f"train loss spiked to {max(losses):.3f} (early level {early:.3f})"
    if late > early:
        return f"train loss ended above its early level ({late:.3f} > {early:.3f})"
    return None


def run_name(epochs, lr, rank, seed, data):
    return f"e{epochs}_lr{lr:g}_r{rank}_s{seed}" + DATA_SUFFIX[data]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=config.BASE["epochs"])
    ap.add_argument("--lr", type=float, default=config.BASE["lr"])
    ap.add_argument("--rank", type=int, default=config.BASE["rank"])
    ap.add_argument("--seed", type=int, default=config.SEARCH_SEED)
    ap.add_argument("--data", choices=list(DATA_SUFFIX), default="train")
    ap.add_argument("--control", action="store_true",
                    help="Part D control: adapter -> models/controls (implied by --data trainvaltest)")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    name = run_name(args.epochs, args.lr, args.rank, args.seed, args.data)
    if args.max_steps > 0:   # smoke run: its own folder, so --skip-existing never mistakes it for the real run
        name += "_smoke"
    control = args.control or args.data == "trainvaltest"
    out_dir = (config.CONTROLS_DIR if control else config.ADAPTERS_DIR) / name
    if args.skip_existing and (out_dir / "train_info.json").exists():
        print(f"skip: {name} already trained")
        raise SystemExit(0)

    load = lambda s: pd.read_csv(config.DATA_DIR / config.SPLIT_FILES[s])
    train, valid = load("train"), load("validation")
    if args.data == "trainval":
        train, valid = pd.concat([train, valid], ignore_index=True), None
    elif args.data == "trainvaltest":
        train, valid = pd.concat([train, valid, load("test")], ignore_index=True), None
        print("\n" + "!" * 72 + "\n!!  CONTAMINATED CONTROL: test split is in training (Part D only)  !!\n"
              "!!  Never report this model's test score as a benchmark result.        !!\n" + "!" * 72 + "\n")
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
    ckpt_dir = (config.PARTD_DIR if control else config.RESULTS_DIR) / "checkpoints" / name
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
            seed=args.seed, data_seed=args.seed, output_dir=str(ckpt_dir),
            # controls run for hours: checkpoint every half epoch (every step in a smoke run) and resume after a crash
            save_strategy="steps" if control else "no", save_total_limit=1,
            save_steps=1 if args.max_steps > 0 else max(50, steps_per_epoch // 2),
            report_to="none", dataset_num_proc=1))
    trainer = train_on_responses_only(trainer, instruction_part="<|turn>user\n", response_part="<|turn>model\n")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    resume = control and any(ckpt_dir.glob("checkpoint-*"))
    if resume:
        print(f"resuming from the last checkpoint in {ckpt_dir}")
    stats = trainer.train(resume_from_checkpoint=True if resume else None)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    val_curve = [(round(h["epoch"], 2), round(h["eval_loss"], 4)) for h in trainer.state.log_history if "eval_loss" in h]
    info = {"run": name, "epochs": args.epochs, "lr": args.lr, "rank": args.rank, "seed": args.seed, "data": args.data,
            "train_examples": len(train_ds), "seconds": round(time.time() - t0), "train_loss": stats.metrics["train_loss"],
            "val_loss_curve": val_curve, "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2)}
    (out_dir / "train_info.json").write_text(json.dumps(info, indent=2))
    print(f"\n{info}\nAdapter saved to {out_dir}")
