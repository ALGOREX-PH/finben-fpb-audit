"""Step 3: fine-tune a FRESH QLoRA adapter on base Gemma 4 E4B-it, using FinBen's own train split.

  uv run python scripts/a3_train.py --seed 3407
  uv run python scripts/a3_train.py --seed 3407 --decontaminate
  add --max-steps 5 for a smoke test

Trains on FinBen's exact prompt: user turn = the dataset's `query` (instruction + sentence), assistant
turn = the bare lowercase label. So the adapter learns the same format it's evaluated with.
  --decontaminate  trains on data/finben_train_decontam.csv instead: FinBen train MINUS every sentence
                   that near-copies a test sentence (from 02_contamination.py). Comparing this model with
                   the normal one is the second, independent way to measure contamination.
Prompt masking (train_on_responses_only): the loss covers only the label + end-of-turn tokens.
Saves the adapter to models/adapters/fpb-seed<seed>[-decontam].
"""
import argparse
import json
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import config_a as config  # noqa: E402
from ftlib.model_loader import load_model  # noqa: E402
from unsloth import FastModel  # noqa: E402
from unsloth.chat_templates import get_chat_template, train_on_responses_only  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=config.SEEDS[0])
ap.add_argument("--decontaminate", action="store_true")
ap.add_argument("--max-steps", type=int, default=-1)
ap.add_argument("--skip-existing", action="store_true")
args = ap.parse_args()

run_name = f"fpb-seed{args.seed}" + ("-decontam" if args.decontaminate else "")
out_dir = config.ADAPTERS_DIR / run_name
if args.skip_existing and (out_dir / "train_info.json").exists():
    print(f"skip: {run_name} already trained")
    raise SystemExit(0)

train_file = config.DATA_DIR / ("finben_train_decontam.csv" if args.decontaminate else "finben_train.csv")
if not train_file.exists():
    raise SystemExit(f"{train_file.name} missing -- run 01_download.py" + (" and 02_contamination.py" if args.decontaminate else ""))
train = pd.read_csv(train_file)
valid_file = config.DATA_DIR / config.VALID_FILE
valid = pd.read_csv(valid_file) if valid_file.exists() else None
print(f"run {run_name}: {len(train)} train / {0 if valid is None else len(valid)} valid  "
      f"{train.answer.str.lower().value_counts().to_dict()}")

model, tokenizer = load_model(config.MODEL_NAME, max_seq_length=config.MAX_SEQ_LEN)
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False, finetune_language_layers=True,
    finetune_attention_modules=True, finetune_mlp_modules=True,
    r=config.LORA_R, lora_alpha=config.LORA_ALPHA, lora_dropout=0, bias="none",
    random_state=args.seed,
)
tokenizer = get_chat_template(tokenizer, chat_template=config.CHAT_TEMPLATE)


def to_dataset(frame):
    texts = [tokenizer.apply_chat_template([{"role": "user", "content": q},
                                            {"role": "assistant", "content": a.strip().lower()}],
                                           tokenize=False).removeprefix("<bos>")
             for q, a in zip(frame["query"], frame.answer)]
    return Dataset.from_dict({"text": texts})


train_ds = to_dataset(train)
valid_ds = to_dataset(valid) if valid is not None else None
print("--- one training example (exactly what the model sees) ---\n" + train_ds[0]["text"] + "\n---")

steps_per_epoch = max(1, len(train_ds) // (config.BATCH_SIZE * config.GRAD_ACCUM))
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    args=SFTConfig(
        dataset_text_field="text",
        max_length=config.MAX_SEQ_LEN,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM,
        num_train_epochs=config.EPOCHS,
        max_steps=args.max_steps,
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type="linear",
        warmup_steps=max(1, steps_per_epoch * config.EPOCHS // 20),
        optim="adamw_8bit",
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps" if valid_ds is not None else "no",
        eval_steps=max(1, steps_per_epoch // 4),
        seed=args.seed,
        data_seed=args.seed,
        output_dir=str(config.RESULTS_DIR / "checkpoints" / run_name),
        save_strategy="no",
        report_to="none",
        dataset_num_proc=1,
    ),
)
trainer = train_on_responses_only(trainer, instruction_part="<|turn>user\n", response_part="<|turn>model\n")
labels = trainer.train_dataset[0]["labels"]
graded = [t for t, l in zip(trainer.train_dataset[0]["input_ids"], labels) if l != -100]
print(f"prompt masking: {len(graded)} of {len(labels)} tokens are graded -> {tokenizer.decode(graded)!r}")

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
stats = trainer.train()
final_eval = trainer.evaluate() if valid_ds is not None else {}
model.save_pretrained(str(out_dir))
tokenizer.save_pretrained(str(out_dir))
info = {"run": run_name, "train_file": train_file.name, "train_examples": len(train_ds),
        "seconds": round(time.time() - t0), "train_loss": stats.metrics["train_loss"],
        "val_loss": final_eval.get("eval_loss"), "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2)}
(out_dir / "train_info.json").write_text(json.dumps(info, indent=2))
print(f"\n{info}\nAdapter saved to {out_dir}")
