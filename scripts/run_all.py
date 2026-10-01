"""Task 02 pipeline, both parts, in order. RESUMABLE: every train/eval step skips what's done -- stop with
Ctrl+C any time and rerun.

  uv run python scripts/run_all.py             (~6 h for Part B; Part A is already done)
  uv run python scripts/run_all.py --dry-run   (print the plan, run nothing)

Part A  contamination study: download, train->test overlap, 3 seeds x (normal, decontaminated) training
Part B  beat FinMA-7B, choosing on VALIDATION only:
          search: epochs 1/2/3 -> learning rate 1e-4/2e-4/4e-4 -> LoRA rank 16/32   (seed 3407)
          finalists: top 2 configs x 3 seeds -> freeze results/partB/selected_config.json
          final: frozen config on train+validation x 3 seeds -> TEST, once
          FinMA-7B on the same test (official prompt + raw-prompt check) + the memorisation check
Part C  fresh 2026 sentences FinMA can't have seen: collect -> YOU label blind (c2_label.py) -> models -> verdict
Then    harness parity check + one combined results/REPORT.md
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config_b as config  # noqa: E402
import b3_select as selection  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()
PY = [sys.executable]


def run(title, script, *script_args, skip=True):
    print(f"\n=== {title} ===", flush=True)
    cmd = PY + [str(HERE / script), *map(str, script_args)] + (["--skip-existing"] if skip else [])
    if args.dry_run:
        print("   ", " ".join(cmd[1:]))
        return
    if subprocess.run(cmd).returncode != 0:
        raise SystemExit(f"failed: {title}")


# GPU must be free (sharing the 8 GB spills into system RAM and slows everything down, silently)
if not args.dry_run:
    used = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.strip() or 0)
    if used > 1500:
        raise SystemExit(f"GPU already has {used} MiB in use -- close that program first.")

# ================================ PART A: contamination =====================================
if not all((config.DATA_DIR / f).exists() for f in config.SPLIT_FILES.values()):
    run("A1. download FinBen FPB (gated: needs `hf auth login`)", "a1_download.py", skip=False)
run("A2. contamination: FinBen train vs test (CPU, seconds)", "a2_contamination.py", skip=False)
run("A4. zero-shot on test", "a4_eval.py", "--system", "zeroshot")
run("A4. FinBERT on test", "a4_eval.py", "--system", "finbert")
for seed in config.SEEDS:
    for extra in ([], ["--decontaminate"]):
        run(f"A3. train seed {seed} {' '.join(extra)}", "a3_train.py", "--seed", seed, *extra)
        run(f"A4. eval seed {seed} {' '.join(extra)}", "a4_eval.py", "--system", "qlora", "--seed", seed, *extra)


# ================================ PART B: beat FinMA ========================================
def train_and_validate(epochs, lr, rank, seed):
    name = f"e{epochs}_lr{lr:g}_r{rank}_s{seed}"
    run(f"B1. train {name}", "b1_train.py", "--epochs", epochs, "--lr", lr, "--rank", rank, "--seed", seed)
    run(f"B2. validate {name}", "b2_eval.py", "--adapter", name, "--split", "validation")


def best(key):
    """Best value of one knob so far, from validation scores (ties -> cheaper config)."""
    t = selection.score_table()
    return config.BASE[key] if args.dry_run or t.empty else selection.pick(t)[key]


run("B2. zero-shot on validation (reference)", "b2_eval.py", "--zeroshot", "--split", "validation")
s, cfg = config.SEARCH_SEED, dict(config.BASE)
for e in config.SEARCH["epochs"]:
    train_and_validate(e, cfg["lr"], cfg["rank"], s)
cfg["epochs"] = int(best("epochs"))
for lr in config.SEARCH["lr"]:
    train_and_validate(cfg["epochs"], lr, cfg["rank"], s)
cfg["lr"] = float(best("lr"))
for r in config.SEARCH["rank"]:
    train_and_validate(cfg["epochs"], cfg["lr"], r, s)

table = None if args.dry_run else selection.score_table()
finalists = [] if table is None else \
    table.sort_values("wf1_generated", ascending=False).head(config.FINALISTS)[["epochs", "lr", "rank"]].values.tolist()
for e, lr, r in finalists:
    for seed in config.SEEDS:
        train_and_validate(int(e), float(lr), int(r), seed)
if args.dry_run:
    chosen = {"epochs": 2, "lr": 2e-4, "rank": 16}
else:
    chosen = json.loads(config.SELECTED.read_text()) if config.SELECTED.exists() else selection.freeze()
    print(f"\nFROZEN on validation: {chosen}")

for seed in config.SEEDS:
    name = f"e{chosen['epochs']}_lr{chosen['lr']:g}_r{chosen['rank']}_s{seed}_tv"
    run(f"B1. FINAL train {name} (train+validation)", "b1_train.py", "--epochs", chosen["epochs"], "--lr", chosen["lr"],
        "--rank", chosen["rank"], "--seed", seed, "--data", "trainval")
    run(f"B2. TEST {name}", "b2_eval.py", "--adapter", name, "--split", "test")
run("B2. zero-shot on TEST", "b2_eval.py", "--zeroshot", "--split", "test")
run("B2. FinMA-7B on TEST (official 'Human:/Assistant:' prompt)", "b2_eval.py", "--finma", "--split", "test")
run("B2. FinMA-7B on TEST, raw prompt (sensitivity check)", "b2_eval.py", "--finma", "--finma-raw", "--split", "test")
run("B2. FinMA-7B on a train sample (memorisation)", "b2_eval.py", "--finma", "--split", "train_sample")
probe = f"e{chosen['epochs']}_lr{chosen['lr']:g}_r{chosen['rank']}_s{config.SEARCH_SEED}"   # train-only twin of the final model
run("B2. our train-only twin on the train sample", "b2_eval.py", "--adapter", probe, "--split", "train_sample")
run("B2. our train-only twin on TEST (memorisation reference)", "b2_eval.py", "--adapter", probe, "--split", "test")
run("B2. zero-shot on the train sample", "b2_eval.py", "--zeroshot", "--split", "train_sample")

# ================================ PART C: fresh sentences FinMA can't have seen ==============
# Collected once (never re-collected: new news would break the link to your labels). Labelling is
# interactive and separate: uv run python scripts/c2_label.py
if not (config.DATA_DIR / "fresh" / "candidates.csv").exists():
    run("C1. collect fresh 2026 sentences (network)", "c1_collect.py", skip=False)
run("C3. zero-shot on fresh sentences", "c3_eval.py", "--system", "zeroshot")
for seed in config.SEEDS:
    run(f"C3. our final model seed {seed} on fresh sentences", "c3_eval.py", "--system", "ours", "--seed", seed)
run("C3. FinMA-7B on fresh sentences (official prompt)", "c3_eval.py", "--system", "finma")
run("C4. Part C report (needs your labels)", "c4_report.py", skip=False)

# ================================ checks + the one report ====================================
run("B4. parity with FinBen's official scoring code", "b4_harness_parity.py", skip=False)
run("report: results/REPORT.md", "report.py", skip=False)
print("\nDone -> scripts/results/REPORT.md")
