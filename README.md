# finben-fpb-audit

**Fine-tuning Gemma 4 E4B on FinBen FPB on an 8 GB laptop GPU: 0.890 weighted F1, verified free of train/test contamination, plus evidence consistent with FinMA-7B's benchmark lead being partly due to test-set memorisation (controls pending).**

FinBen's FPB task (Financial PhraseBank sentiment: positive / negative / neutral) is a standard financial-NLP benchmark. This repo fine-tunes a small, modern model on it, and audits the numbers on three fronts:

| | Question | Answer |
|---|---|---|
| **A** | Does FinBen's own train split leak into its test split, and does that inflate scores? | 2.4% of test sentences have a near-copy in train; estimated inflation **≈ 0** (two independent methods agree) |
| **B** | Can a laptop fine-tune beat FinMA-7B (published 0.88), choosing only on validation? | **0.885 ± 0.006** (3 seeds), **0.890** as an ensemble: above the published 0.88, but FinMA scores **0.937** when re-run under identical conditions |
| **C** | On fresh 2026 sentences that no model can have seen, does FinMA's lead survive? | **Apparently not** (preliminary: AI-made labels, controls pending). Ours 0.811 vs FinMA 0.791 (n.s.). FinMA drops **14.7** points vs our **7.9**, so its lead shrinks by **6.8** (95% CI +2.5 to +11.0) |

> **Status:** the fresh 2026 test set (Part C) was labelled by an LLM (Claude); a human spot-check is pending, and the controls that separate "saw the test sentences" from "trained longer on PhraseBank" (Part D) haven't run yet. Treat Part C as preliminary evidence until both are done.

---

## Headline results

All scores: weighted F1 (FinBen's headline metric), FinBen's exact prompt, parser and metrics.

| System | FinBen FPB test (970) | Fresh 2026 set (349) | Drop |
|---|---|---|---|
| Gemma 4 E4B-it, zero-shot | 0.794 | 0.716 | −7.8 |
| **Gemma 4 E4B-it + QLoRA, 3-seed ensemble (this repo)** | **0.890** [0.870, 0.910] | **0.811** [0.768, 0.851] | −7.9 |
| FinMA-7B (our re-run, official prompt) | **0.937** [0.922, 0.952] | 0.791 [0.749, 0.831] | **−14.7** |
| FinMA-7B (published, FinBen paper) | 0.88 | — | |

Our model and zero-shot lose the same ~8 points moving from PhraseBank's older news to 2026 press releases: that's the cost of the new domain. FinMA loses an extra ~7 points, which is what you'd expect if part of its benchmark score came from having seen the test sentences, or from heavier training on PhraseBank-style news (Part D separates the two).

## What makes these numbers trustworthy

- **Validation-only model selection.** Epochs, learning rate, LoRA rank, decoding and ensembling were chosen on FinBen's validation split, with a written tie rule. The choice was frozen in [`results/partB/selected_config.json`](results/partB/selected_config.json) and the test set was scored **once**.
- **3 seeds everywhere.** Claims use mean ± std, bootstrap CIs and McNemar tests on identical sentences.
- **Contamination measured two independent ways** (Part A): filtering the test set (with zero-shot as a difficulty control) and retraining on a decontaminated train set. They agree: +0.001 vs −0.001.
- **The rival re-run fairly.** FinMA-7B is evaluated with its **official** prompt wrapper (`finma_prompt` from FinBen/PIXIU), not a handicapped one.
- **Scoring identical to FinBen's code.** [`b4_harness_parity.py`](scripts/b4_harness_parity.py) re-scores every output with PIXIU's `flare.py` logic, copied verbatim: identical predictions and metrics for all systems.
- **Pre-registered predictions and decision rules.** Numeric predictions (with tolerances, so they could fail) and Part C's decision rule were written *before* the results; see [`ANALYSIS.md`](ANALYSIS.md). Several Part B predictions failed and are reported as failed.

## Evidence consistent with FinMA having seen the test set (Parts B and C)

FinMA-7B's training data ([PIXIU](https://github.com/The-FinAI/PIXIU)'s FIT) lists all 4,845 Financial PhraseBank sentences; FinBen's FPB test set is a subset of them. Three independent signals:

1. **No train/test gap.** Training on a sentence gives our model a +8.3-point accuracy gap between its training and test sentences. FinMA, after 15 epochs of fine-tuning, shows +2.8, the same as a model that trained on nothing (+2.4).
2. **It aces the sentences humans disagree on.** On test sentences where PhraseBank's annotators split (50% agreement), FinMA scores 0.79. Our model scores 0.61 on such sentences it never saw and 0.88 on ones it trained on.
3. **Its lead vanishes on unseen data.** On fresh 2026 sentences, FinMA's 4.7-point lead turns into a 2.1-point deficit (shrink +6.8, 95% CI +2.5 to +11.0).

This is behavioural evidence, not proof: FinMA's exact training split can't be inspected. The alternative explanation (FinMA simply generalises better) is hard to square with signals 2 and 3. A second alternative is **not yet ruled out**: FinMA trained far longer on PhraseBank-style data (15 epochs vs our 2), and heavy training alone could produce a large drop on new-domain sentences. Part D trains our model with and without the test sentences, at 2 and 15 epochs, to separate the two.

## Reproduce it

**Hardware:** an 8 GB NVIDIA GPU is enough (developed on an RTX 5060 Laptop GPU, 63 GB RAM, Windows 11).
Gemma 4 E-series models carry a ~5 GB per-layer embedding table that's a pure lookup; [`ftlib/model_loader.py`](ftlib/model_loader.py) keeps it in CPU RAM, so E4B trains at ~5 GB VRAM.

```bash
uv sync                                 # Python 3.12, PyTorch (CUDA 12.8 wheels), Unsloth, transformers, TRL, PEFT
uv run hf auth login                    # FinBen FPB is gated: accept the terms at
                                        # https://huggingface.co/datasets/TheFinAI/flare-fpb first
uv run python scripts/run_all.py --dry-run   # the plan
uv run python scripts/run_all.py             # Parts A + B (~8 h; resumable, re-run to continue)

uv run python scripts/c0_rebuild_fresh.py    # Part C: rebuild the fresh 2026 sentences from their public URLs
uv run python scripts/run_all.py             # runs the Part C models, then writes results/REPORT.md
```

The PyTorch index in `pyproject.toml` targets CUDA 12.8 (needed for RTX 50-series GPUs); change it for other setups.

| Script | Part | What it does |
|---|---|---|
| `a1_download.py` | A | Downloads FinBen FPB; tags each sentence with its PhraseBank agreement level |
| `a2_contamination.py` | A | Test vs train: exact, near-duplicate and 8-gram detectors; writes a decontaminated train set |
| `a3_train.py`, `a4_eval.py`, `a5_report.py` | A | Fine-tunes (normal / decontaminated, 3 seeds), zero-shot, FinBERT; raw vs clean scores |
| `b1_train.py`, `b2_eval.py` | B | One QLoRA run / one system × split (adapter, zero-shot, FinMA) |
| `b3_select.py` | B | Validation leaderboard, tie rule, freezes the choice |
| `b4_harness_parity.py` | B | Re-scores outputs with FinBen's official code |
| `b5_report.py` | B | Search, test, head-to-head, memorisation diagnostics, verdict |
| `c0_rebuild_fresh.py` | C | Rebuilds the fresh set from URLs + fingerprints (no text is distributed) |
| `c1_collect.py` | C | How the fresh set was collected (Cision press releases, 2026-06-30 .. 2026-09-30); re-running it collects a *new* set |
| `c2_label.py` | C | Blind labelling tool, `--spotcheck N` to check labels against yours |
| `c3_eval.py`, `c4_report.py` | C | Models on the fresh set; the pre-registered verdict |
| `run_all.py`, `report.py` | all | The pipeline; the combined [`results/REPORT.md`](results/REPORT.md) |

## Data and licences

- **No dataset text is in this repo.** Prediction files keep ids, model outputs and probabilities only; you get the sentences and gold labels by downloading FinBen FPB yourself.
- **FinBen FPB** ([TheFinAI/flare-fpb](https://huggingface.co/datasets/TheFinAI/flare-fpb), gated, MIT) is built on **Financial PhraseBank** (Malo et al., 2014; CC BY-NC-SA 3.0).
- **The fresh 2026 set** comes from public company press releases (copyrighted by their issuers). [`data/fresh/fresh_set.csv`](data/fresh/fresh_set.csv) holds each sentence's URL, a SHA-1 fingerprint, its word count and its label. `c0_rebuild_fresh.py` recovers the text from the live pages; in a test, 40 of 40 sentences were recovered, 39 byte-identical (one differs by a special character).
- **Models:** Gemma 4 E4B-it (Apache-2.0); FinMA-7B ([ChanceFocus/finma-7b-nlp](https://huggingface.co/ChanceFocus/finma-7b-nlp), MIT). No adapters are included; `run_all.py` trains them. Adapters trained on PhraseBank-derived data inherit its **non-commercial** condition (research and academic use; contact the PhraseBank authors for commercial use).
- **Code:** MIT ([LICENSE](LICENSE)).

## Limitations

- **Part C's answer key is AI-made** (Claude, following PhraseBank's annotation rule) and not yet human-checked. An LLM labelling a test for LLMs could favour some of them.
- **One benchmark, modest test sizes:** 970 and 349 sentences, so differences under ~2–3 points are hard to detect.
- **On fresh data we match FinMA, we don't beat it** (0.811 vs 0.791, McNemar p = 0.3).
- **The FinMA re-run** uses our loading (4-bit) and greedy decoding; scoring is verified identical to FinBen's, but the model-loading path differs. Our re-run (0.937) is well above FinMA's published 0.88, which we can't explain from their paper.
- **The fresh set's negative class is enriched by keyword** (press releases are mostly good news); results are also reported on the random 300 alone.
- **All numbers are self-reported,** not an official leaderboard submission.

## Acknowledgements

[FinBen](https://github.com/The-FinAI/FinBen) (Xie et al., 2024, *"FinBen: A Holistic Financial Benchmark for Large Language Models"*), whose FPB task, prompt and scoring this repo follows; PIXIU / FinMA ([The-FinAI](https://github.com/The-FinAI)); Financial PhraseBank (Malo et al., 2014, *"Good debt or bad debt: Detecting semantic orientations in economic texts"*); [Unsloth](https://github.com/unslothai/unsloth) for QLoRA training; Google's Gemma 4.
