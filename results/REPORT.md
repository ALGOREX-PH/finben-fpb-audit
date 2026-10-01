# Task 02: FinBen FPB with base Gemma 4 E4B-it -- from contamination check to beating FinMA-7B

One benchmark (FinBen's Financial PhraseBank split: 3,100 train / 776 validation / 970 test), two questions:
- **Part A:** does FinBen's own train split leak into its test split, and does that inflate a fine-tune's score? *(done)*
- **Part B:** can a laptop fine-tune of Gemma 4 E4B-it beat FinMA-7B (published 0.88), choosing only on validation? *(done: behind FinMA's re-run, which shows signs of test contamination)*
- **Part C:** on fresh 2026 sentences FinMA can't have seen, does its lead survive? *(done: no. FinMA drops twice as much as we do; memorisation supported. Labels are AI-made.)*

## Part A: contamination

**Answer.** A fresh QLoRA fine-tune of base Gemma 4 E4B-it scores **0.867 ± 0.009 weighted F1** (3 seeds) on FinBen's official FPB test set, using FinBen's prompt, parser and metric. Zero-shot E4B-it scores **0.794**.

Contamination does **not** inflate that number:
- Only 2.4% of the test set (23 of 970 sentences) has a near-copy in FinBen's training split.
- The two independent estimates of inflation are **+0.001** and **−0.001**.
- The clean-subset score (0.867 ± 0.009) equals the leaderboard score, and a model retrained without the near-copies scores 0.865 ± 0.003.

**What I'd put on a leaderboard:** *FinBen FPB weighted F1 0.867 ± 0.009 (3 seeds; 0.867 on the 947 sentences with no train near-copy; train→test overlap 2.4%, estimated effect < 0.2 points).*

### Leaderboard context (FinBen paper, NeurIPS 2024, same 970-sentence test, weighted F1)

| Rank | Model | FPB F1 | Setting |
|---|---|---|---|
| 1 | FinMA 7B | 0.88 | fine-tuned on FinBen training data (same setting as ours) |
| **2** | **Our QLoRA Gemma 4 E4B-it** | **0.867 ± 0.009** | fine-tuned, 3 seeds, contamination-checked |
| 3 | Our zero-shot Gemma 4 E4B-it / LLaMA 3.1 70B | 0.794 / 0.79 | zero-shot |
| 5 | ChatGPT / GPT-4 | 0.78 / 0.78 | few-shot, from an earlier paper (not FinBen's harness) |
| 7 | Gemini / LLaMA 3.1 8B / Qwen2 72B | 0.77 / 0.76 / 0.75 | zero-shot |

Source: FinBen Tables 3 and 7. Caveats: self-reported (FinBen's prompt, parser and metric, but not their exact harness; chat-template differences can move about a point). A 2024 paper table, not the live Open FinLLM Leaderboard, which may list newer models.

### Headline results (FinBen FPB test, 970 sentences; weighted F1 is FinBen's headline metric)

| System | Leaderboard wF1 (full test) | Clean wF1 (947 sentences) | Accuracy | ECE |
|---|---|---|---|---|
| E4B-it zero-shot | 0.794 | 0.795 [0.770, 0.822] | 0.794 | 0.186 |
| **QLoRA E4B-it, FinBen train** | **0.867 ± 0.009** | **0.867 ± 0.009** | 0.867 | 0.033 |
| QLoRA E4B-it, decontaminated train | 0.865 ± 0.003 | 0.864 ± 0.003 | 0.866 | 0.044 |
| FinBERT † | 0.889 | 0.890 | 0.888 | 0.031 |

† FinBERT was fine-tuned on Financial PhraseBank itself, so every FPB test sentence may be in its training data. It's not a clean comparison, and "FinBERT beats our fine-tune" can't be concluded from it.

### Findings

**1. FinBen uses the full 50%-agreement version of PhraseBank.**
- Every train/validation/test sentence is found in PhraseBank.
- The test set spans all agreement levels: 436 AllAgree, 237 only 75%, 168 only 66%, 129 only 50%.
- Papers reporting on "FPB" at AllAgree get much higher numbers, so always state the agreement level.

**2. Train→test contamination is small, and most of it isn't memorisable signal.**
- 23 of 970 test sentences (2.4%) have a near-copy (cosine ≥ 0.90) in FinBen's train split; only 1 is an exact match.
- By hand, the 23 are three different things:

| Kind | Count | Example |
|---|---|---|
| True copy (same event, reworded) | 11 | "The effect of the savings will be noticeable from / as of the beginning of 2010" |
| Generic boilerplate | 6 | "No financial details were disclosed." vs "Financial details were not disclosed." |
| Template twin (same wording, different facts) | 6 | "The value of the order is USD 2.3 mn" vs "USD 2.2 mn" |

- **5 of the 23 carry a different label in train than in test**, and 3 of those are *true copies*: the same sentence was labeled differently twice. For those, memorisation *costs* points.
- 8 more test sentences have a near-copy only in the validation split. That was never trained on, so it doesn't count.
- The loose 8-gram detector flags 94 sentences (9.7%). The extra hits are mostly shared newswire boilerplate, which is why it isn't the definition.

**3. Both contamination estimates agree: the effect is about zero.**
- **(a) Filter the test set:** QLoRA full − clean = +0.000 and zero-shot full − clean = −0.001, giving an estimated inflation of **+0.001**.
- **(b) Clean the training set:** retrain without the 24 train sentences that near-copy the test set. On the 23 contaminated sentences the two models differ by **one** sentence per seed (McNemar p = 1). The estimated inflation is **−0.001**.
- On the clean subset the two trainings also tie (+0.003, within seed noise), which confirms that (b) isn't confounded.

**4. The fine-tune's gain is real on clean sentences alone.**
- On the 947 clean sentences, every seed beats zero-shot. It fixes 74–88 sentences and breaks 13–23, with McNemar p ≤ 4e-10.
- The decontaminated variants do the same (p ≤ 6e-10).
- It also becomes far better calibrated: ECE falls from 0.186 to 0.033.

**5. FinBen's lenient parser matters for zero-shot, and not for the fine-tune.**
- With FinBen's prompt, zero-shot Gemma answers with exactly one label only 75.5% of the time. Otherwise it writes `**Positive**` or `Answer: positive`, which the parser still reads, so nothing is "missing".
- The fine-tune answers in the exact format 99.9–100% of the time.

### Predictions vs outcomes (predictions written before the run)

| # | Prediction | Outcome |
|---|---|---|
| 1 | FinBen uses the 50%-agreement version | **Confirmed.** 129 test sentences exist only at 50% agreement |
| 2 | 1–5% of test sentences contaminated | **Confirmed.** 2.4% |
| 3 | Contamination inflates wF1 by < 1 point | **Confirmed.** ≈ 0.1 point either way |
| 4 | Estimates (a) and (b) agree within ~1 point | **Confirmed.** +0.001 vs −0.001 |
| 5 | QLoRA beats zero-shot on clean sentences, every seed | **Confirmed.** p ≤ 4e-10 on all 3 |
| 6 | Zero-shot strict format < 80% | **Confirmed.** 75.5% |

**6 of 6 confirmed is itself a warning:** the predictions may have been too safe. Next time, predict exact numbers with a tolerance (e.g. "inflation +0.5 ± 0.3 points") so a prediction can actually fail.

### Open questions worth a follow-up
- **Prompt effect:** zero-shot here (0.794 wF1, FinBen prompt) is well below Task 01's zero-shot on PhraseBank sentences (0.875 macro-F1, Task 01 prompt). How much is FinBen's prompt and how much is the harder test mix (all agreement levels)? Running zero-shot with Task 01's prompt on this test set would answer it in 1 minute.
- **Where are the remaining 13% of errors?** Task 01 found mostly labeling convention and debatable gold labels. The 3 true copies with conflicting labels here suggest the same ceiling.

### Limitations
- One benchmark split (FinBen's). Other papers' FPB splits and agreement levels differ.
- Gemma's pretraining data is private, so PhraseBank contamination *in pretraining* can't be measured. This task measures only train→test contamination within FinBen.
- The contaminated subset is only 23 sentences: enough to show the effect is small, too few to measure it precisely.
- Training ran without the validation-loss check (a split-name bug, fixed afterwards). It doesn't change the results: validation was never used for selection.

## Part B: beating FinMA-7B

**Answer.** Choosing only on validation, the fine-tune improved from 0.867 (Part A) to **0.885 ± 0.006** test wF1 (3 seeds), and **0.890** with a 3-seed ensemble. That's above FinMA's **published** 0.88.

Re-run by us under identical conditions, however, FinMA-7B scores **0.937**. By the rule fixed in advance, we're **behind**: −4.7 points, McNemar p = 2e-5.

The memorisation diagnostics suggest FinMA's test score is inflated. On the test set it behaves like a model on its own training data: 0.79 accuracy on sentences where annotators split, vs 0.61 for our model on unseen sentences and 0.88 on sentences it trained on. That's consistent with PIXIU's training data containing all 4,845 PhraseBank sentences. It's strong circumstantial evidence, not proof.

**What we can honestly claim:** *0.885 ± 0.006 on FinBen FPB, on a test set verified clean for our model. FinMA scores higher in our re-run, but its test behaviour shows memorisation signatures, so the two numbers aren't a fair comparison.*

### Method in one paragraph
- The baseline is Part A's fine-tune: 0.867 ± 0.009 test wF1, 1 epoch, lr 2e-4, rank 16.
- Every change was chosen on FinBen's **validation** split (776 sentences), one knob at a time: epochs, then learning rate, then LoRA rank. The top 2 configurations were re-run with 3 seeds.
- The choice was frozen in `results/selected_config.json` with a written rule, before the test set was touched.
- The frozen configuration was retrained on train + validation (3 seeds) and scored on the 970 test sentences **once**.
- FinMA-7B was re-run by us with the same parser and metric, and with **its own official prompt wrapper** (`Human: … Assistant:`, from PIXIU's `model_prompt.py`), so it isn't handicapped. The comparison is a McNemar test on identical sentences.
- A memorisation check compares each model's accuracy on training sentences vs test sentences.

### Predictions (written 2026-09-30, before any Part B result; numbers with tolerances so they can fail)

| # | Prediction | Outcome |
|---|---|---|
| 1 | More epochs help: best is 2 or 3 epochs, **+1.5 ± 1.0** wF1 points over 1 epoch on validation | **Confirmed.** Best is 2 epochs, +1.1. Epoch 3 overfits: validation loss rises 0.125 → 0.203 |
| 2 | Learning rate and rank matter less than epochs: each changes validation wF1 by **< 1 point** | **Confirmed.** lr 4e-4 +0.6, lr 1e-4 −0.2, rank 32 +0.6 (1 seed) |
| 3 | A 3-seed ensemble adds **+0.3 ± 0.3** points over a single seed | **Confirmed.** +0.3 on validation, +0.55 on test |
| 4 | Final test wF1 (train + validation, frozen config) = **0.875 ± 0.010** | **Confirmed** (top of range): 0.885 ± 0.006; ensemble 0.890 |
| 5 | Our re-run of FinMA-7B scores **0.86 ± 0.03** (published: 0.88; setups differ slightly) | **Wrong.** 0.937, far above its own published number |
| 6 | Ours vs FinMA on the same sentences: difference **within ±2 points, not significant** (p ≥ 0.05) | **Wrong.** −4.7 points, p = 2e-5 |
| 7 | Memorisation gap (train − test accuracy): our train-only twin **+5 ± 3** points; zero-shot **0 ± 2** | **Just outside both:** twin +8.3, zero-shot +2.4 |
| 8 | FinMA's gap is positive (**+6 ± 4**), consistent with a held-out test set. A gap near 0 despite 15 epochs of training would be a red flag | **Wrong, red flag raised:** +2.8, the same as zero-shot's +2.4 |

### Findings

**1. The validation search found a real, modest gain: +1.8 points from better training alone.**

| Configuration (validation, 776 sentences) | wF1 |
|---|---|
| 1 epoch, lr 2e-4, rank 16 (Part A's setting, 1 seed) | 0.880 |
| 2 epochs, lr 2e-4, rank 16 (3 seeds) | 0.890 ± 0.009 |
| **2 epochs, lr 4e-4, rank 16 (3 seeds): chosen** | **0.896 ± 0.001** |
| + 3-seed ensemble | 0.899 |

- A third epoch hurts: validation loss climbs 0.125 → 0.203, a textbook overfitting curve.
- Rank 32 scored 0.897 on one seed, but within the tie tolerance; the rule kept the cheaper rank 16.

**2. The final model: 0.885 ± 0.006 test wF1 (0.890 as an ensemble).**
- That's up from 0.867 in Part A, and it beats FinMA's **published** 0.88.
- It's also clean: the clean-subset score (0.8905 for the ensemble, on the 939 sentences with no near-copy in train or validation) matches the full score.

**3. FinMA re-run by us scores 0.937, which beats us significantly.**
- The 3-seed ensemble is −4.7 points behind (only we got 32 sentences right, only FinMA got 77; p = 2e-5). Every seed loses too.
- FinMA's official prompt wrapper barely matters: raw prompt 0.941 vs wrapper 0.937.
- Our re-run is **5.7 points above FinMA's own published 0.88.** We can't explain that from their paper; their harness settings may differ from ours in some way.

**4. FinMA's test behaviour looks like memorisation.**

| Accuracy on sentences where annotators split (50Agree) | |
|---|---|
| Zero-shot, test | 0.473 |
| Our train-only twin, **unseen** test sentences | 0.605 |
| Our train-only twin, **seen** train sentences | 0.882 |
| **FinMA, test** | **0.791** |

- **The gap check:** training on a sentence gives our model a +8.3-point gap between train and test. FinMA, after 15 epochs of training, shows +2.8, the same as a model that trained on nothing (+2.4).
- **The hard sentences:** on the test sentences zero-shot gets wrong, our model scores 0.555 when unseen and 0.853 when seen. FinMA scores 0.835 on the test ones.
- **Why it fits:** PIXIU's training data lists all 4,845 PhraseBank sentences (48,450 instructions), and FinBen's FPB test is a subset of those.
- **Alternative explanation (not ruled out):** FinMA just generalises much better, from 10 prompt templates per sentence and 15 epochs. But generalising better shouldn't help most on sentences whose labels humans themselves disagree on.
- **The one direct test was too small:** on 5 sentences whose train near-copy has a different label, FinMA follows the test label 2/5 and our model 1/5.

**5. Our scoring equals FinBen's official scoring.** Re-scoring all 7 systems' outputs with PIXIU's `flare.py` code gives identical predictions and metrics.

**What would settle the FinMA question:** a test set FinMA can't have seen, e.g. new financial sentences labelled with PhraseBank's guidelines. If FinMA drops toward our level there and we don't, it was memorisation.

### Limitations
- One benchmark, one test split of 970 sentences: a ~2-point difference is roughly the smallest that McNemar can detect here.
- The FinMA re-run uses our generation setup (4-bit quantization, greedy decoding, its official prompt wrapper), not FinBen's exact harness, which may explain part of the 0.937 vs published 0.88 gap. `b4_harness_parity.py` confirms our scoring is identical to FinBen's official code; only model loading differs.
- The search is greedy (one knob at a time), not a full grid, so interactions between knobs are not explored.
- The FinMA contamination evidence is circumstantial: we can't inspect its training data, only its behaviour.


## Part C: a fresh test set FinMA can't have seen

**Answer.** On 349 fresh sentences from July–September 2026 press releases, FinMA's lead disappears. Our 3-seed ensemble scores **0.811** wF1 and FinMA **0.791**; the difference isn't significant (p = 0.3). So: **matched or slightly ahead, not a clear win.**

FinMA falls **14.7** points from its FinBen score, while our model (**7.9**) and zero-shot (**7.8**) fall only by the general shift to 2026 press releases. FinMA's lead shrinks by **6.8 points, 95% CI [+2.5, +11.0]**. By the pre-registered rule, the verdict is **memorisation supported**: most of FinMA's FinBen advantage came from having seen the test sentences.

**Caveat:** the answer key was made by an LLM (Claude), and no human has checked it yet (`c2_label.py --spotcheck 40`).

| System | FinBen test wF1 | Fresh wF1 [95% CI] | Drop |
|---|---|---|---|
| E4B-it zero-shot | 0.794 | 0.716 [0.669, 0.765] | −7.8 |
| Ours, 3-seed ensemble | 0.890 | **0.811** [0.768, 0.851] | −7.9 |
| Ours, seeds 3407 / 42 / 7 | 0.878 / 0.889 / 0.887 | 0.824 / 0.809 / 0.797 | −5.4 / −8.0 / −9.0 |
| FinMA-7B (official prompt) | **0.937** | 0.791 [0.749, 0.831] | **−14.7** |

**What we can honestly claim:** *a laptop fine-tune of Gemma 4 E4B (0.890 on FinBen FPB, verified clean) matches FinMA-7B on fresh 2026 data (0.811 vs 0.791, n.s.). FinMA's 4.7-point FinBen lead shrinks by 6.8 points (CI +2.5 to +11.0) on unseen sentences, consistent with test-set memorisation. Answer key: AI-labelled, human spot-check pending.*

### Design
- **Sentences:** 360 from Nordic company press releases dated 30 June – 30 September 2026 (Cision feed), the same kind of source as PhraseBank.
  - 300 were sampled at random.
  - 60 were added by **negative-keyword enrichment**, because press releases are mostly good news (the random sample had only ~7 negatives). The keyword only chooses which sentences get labelled; the label comes from reading the sentence.
- **Filters:** no page headers, event logistics or fragments, and no near-copy of any FinBen FPB sentence.
- **Why nobody has seen them:**
  - FinMA (2023, LLaMA-1) can't have seen them.
  - Gemma 4's pretraining very likely ended before July 2026, so it probably hasn't either.
  - Crucially, nobody has seen their *labels*.
- **Gold labels: made by Claude (AI), at the user's request,** following PhraseBank's rule (investor view: up / down / no clear effect).
  - Claude labelled all 360 before any model was run on them, without seeing any model output.
  - **Caveat:** an LLM made the answer key for a comparison between LLMs. Its reading of sentiment could favour some models.
  - A human can measure the answer key's quality with a blind spot-check: `c2_label.py --spotcheck 40`.
- **Models:** FinBen's exact template and parser. Our frozen Part B model (3 seeds + ensemble), FinMA-7B with its official prompt, zero-shot E4B-it.

### Decision rule (fixed 2026-10-01, before any fresh result)
- lead(X) = FinMA wF1 − our ensemble's wF1 on test set X. On FinBen's test, lead = +0.047.
- **Memorisation supported:** the lead shrinks by ≥ 0.03 on fresh sentences, with a 95% bootstrap CI excluding 0.
- **Memorisation not supported:** FinMA still leads by ≥ 0.03 on fresh sentences, with McNemar p < 0.05.
- **Inconclusive:** anything else.

### Predictions (before any fresh result; numbers with tolerances)

| # | Prediction | Outcome |
|---|---|---|
| C1 | If spot-checked: human vs Claude agreement **75 ± 7%**, kappa **0.60 ± 0.10** | *not run yet* |
| C2 | Every model scores lower on fresh 2026 press releases than on FinBen's test (new style, new era): our ensemble **0.83 ± 0.05** | **Confirmed.** All drop; ensemble 0.811 |
| C3 | FinMA drops more than we do: fresh wF1 **0.80 ± 0.06** | **Confirmed.** 0.791; −14.7 vs our −7.9 |
| C4 | Verdict: **memorisation supported**, with FinMA's lead turning to **−0.01 ± 0.04** | **Confirmed.** Lead −0.021; shrink +0.068 [+0.025, +0.110] |
| C5 | *(dropped: with Claude as the only annotator there's no second label set; the human spot-check replaces it)* | — |

### Limitations
- 349 usable sentences give wide confidence intervals: a real difference under ~3 points may go undetected.
- The gold labels come from one AI annotator, not PhraseBank's 16 finance-trained annotators, and an LLM labelling a test for LLMs may favour some of them.
- 2026 press releases differ from PhraseBank's 2004–2008 news in style, so every model shifts domain. The comparison relies on FinMA's drop *relative to ours*.
- The negative class is enriched by keyword, so the class mix isn't the natural one. Results are also reported on the random 300 alone.

---

*Everything below is generated from `results/` by `report.py` (Part A: `a5_report.py`, Part B: `b5_report.py`).*

# Part A: contamination (generated by `a5_report.py`)

## 1. How much of FinBen's FPB test set is in its own training split?

| Detector | Test sentences | Share |
|---|---|---|
| exact (normalised text) | 1 | 0.1% |
| near-duplicate (cosine ≥ 0.9) | 23 | 2.4% |
| **contaminated = exact or near-duplicate** | 23 | 2.4% |
| shares a word 8-gram (loose; over-fires on boilerplate) | 88 | 9.1% |
| loose = any of the three (sensitivity check) | 94 | 9.7% |

**Clean subset: 947 of 970** (97.6%).
Contaminated sentences whose training copy has the same label: **78.3%** (5 carry a DIFFERENT label in train: memorising those costs points).
Test sentences with a near-copy in the validation split (watched, never trained on): 8.
Decontaminated training set: 3076 of 3100 train sentences kept.

| PhraseBank agreement level | Test sentences | Contaminated |
|---|---|---|
| 50Agree | 129 | 3.1% |
| 66Agree | 168 | 1.8% |
| 75Agree | 237 | 1.3% |
| AllAgree | 436 | 3.0% |

## 2. Scores: leaderboard (full) vs clean vs contaminated

Weighted F1 is FinBen's headline metric. QLoRA rows: mean ± std over seeds.

| System | Full (n=970): acc / **wF1** / macro-F1 / MCC / missing | Clean wF1 (n=947) | Contaminated wF1 (n=23) | ECE (full) |
|---|---|---|---|---|
| E4B-it zero-shot | **0.794** / 0.794 / 0.782 / 0.626 / 0.0% | 0.795 [0.770, 0.822] | 0.795 | 0.186 |
| QLoRA, FinBen train (n=3) | 0.867 / **0.867 ± 0.009** / 0.863 / 0.758 / 0.0% | 0.867 ± 0.009 | 0.890 ± 0.000 | 0.033 |
| QLoRA, decontaminated train (n=3) | 0.866 / **0.865 ± 0.003** / 0.859 / 0.754 / 0.0% | 0.864 ± 0.003 | 0.912 ± 0.018 | 0.044 |
| FinBERT † | 0.888 / **0.889** / 0.885 / 0.811 / 0.0% | 0.890 [0.869, 0.909] | 0.860 | 0.031 |

*Only 23 contaminated sentences: scores on that subset are too noisy to interpret.*

† FinBERT was fine-tuned on Financial PhraseBank itself, so every FPB test sentence may be in its training data; none of its scores is a clean measurement.

## 3. How much does contamination inflate the leaderboard number?

| Method | Ingredients (weighted F1) | Estimated inflation of full-test wF1 |
|---|---|---|
| (a) filter the test set | QLoRA full − clean = +0.000; zero-shot full − clean = -0.001 | **+0.001** |
| (b) clean the training set | trained-on-copies − decontaminated: contaminated subset -0.021, clean subset +0.003 (control, should be ≈ 0) | **-0.001** (= -0.021 × 2% of test) |

*If (a) and (b) agree, the estimate is trustworthy. If (b)'s clean-subset control is far from 0, the two training sets differ in more than the copies (e.g. size), and (b) is confounded.*

## 4. Significance (McNemar, same sentences)

| Comparison | Subset | Only first right / only second right | p |
|---|---|---|---|
| qlora_seed3407 vs zero-shot | clean | +88 / −13 | 7.5e-15 |
| qlora_seed3407-decontam vs zero-shot | clean | +87 / −23 | 6e-10 |
| qlora_seed3407 vs qlora_seed3407-decontam | contaminated | +0 / −0 | 1 |
| qlora_seed42 vs zero-shot | clean | +74 / −16 | 4.4e-10 |
| qlora_seed42-decontam vs zero-shot | clean | +88 / −19 | 8.5e-12 |
| qlora_seed42 vs qlora_seed42-decontam | contaminated | +0 / −1 | 1 |
| qlora_seed7 vs zero-shot | clean | +88 / −17 | 1e-12 |
| qlora_seed7-decontam vs zero-shot | clean | +88 / −23 | 3.8e-10 |
| qlora_seed7 vs qlora_seed7-decontam | contaminated | +0 / −1 | 1 |

## 5. Answer format

| System | Reply is exactly one label | FinBen 'missing' |
|---|---|---|
| e4b_zeroshot | 75.5% | 0.0% |
| finbert | 100.0% | 0.0% |
| qlora_seed3407-decontam | 100.0% | 0.0% |
| qlora_seed3407 | 100.0% | 0.0% |
| qlora_seed42-decontam | 100.0% | 0.0% |
| qlora_seed42 | 99.9% | 0.1% |
| qlora_seed7-decontam | 100.0% | 0.0% |
| qlora_seed7 | 100.0% | 0.0% |

## 6. Training runs

| Run | Train examples | Minutes | Val loss | Peak VRAM (GB) |
|---|---|---|---|---|
| fpb-seed3407 | 3100 | 7.7 | None | 4.67 |
| fpb-seed3407-decontam | 3076 | 7.6 | None | 4.84 |
| fpb-seed42 | 3100 | 7.6 | None | 4.93 |
| fpb-seed42-decontam | 3076 | 10.3 | None | 4.87 |
| fpb-seed7 | 3100 | 7.9 | None | 5.05 |
| fpb-seed7-decontam | 3076 | 9.8 | None | 4.69 |


# Part B: beating FinMA-7B (generated by `b5_report.py`)

## 1. Search on VALIDATION (776 sentences; the test set was not used)

| Epochs | LR | Rank | Seeds | wF1 generated | ± std | wF1 argmax | wF1 3-seed ensemble |
|---|---|---|---|---|---|---|---|
| 1.0 | 0.0002 | 16.0 | 1.0 | **0.8795** | — | 0.8806 | — |
| 2.0 | 0.0001 | 16.0 | 1.0 | **0.8882** | — | 0.8841 | — |
| 2.0 | 0.0002 | 16.0 | 3.0 | **0.8903** | 0.0088 | 0.8919 | 0.8898 |
| 2.0 | 0.0002 | 32.0 | 3.0 | **0.8992** | 0.0025 | 0.8987 | 0.9004 |
| 2.0 | 0.0004 | 16.0 | 3.0 | **0.8959** | 0.0014 | 0.8932 | 0.8993 |
| 2.0 | 0.0004 | 32.0 | 1.0 | **0.8901** | — | 0.8901 | — |
| 3.0 | 0.0002 | 16.0 | 1.0 | **0.8836** | — | 0.8836 | — |
| zero-shot | — | — | — | 0.8228 | | | |

Validation loss during training (logged twice per epoch):

| Run | Validation loss |
|---|---|
| e1_lr0.0002_r16_s3407 | 0.164 → 0.125 → 0.125 |
| e2_lr0.0001_r16_s3407 | 0.161 → 0.135 → 0.130 → 0.136 → 0.135 |
| e2_lr0.0002_r16_s3407 | 0.179 → 0.132 → 0.139 → 0.125 → 0.125 |
| e2_lr0.0002_r32_s3407 | 0.183 → 0.127 → 0.140 → 0.126 → 0.126 |
| e2_lr0.0004_r16_s3407 | 0.189 → 0.129 → 0.141 → 0.126 → 0.126 |
| e2_lr0.0004_r32_s3407 | 0.206 → 0.164 → 0.156 → 0.132 → 0.132 |
| e3_lr0.0002_r16_s3407 | 0.177 → 0.128 → 0.156 → 0.138 → 0.184 → 0.202 → 0.203 |

**Frozen choice:** 2 epoch(s), learning rate 0.0004, rank 16, decoding = generated, seed ensemble = yes. Rule: max validation weighted F1 over 3 seeds; ties within 0.003 -> cheaper config.

## 2. TEST results (970 sentences, scored once)

| System | wF1 [95% CI] | Accuracy | Macro-F1 | MCC | Clean-subset wF1 |
|---|---|---|---|---|---|
| Ours, seed 7 | **0.8871** [0.868, 0.908] | 0.8876 | 0.8845 | 0.7939 | 0.8875 (n=939) |
| Ours, seed 42 | **0.8886** [0.868, 0.907] | 0.8887 | 0.8857 | 0.7982 | 0.8901 (n=939) |
| Ours, seed 3407 | **0.8779** [0.858, 0.899] | 0.8794 | 0.8751 | 0.7783 | 0.8780 (n=939) |
| Ours, 3-seed ensemble | **0.8900** [0.870, 0.910] | 0.8907 | 0.8899 | 0.7997 | 0.8905 (n=939) |
| E4B-it zero-shot | **0.7944** [0.770, 0.820] | 0.7938 | 0.7816 | 0.6262 | 0.7938 (n=939) |
| FinMA-7B (our re-run) | **0.9371** [0.922, 0.952] | 0.9371 | 0.9310 | 0.8853 | 0.9382 (n=939) |
| FinMA-7B, raw prompt (sensitivity) | **0.9412** [0.926, 0.955] | 0.9412 | 0.9352 | 0.8926 | 0.9424 (n=939) |
| **Ours, mean ± std over seeds** | **0.8845 ± 0.0058** | | | | |
| FinMA-7B (published, FinBen paper) | 0.88 | | | | |

*FinMA (our re-run) uses its official prompt wrapper (`Human:` + query + `Assistant:`, `finma_prompt` in PIXIU's `model_prompt.py`); the raw-prompt row shows how much that wrapper matters. The verdict uses the official one.*

*Clean subset = test sentences with no near-copy in train or validation (the final model trained on both).*

## 3. Head to head with FinMA-7B on the same sentences

| Our system | Subset | wF1 difference (ours − FinMA) | Only ours right / only FinMA right | McNemar p |
|---|---|---|---|---|
| Ours, seed 7 | full | -0.0500 | +29 / −77 | 3.5e-06 |
| Ours, seed 7 | clean | -0.0506 | +29 / −76 | 5.1e-06 |
| Ours, seed 42 | full | -0.0485 | +28 / −75 | 4e-06 |
| Ours, seed 42 | clean | -0.0481 | +28 / −73 | 8.6e-06 |
| Ours, seed 3407 | full | -0.0592 | +31 / −87 | 2.5e-07 |
| Ours, seed 3407 | clean | -0.0602 | +31 / −86 | 3.7e-07 |
| Ours, 3-seed ensemble | full | -0.0471 | +32 / −77 | 1.9e-05 |
| Ours, 3-seed ensemble | clean | -0.0476 | +32 / −76 | 2.8e-05 |

## 4. Memorisation check: accuracy on training sentences vs test sentences

A model that trained on a sentence does better on it. Zero-shot trained on nothing, so its gap is the difficulty difference between the two sets (970 random train sentences vs the 970 test sentences).

| Model | Train-sample accuracy | Test accuracy | Gap (train − test) |
|---|---|---|---|
| E4B-it zero-shot (trained on nothing) | 0.8175 | 0.7938 | +0.0237 |
| FinMA-7B | 0.9649 | 0.9371 | +0.0278 |
| Ours, train-only twin (seed 3407) | 0.9660 | 0.8835 | +0.0825 |

*A fine-tuned model trained on train only should show a clear positive gap. A gap near zero for a fine-tuned model means it either memorised little, or also trained on the test sentences. Compare with our train-only twin.*

## 4b. Did FinMA see the test sentences? Accuracy by annotator agreement

PhraseBank records how many annotators agreed. On **50Agree** sentences the humans split, so a model that has not seen the gold label can't do much better than the label's own ambiguity allows. A model that memorised the label can.

| Model, split | AllAgree | 75Agree | 66Agree | **50Agree** |
|---|---|---|---|---|
| E4B-it zero-shot, test (never trained) | 0.938 | 0.806 | 0.649 | 0.473 |
| Ours, train-only twin, **test (unseen)** | 0.993 | 0.924 | 0.756 | 0.605 |
| Ours, train-only twin, **train (seen)** | 0.998 | 0.976 | 0.928 | 0.882 |
| **FinMA-7B, test** | 0.993 | 0.949 | 0.887 | 0.791 |
| FinMA-7B, train | 0.993 | 0.960 | 0.941 | 0.906 |

On the sentences zero-shot gets wrong (the hard ones): our twin scores **0.555** on unseen test sentences (n=200) and **0.853** on train sentences it trained on (n=177); FinMA scores **0.835 on the test ones**.

The 5 test sentences whose train near-copy carries a DIFFERENT label: our twin follows the test label 1/5 (train label 4); FinMA follows the test label 2/5 (train label 3). Too few to decide anything.

## 5. Parity with FinBen's official scoring code

Re-scoring every saved test output with PIXIU's `flare.py` code (copied verbatim, `b4_harness_parity.py`): **identical predictions and metrics** for 7 systems. The remaining difference from the official harness is only how the model is loaded (4-bit with the per-layer table in CPU RAM), not how it's prompted, decoded or scored.

## Verdict (rule fixed before the test run)

**3-seed ensemble: behind**


# Part C: a fresh test set FinMA can't have seen (generated by `c4_report.py`)

360 sentences from Nordic company press releases dated 2026-06-30 to 2026-09-30 (300 random, 60 added by negative-keyword enrichment). Labelled: **360/360** **by Claude (AI), at the user's request -- not human gold labels**.

> **Caveat:** the answer key was made by an LLM, and the systems being compared are LLMs. An LLM's reading of sentiment could favour some models over others. See section 4 for how well a human agrees with it.

Gold labels used: 349 sentences (negative 27, neutral 210, positive 112); 11 excluded or unlabelled.

## 1. Scores: FinBen test (970) vs fresh 2026 sentences

| System | FinBen test wF1 | Fresh wF1 [95% CI] | Fresh accuracy | Fresh macro-F1 | Drop (fresh − FinBen) |
|---|---|---|---|---|---|
| E4B-it zero-shot | 0.7944 | **0.7164** [0.669, 0.765] | 0.7135 | 0.7055 | -0.0781 |
| Ours, seed 3407 | 0.8779 | **0.8237** [0.782, 0.862] | 0.8252 | 0.7910 | -0.0542 |
| Ours, seed 42 | 0.8886 | **0.8089** [0.765, 0.848] | 0.8080 | 0.8019 | -0.0797 |
| Ours, seed 7 | 0.8871 | **0.7969** [0.752, 0.837] | 0.7966 | 0.7755 | -0.0902 |
| Ours, 3-seed ensemble | 0.8900 | **0.8114** [0.768, 0.851] | 0.8109 | 0.7995 | -0.0786 |
| FinMA-7B (official prompt) | 0.9371 | **0.7905** [0.749, 0.831] | 0.7880 | 0.7834 | -0.1466 |

## 2. The decision (rule fixed before the fresh results existed)

- FinMA's lead over our ensemble on the FinBen test: **+0.0471**
- FinMA's lead on the fresh sentences: **-0.0209** (McNemar p = 0.3; only FinMA right 19, only ours right 27)
- Shrink: **+0.0680**, 95% CI [+0.025, +0.110]

**Verdict: memorisation SUPPORTED.**

*Rule: supported if the lead shrinks by ≥ 0.03 with a CI excluding 0; not supported if FinMA still leads by ≥ 0.03 with p < 0.05; otherwise inconclusive.*

## 3. Breakdowns

| System | wF1, random sample | wF1, negative-keyword enrichment | Negative-class F1 |
|---|---|---|---|
| E4B-it zero-shot | 0.7328 (n=291) | 0.6416 (n=58) | 0.6939 |
| Ours, seed 3407 | 0.8486 (n=291) | 0.6903 (n=58) | 0.7347 |
| Ours, seed 42 | 0.8184 (n=291) | 0.7476 (n=58) | 0.8163 |
| Ours, seed 7 | 0.8084 (n=291) | 0.7272 (n=58) | 0.7500 |
| Ours, 3-seed ensemble | 0.8231 (n=291) | 0.7438 (n=58) | 0.8000 |
| FinMA-7B (official prompt) | 0.7957 (n=291) | 0.7593 (n=58) | 0.7843 |

## 4. Label reliability

No human check of the AI answer key yet. Run `c2_label.py --spotcheck 40` (~8 min) to measure it.

