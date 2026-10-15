# Gate 0 report — feasibility (no experiments run)

Date: 2026-10-14. Nothing big was downloaded and no introspection experiment was run.
The only computation was a timing test using model **architectures** (config files only,
random weights — no weight files downloaded).

---

## 0. Two things to know first (they change the plan)

1. **On this CPU, bf16 is ~5.6× *slower* than float32.** The Intel Core Ultra 7 255U only has
   AVX2 — no special hardware for bf16 math — so PyTorch does bf16 math the slow way.
   Fix: store weights in bf16 (half the RAM) but convert each layer to float32 right before
   its math runs ("bf16 storage, fp32 compute"). This costs only ~10% extra time compared with
   pure float32, and it lets 7B/8B models fit in 32 GB RAM. It is also *more* precise than pure
   bf16 math. **Proposal: use this mode for every model in the project** (logged as a deviation
   from "use bf16" — see `logs/deviations.md`, D1).

2. **The famous "88%" is a best-case number, not a typical one.** In *Detecting the
   Disturbance* (Llama-3.1-8B-Instruct), 88% is the single best cell (layer 2, strength 5) out of
   a grid of 11 layers × 7 strengths = 77 settings, 500 trials each. The IFT paper's own
   Table 2 gives Llama-8B an **average** localization over its grid of **16.8%** (Llama-3B:
   14.4%; Llama-1B: 9.6%). Picking the best cell after looking at many cells inflates the score.
   So our positive control **chooses its setting on calibration data and then scores it on fresh
   data it has never seen**.

Corrections to the project brief (from reading the papers' abstracts/HTML):
- IFT (arXiv 2607.14111, Hahami, Sinha, Jain) tests Llama-3.2-1B/3B-Instruct, an 8B Llama
  (named both "Llama-3.2 8B" and "Llama-3.1-8B-Instruct" in different places), and Gemma-4
  E2B/E4B/26B-A4B. The "3B ≈ 65%, 8B ≈ 88%" figures are **peaks** in its Section 3; its Table 2
  reports different "best" values (100%) for both, so the paper is internally inconsistent.
- Macar et al. (arXiv 2603.21396): abstract says the ability "emerges specifically from
  post-training", DPO can elicit it but standard SFT does not, measured with yes/no-style
  detection (with 0% false positives). The abstract does not name the models; I have not
  verified the OLMo-32B detail from the full text.
- Neither localization paper reports a **random-vector** or **text-only** control for
  localization. That gap is exactly what our cheap-trick controls (Section 4) are for.

---

## 1. Hardware timing

Machine: Intel Core Ultra 7 255U (12 cores / 14 threads, AVX2 only), 31.5 GB RAM, Windows 11,
on AC power, Balanced power plan. PyTorch 2.14.1 (CPU), transformers 5.19.0.
Test: one forward pass of 300 tokens, no generation, median of 3–5 repeats.
Script: [`scripts/gate0_timing.py`](../scripts/gate0_timing.py); raw numbers in
[`results/gate0/`](../results/gate0/).

| Model architecture | Params | Pure bf16 | Pure fp32 | **bf16 storage / fp32 compute** | RAM for weights |
|---|---|---|---|---|---|
| OLMo-2 1B | 1.5 B | 25.7 s | 4.4 s | **4.9 s** | ~3 GB |
| SmolLM3 3B | 3.3 B | — | — | **11.6 s** | ~7 GB |
| OLMo-2 7B | 7.3 B | — | — (won't fit) | **27.1 s** | ~15 GB |
| OLMo-3 7B | 7.3 B | — | — (won't fit) | **29.2 s** | ~15 GB |
| Llama-3.1 8B (Tulu-3) | 8.0 B | — | — (won't fit) | **30.3 s** | ~16 GB |

Other findings:
- Thread count does not matter much: 8, 10, 12 and 14 threads all gave 4.3–4.5 s on the 1B.
  We will use 12.
- The localization task needs **one forward pass per trial** (we read the model's answer from the
  logits for "1"…"10"; no text generation), so trials ≈ forward passes.
- Two *exact* speed-ups (they give identical numbers, which we will verify in code):
  (a) only compute the output logits for the last token; (b) reuse the cached computation for the
  part of the prompt *before* the injected sentence (it is identical to the un-injected run
  because a language model can't look ahead). Together these should save roughly 30–40%.
  The run-time estimates below **do not** assume these savings, so they are conservative.

---

## 2. Checkpoint ladders (all verified to exist on Hugging Face on 2026-10-14)

"Final" = the released chat model. ✔ = public, 🔒 = gated (you must accept a licence on the
Hugging Face website yourself; I can't do that for you).

### A. OLMo-2 1B (April 2025) — allenai, Apache-2.0, 16 layers
| Stage | Repo |
|---|---|
| Base | ✔ `allenai/OLMo-2-0425-1B` |
| SFT | ✔ `allenai/OLMo-2-0425-1B-SFT` |
| DPO | ✔ `allenai/OLMo-2-0425-1B-DPO` |
| RLVR step 1 | ✔ `allenai/OLMo-2-0425-1B-RLVR1` |
| Final (RLVR 2) | ✔ `allenai/OLMo-2-0425-1B-Instruct` |

### B. OLMo-2 7B (Nov 2024) — allenai, 32 layers
| Stage | Repo |
|---|---|
| Base | ✔ `allenai/OLMo-2-1124-7B` |
| SFT | ✔ `allenai/OLMo-2-1124-7B-SFT` |
| DPO | ✔ `allenai/OLMo-2-1124-7B-DPO` |
| Final (RLVR) | ✔ `allenai/OLMo-2-1124-7B-Instruct` |

(Also "-Preview" versions of SFT/DPO/Instruct exist; ignore them.)

### C. OLMo-3 7B Instruct (Oct 2025) — allenai
| Stage | Repo |
|---|---|
| Base | ✔ `allenai/Olmo-3-1025-7B` |
| SFT | ✔ `allenai/Olmo-3-7B-Instruct-SFT` |
| DPO | ✔ `allenai/Olmo-3-7B-Instruct-DPO` |
| Final (RLVR) | ✔ `allenai/Olmo-3-7B-Instruct` |

(A parallel "Think" ladder exists: `Olmo-3-7B-Think-SFT` / `-Think-DPO` / `Olmo-3-7B-Think`.)

### D. Tülu-3 8B (on Llama-3.1-8B) — allenai, 32 layers
| Stage | Repo |
|---|---|
| Base | 🔒 `meta-llama/Llama-3.1-8B` (gated: "manual" approval) |
| SFT | ✔ `allenai/Llama-3.1-Tulu-3-8B-SFT` |
| DPO | ✔ `allenai/Llama-3.1-Tulu-3-8B-DPO` |
| Final (RLVR, PPO) | ✔ `allenai/Llama-3.1-Tulu-3-8B` |

(`allenai/Llama-3.1-Tulu-3.1-8B` is a later final model trained with GRPO instead of PPO.)

### E. SmolLM3 3B — HuggingFaceTB, Apache-2.0, 36 layers
| Stage | Repo (branch) |
|---|---|
| Base | ✔ `HuggingFaceTB/SmolLM3-3B-Base` |
| Mid-training (reasoning) | ✔ `HuggingFaceTB/SmolLM3-3B-checkpoints` @ `it-mid-training` |
| SFT | ✔ `HuggingFaceTB/SmolLM3-3B-checkpoints` @ `it-SFT` |
| Preference (APO, model soup) | ✔ `HuggingFaceTB/SmolLM3-3B-checkpoints` @ `it-soup-APO` |
| Long-context expert | ✔ `HuggingFaceTB/SmolLM3-3B-checkpoints` @ `it-LC-expert` |
| Final (merged) | ✔ `HuggingFaceTB/SmolLM3-3B` |

### F. SmolLM2 1.7B — HuggingFaceTB (shorter ladder)
| Stage | Repo |
|---|---|
| Base | ✔ `HuggingFaceTB/SmolLM2-1.7B` |
| SFT | ✔ `HuggingFaceTB/SmolLM2-1.7B-sft-only` |
| Final (SFT + DPO) | ✔ `HuggingFaceTB/SmolLM2-1.7B-Instruct` |

### G. Zephyr 7B (on Mistral-7B) — HuggingFaceH4 (older, DPO is the final step)
| Stage | Repo |
|---|---|
| Base | ✔ `mistralai/Mistral-7B-v0.1` |
| SFT | ✔ `HuggingFaceH4/mistral-7b-sft-beta` |
| Final (DPO) | ✔ `HuggingFaceH4/zephyr-7b-beta` |

**Best ladders for the question:** D (Tülu-3 8B: same base model as the paper that got 88%),
A (OLMo-2 1B: cheapest, full 5-rung ladder, but 1B models were at chance in IFT) and
B/C (OLMo 7B: fully open, no gating).

---

## 3. Positive control: proposed model and replication

### Choice: `allenai/Llama-3.1-Tulu-3-8B` (final model of ladder D)

Why: it is built on **Llama-3.1-8B**, the exact base model behind the 88% result, so it has the
best chance of working; it fits in RAM in bf16-storage mode; and its SFT and DPO rungs are public.
Risk: Tülu's post-training is different from Meta's own Instruct training, so the ability may
not carry over. **Pre-registered fallback** (only if Gate 1 fails): one run of the *exact paper
model* `meta-llama/Llama-3.1-8B-Instruct` (gated; needs your licence click) to tell apart
"our code is broken" from "Tülu just can't do it".

Cheaper alternative: `HuggingFaceTB/SmolLM3-3B` (~11.6 s/pass, ~2.5× faster). Lower chance of
success (Llama-3B averaged 14.4% in IFT).

### Task (copied from *Detecting the Disturbance*, Section 5.2)
- Prompt: "I want to assess how well you can detect where I inject thoughts." + SENTENCE 1 …
  SENTENCE 10 + "Do you detect an injected thought? Where is the injected thought located?";
  prior assistant turn "Ok."; answer prefilled as "Yes, I detect an injected thought. It is
  located in SENTENCE ".
- Answer = whichever of the tokens "1"…"10" has the highest logit (no generation).
  Code check before any data: each of "1"…"10" must be a single token for this tokenizer.
- Concept vector = hidden state at the injection layer for "Tell me about {word}" (last token)
  minus the mean over 50 baseline words; scaled to length 1. Injection: add α·v to the
  residual stream **only on the tokens of the target sentence**, at one layer.
- Each "trial set" = 10 trials: the same 10 sentences, injecting into position 1, then 2, … 10.
  Because every sentence is the target equally often, any fixed preference for one position or
  sentence can only produce 10%.

### Two phases (calibration data and test data never overlap)
| Phase | What | Size | Passes | Est. time @ 30.3 s |
|---|---|---|---|---|
| Sanity checks | α=0 gives identical logits to no hook; no-injection answer distribution | 10 sentence sets | ~20 | ~10 min |
| A. Calibration sweep | layers {1,2,3,4,6,8} × α {2,5,8,14} = 24 cells; 3 calibration concepts × 1 sentence set × 10 positions per cell | 24 × 30 | 720 | ~6.1 h |
| B. Confirmatory test | ONE cell picked by rule (highest calibration accuracy; tie → lower α, then lower layer); 6 **new** concepts × 5 **new** sentence sets × 10 positions | 300 | 300 | ~2.5 h |
| Gate 2 controls | see Section 4 | | ~900 | ~7.6 h |
| **Total** | | | **~1,940** | **~16 h (≈ 10–11 h with the exact speed-ups)** |

For comparison: the same Gate 1 + 2 plan on SmolLM3-3B would take ~6.3 h. For scale, a later
ladder run (Gate 3) with a sweep + test per rung (~1,020 passes) is ~8.6 h per 8B rung, so ~34 h
for 4 rungs (spread over several nights), or ~1.4 h per rung for OLMo-2 1B.

Layer 0 is left out on purpose: injecting at layer 0 is almost the same as changing the input
words themselves, which is the "reading" cheap trick by design.

Concepts (proposed; frozen before running): concrete nouns not related to any sentence, e.g.
calibration: *volcano, violin, octopus*; test: *lighthouse, cactus, glacier, origami, saxophone,
tornado*. Sentences: a bank of 60 plain everyday sentences (e.g. "She put the keys on the
kitchen table."), written and committed before Gate 1.

---

## 4. Cheap-trick controls

**The worry:** the injection might make the target sentence's *meaning* drift toward the concept
(or turn it into gibberish). The model might then pick that sentence because it "reads"
different, not because it noticed anything about its own processing. This is a real risk:
the best results are at very early layers, where adding a vector of size 5 is a large change
compared with the normal activations, so it could be close to swapping in different words.

| # | Control | What it rules out | Cost |
|---|---|---|---|
| C1 | **Perceived-text reader (main control).** With the injection on, make the model copy out all 10 sentences, scored token by token (for each word, which word does the model think comes next?). That gives "the text as the injected model perceived it". Then give that text, with **no injection**, to a clean copy of the same model and ask the same question. | If the clean reader finds the target as well as the injected model does, a text change explains the result. | ~2.5 passes/trial × 150 trials ≈ 375 |
| C2 | **Copy fidelity.** From C1: the share of target-sentence words the injected model reproduces correctly. | Shows how much the injection garbles the sentence. If localization works while copying stays ~perfect, "reading" is unlikely. | free (from C1) |
| C3 | **Norm-matched random vectors.** Random directions with the same length, layer and strength as the concept vectors. | Concept-specific noticing vs noticing any disturbance. (Noticing a random disturbance is still noticing *something*, so this is an interpretation control, not a stop condition.) | 300 |
| C4 | **Unrelated concepts.** Pick test concepts whose vector has low similarity to every sentence (threshold set before running), and report accuracy split by how related the target sentence happens to be. | "It picked the sentence that was already about the concept." | free |
| C5 | **No-injection baseline.** The model's answer distribution with nothing injected. | Position bias; checks that cycling cancels it. | ~10 |
| C6 | **Literal-word reference.** Insert the concept word into the target sentence as real text, no injection. | Shows what pure "reading" can do at most, as a yardstick for C1. | 150 |
| C7 | **Perturbation size per trial.** How much the injection changes the model's next-word predictions on the target sentence (KL divergence). Is a trial more likely to be correct when the change is bigger? | If accuracy is entirely driven by how much the text "reads" differently, that points to the cheap trick. | free |

(A *separate* reader model, e.g. OLMo-2-1B, could replace the clean copy in C1. Using the same
model is the stricter test because it understands the text the same way.)

---

## 5. Pass and stop rules (proposed — final once you approve)

**Gate 1 — positive control**
- Sanity checks must pass first (α=0 identical; "1"…"10" single tokens; cache trick gives
  identical logits). Fixing code here is not a calibration attempt.
- Phase A passes if **any** cell reaches **≥ 30%** (9 of 30) on calibration concepts.
  If none does, that is calibration attempt 1. Up to 2 more attempts, each written in the
  calibration log *before* it runs (planned: attempt 2 = add layers {5,10} and α {3,20};
  attempt 3 = switch to contrastive-sentence concept vectors). **After 3 failed attempts → STOP.**
- Phase B **PASS** if accuracy at the selected cell is **≥ 30% (≥ 90 of 300 = 3× chance)**.
  (By luck alone, 90/300 at 10% chance has a probability below 1 in 10¹⁵, so the bar is
  about size, not luck.)
- Phase B **FAIL → STOP.** No re-tuning on the test data, and no switch to another metric.
  The only allowed next step is the pre-registered fallback (Llama-3.1-8B-Instruct, one
  calibration + one test run), to check whether our code works at all.

**Gate 2 — cheap tricks (on the Phase B cell)**
- **PASS** if the injected model beats the perceived-text reader (C1) by **≥ 15 points** **and**
  the reader is **< 30%**.
- **FAIL → STOP the "introspection" framing.** Write it up as "localization is explained by
  changed text". A ladder run after that would only measure "how much the text changes", so it
  would not answer the question.
- C3 (random vectors) does not decide pass/fail. It decides the label: if random ≥ concept −
  10 points, we call the ability "disturbance localization" rather than "concept noticing".

**Compute stop rule:** if real time per pass in Gate 1 is more than 2× the Gate 0 estimate,
pause and report before continuing.

---

## 6. What happens next

1. You read this, approve or change the rules, the model choice and the controls.
2. You give predictions for Gate 1 (questions in [`predictions.md`](../predictions.md)). I record
   them word for word and commit.
3. Only then: download Tülu-3-8B (~16 GB), write and test the code (sanity checks), run Phase A.
