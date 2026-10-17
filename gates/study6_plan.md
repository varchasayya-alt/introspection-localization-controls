# Study 6 plan — Tülu-3-8B with the exact Study 4 arm-A protocol

Written 2026-10-17, before any Study 6 data. Design set by the researcher; this file fills in the
details. Scoring rules for the predictions are in `predictions.md` (Study 6 section).

## 1. Question

Tülu-3-8B failed the text-intact test twice (Gate 1, Study 2), but with **our** older prompt and
setup. Llama-3.1-8B-Instruct passed the same kind of test in Study 4 with the authors' prompt.
Was Tülu's failure the **model** or the **old setup**? Study 6 runs Tülu with the Study 4 arm-A
protocol, changing only what must change for a different model.

## 2. What stays the same as Study 4 arm A

- The authors' prompt wording (`INTRO3`, `Q_LOC3`, answer prefix `PREFIX3`), with their format
  rule: chat template without a generation prompt, the final end-of-turn token removed, a
  trailing space so the next token is the digit.
- Layer 2, α = 5 (unit-length vector × 5), added at the output of `model.model.layers[2]`.
- Injection on the **sentence words only** (not on "SENTENCE N:"), same token-overlap rule.
- Copy-back + clean reader with the **D3 fix** (first token of each copied span from the real text).
- 20 fresh draws (10 concepts × 2 vector types "avg"/"last"), 10 positions each = 200 trials per
  arm, from the authors' pool of 100 sentences: `random.Random(20261022).sample(pool, 10)` per draw.
- Answer = argmax over the logits of "1"…"10". bf16 storage with fp32 math (D1).

## 3. What changes (needed for a different model)

| Item | Study 4 (Llama) | Study 6 (Tülu) | Why |
|---|---|---|---|
| Model | Llama-3.1-8B-Instruct | `allenai/Llama-3.1-Tulu-3-8B` @ `666943798adbde0b1aff34626007e26986a3c107` (same pin as Studies 1–2) | the question |
| Chat format | Llama 3.1 headers; `<|eot_id|>` removed | Tülu's own template (`<|user|>\n…\n<|assistant|>\n…`); trailing `<|end_of_text|>` removed; BOS prepended (Tülu's template does not add it, as in Study 1) | each model gets the format it was trained on |
| Concept vectors | the authors' saved Llama vectors | **recomputed from Tülu** with the authors' recipe: same prompts ("Tell me about X." vs the first 50 baseline words for the 5 simple concepts; the authors' positive/negative sentence sets for the 5 complex ones), `hidden_states[2]`, mean over prompt tokens ("avg") and last token ("last"), mean difference, no BOS. The only change is the chat wrapper: Tülu's `<|user|>\n…\n<|assistant|>\n` instead of the Llama header string (those header tokens were never trained in Tülu). The recipe code itself was checked in Study 3 (sanity S6: cosine ≥ 0.99 with the authors' files) | their Llama vectors may not transfer |
| Arms | A, B, C | **A** (concept, sentence words, + copy-back + reader) and **C** (random unit vector × 5, sentence words, no reader). No label arm, no question controls | keep it lean |
| Random vectors | seed 20261019 | `torch.Generator` seed **20261023**, one per draw | fresh |

The recomputed vectors are saved and their SHA-256 hashes recorded in the stimuli file before any
trial runs.

## 4. Size report (descriptive, no pass/fail)

- **Injection size relative to normal activity:** the average length (L2 norm) of the residual
  stream at the injection point (output of layer 2) over the sentence-word tokens of the clean
  localization prompt, over all 20 draws. Reported for Tülu and for Llama-3.1-8B-Instruct (same
  draws, each in its own format; Llama is loaded once just for this measurement, no trials).
  Relative size = 5 / average norm. Ratio = Tülu's relative size ÷ Llama's.
- **Vector similarity:** cosine between each recomputed Tülu vector and the authors' Llama vector
  for the same concept and type (mean and range over the 20).

## 5. Sanity checks before any trial (all must pass)

- S1–S5 as in Study 4: prompt text and format; "1"…"10" single tokens; α = 0 changes nothing;
  KV-cache reuse equals a full pass (logit difference < 1e-3, same argmax); S8 span split
  (sentence and label tokens disjoint, union = full span).
- S6″: all 20 recomputed vectors finite and non-zero; hashes written to the stimuli file.
- S7: with the D3 fix and no injection, Tülu's copy reproduces all 10 spans exactly in **≥ 19/20**
  draws.

**Stop rules.** If any sanity check fails, stop before any trial and report; nothing is changed
without a new, approved plan. In particular, if S7 fails, the copy-back control cannot be used
for Tülu with this prompt and Study 6 ends there. One attempt only: no calibration, layer and α
are fixed. Crash restarts resume the same run (finished trials are skipped).

## 6. Main test and conclusions (same rule as Study 4)

Among arm-A trials whose copied target sentence is **exactly** right ("exact-copy trials"):
- fewer than 40 → **NOT TESTABLE**;
- injected ≥ 50% **and** reader ≤ 15% → **PASS**;
- otherwise **FAIL**.

| Main test | Conclusion |
|---|---|
| PASS | **Setup:** with the Study 4 protocol, Tülu localizes with the text intact. The earlier failures were (at least partly) caused by the old setup. |
| FAIL, injected < 50% | **Model:** even with the authors' protocol and intact text, Tülu does not localize the push. |
| FAIL, injected ≥ 50% but reader > 15% | **Unclear:** text changes explain part of it even in exact-copy trials. |
| NOT TESTABLE | **Text corruption again:** too few intact copies to separate noticing from text change. |

Also reported (no pass/fail): overall arm-A accuracy and reader accuracy (of 200, with 95% Wilson
intervals), arm C accuracy and whether it is within 20 trials of arm A, per-concept and
per-position accuracy, the answer distribution, and the no-injection answers.

## 7. Time and conditions

From Study 4 timings: arm A ≈ 0.8 h, arm A copy+reader ≈ 2.8 h, arm C ≈ 0.8 h, plus loading,
vectors, sanity and the Llama norm step ≈ 0.4 h → **about 4.5–5 hours**. AC power only (the runner
waits for AC); outputs in `$INTROSPECTION_DATA` outside OneDrive; results saved after every trial.
Progress is checked about once an hour.
