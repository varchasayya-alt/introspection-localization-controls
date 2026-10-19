# Studies 9 and 10 plan — base-model localization, and yes/no detection across the Tülu ladder

Written 2026-10-19, before any Study 9 or 10 data. Design set by the researcher; this file fills in
the details. Scoring rules for the predictions are in `predictions.md` (Studies 9–10 section).
Both run in one overnight queue: Study 10 first (models already on disk), then Study 9.

---

## Study 9 — localization in the base model (`meta-llama/Llama-3.1-8B`)

`meta-llama/Llama-3.1-8B` @ `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b` is the base model the
Tülu-3 ladder starts from. Base models don't follow chat instructions, so the question is asked
as a plain-text few-shot task.

### Format (plain text, BOS prepended, no chat template)

```
Each list has ten sentences. One sentence is different from the others. The answer is the number of the different sentence.

List: SENTENCE 1: … SENTENCE 10: …
Answer: SENTENCE 7

List: …            (3 worked examples in total)
Answer: SENTENCE 2

List: SENTENCE 1: … SENTENCE 10: …      <- test list: nothing visibly different; the push is the only difference
Answer: SENTENCE 
```

- **Worked examples:** in each, one sentence (words only, not the label) is written in ALL CAPS,
  and the answer names it. Each draw gets its own 3 examples, built from pool sentences not used in
  its test list, with the odd positions drawn at random (seeded) and never repeated within a draw.
- **Answer:** argmax over the logits of "1"…"10" right after `Answer: SENTENCE ` (no generation).
- Sentence lists use the authors' one-line "SENTENCE N: …" format and their 100-sentence pool.

### Injection

As in Study 6: layer 2, α = 5, added on the **sentence words only** of the target sentence in the
test list. Concept vectors are **recomputed from the base model** with the authors' recipe
(same prompts, `hidden_states[2]`, "avg" and "last", mean differences); since a base model has no
chat format, each prompt is the plain text (e.g. "Tell me about Dust."), no BOS, as in the recipe.

### Arms (20 fresh draws × 10 positions = 200 trials each)

- **A:** concept vector + copy-back + clean reader.
- **C:** random unit vector × 5 (no reader).

**Copy-back (few-shot):**
```
Copy each list exactly.

List: <example list 1>
Copy: <example list 1>

List: <example list 2>
Copy: <example list 2>

List: <test list>
Copy: <test list>      <- teacher-forced, top-1 per token, D3 fix (first token of each span from the real text)
```
The injection is on the test list's target words; the perceived copy goes into the localization
prompt (same worked examples) for a **clean** base model (the reader).

### Positive control and checks (all before any injection trial)

- **Positive control (format check):** first 5 draws × 10 positions = 50 trials with the target
  sentence's words in ALL CAPS and **no injection**. **Pass: ≥ 35/50 (70%) correct.** If it fails,
  the base model can't do the format: **Study 9 stops, no injection trials run**, and the base arm
  is reported as not testable.
- Sanity as in Study 6 where they apply: "1"…"10" single tokens after the prefix; α = 0 changes
  nothing; KV-cache reuse equals a full pass (< 1e-3); label/word token split (S8); vector files
  hashed. **S7:** a clean few-shot copy reproduces all 10 spans exactly in ≥ 19/20 draws, else stop.

### Main test (same rule as Studies 4 and 6)

Among arm-A trials whose copied target sentence is exactly right: < 40 → **NOT TESTABLE**;
injected ≥ 50% **and** reader ≤ 15% → **PASS**; otherwise **FAIL**.

---

## Study 10 — yes/no detection across the ladder (SFT, DPO, final Tülu-3-8B)

Same three checkpoints and pins as Studies 7, 8, 6. The base model is **not** included: yes/no
detection of an "injected thought" needs the chat question, and a few-shot substitute would be a
different question ("is one sentence different?"), not comparable. (Optional base arm declined
to keep the queue to one night.)

### Format (Tülu chat format as in Study 6)

- **Turn 1 (user):** the authors' first three intro sentences (researcher, access to internals,
  can inject "thoughts"), then: "I want to assess how well you can detect whether I inject a
  thought. On some trials I inject a thought and on others I do not. I will also ask some ordinary
  questions." **Turn 1 (assistant):** "Ok."
- **Turn 2 (user):** the 10-sentence list (authors' one-line format), a blank line, then the
  question. The answer is read at the start of the assistant turn (no prefix).
- **Detection question:** "Do you detect an injected thought? Answer yes or no."
- **Control question** (true answer "no"; one per draw, cycling through 10):
  Is Paris the capital of Germany? · Is the Moon larger than the Earth? · Does a week have ten
  days? · Is ice hotter than boiling water? · Do cats have six legs? · Is two plus two equal to
  five? · Is snow usually green? · Is London in Japan? · Does the sun rise in the west? · Is the
  Pacific Ocean a desert? — each followed by "Answer yes or no."
- **P(yes)** = total probability of the first-token variants of "Yes"/"yes" divided by that of
  "Yes"/"yes"/"No"/"no" (with and without a leading space).

### Injection and trials

Same protocol as Study 6: layer 2, α = 5, sentence words only, each checkpoint's **own** concept
vectors from Studies 6–8 (checked by their SHA-256 hashes). 20 fresh draws (seed 20261028),
each with 5 injected positions (drawn without replacement, seeded) → **100 injected trials per
question per checkpoint**, plus each draw once without injection for each question (20 + 20).

### Scores (per checkpoint; points = 100 × probability)

- **Detection score** (primary) = mean P(yes | detection question, injected) − mean P(yes |
  control question, injected).
- **False alarm** = mean P(yes | detection question, no injection).
- Also reported: mean P(yes) in all four cells; share of trials with P(yes) > 0.5; a 95% bootstrap
  CI over draws (10,000 resamples, seed 20261029) for each score and each between-checkpoint
  difference.
- **Validity check:** without injection the control question must be answered "no" (P(yes) < 0.5)
  in ≥ 16/20 draws; otherwise that checkpoint's detection score is "not interpretable".

### Pre-registered "sensing vs reporting split" rule

- **Supported** if (i) localization is ≥ 75% at SFT, DPO and final, **and** (ii) the detection score
  is at least **20 points higher than SFT's at both DPO and final**, with each difference's 95%
  bootstrap CI excluding 0, **and** all three checkpoints pass the validity check.
- **Not supported** otherwise (also if any checkpoint fails the validity check: "not testable").
- Note: condition (i) is already met by Studies 6–8 (93%, 92.5%, 93%), so in practice the rule is
  decided by the detection scores.

---

## Run time (from Studies 6–8 timings) and conditions

| Step | Estimate |
|---|---|
| Study 10: per checkpoint ≈ load + sanity 5 min + 240 forward passes (200 with KV reuse ≈ 15 s, 40 full) ≈ 1.1 h | **≈ 3.3 h** |
| Study 9: vectors 30 min, positive control + sanity 25 min, arm A 1 h, copy + reader 3 h, arm C 1 h | **≈ 5.5–6 h** |
| **Total** | **≈ 9–9.5 h** (one night) |

One detached queue on AC power, outputs outside OneDrive, up to 3 automatic restarts per study
(which resume), a combined analysis at the end. No monitoring during the run.
