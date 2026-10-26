# Study 17 pre-registration — the answer-position fingerprint in a model that reports injections

Written 2026-10-26, **before any Study 17 data** (no real run of any kind; code is tested only on
Gemma-3-1B-it dry runs). Design set by the researcher; this file fills in the details. The two
design choices left open in the draft (judge, primary fingerprint) were decided by the researcher on
2026-10-26, and the predictions (§11) were given then; both are committed here, before any real
data.

## 0. Plain-language version

In Studies 15 and 16, the pattern an injection leaves at the moment a Llama model answers, pushed in
by hand, made the model lean toward "yes" in general, more on unrelated questions ("Is Paris the
capital of Germany?") than on "Do you detect an injected thought?". But those Llama models never
report injections in the first place. Maybe that is why their pattern carries no detection signal.

Study 17 tests a model that **does** report injections, according to Macar et al. (2026):
Gemma-3-27B-it. Its smaller sibling, Gemma-3-4B-it, serves as the same-family comparison.
- **Gate 1:** we first check, with Macar et al.'s own method, that 27B really reports injections
  and whether 4B does.
- **Fingerprint test:** we then repeat the Study 16 test at matched push sizes.

The key question: **does the pattern push "yes" more on the detection question than on unrelated
questions in the model that reports, while it doesn't in the models that don't?**

## 1. Models and compute

| model | pin | role | where | numerics |
|---|---|---|---|---|
| `google/gemma-3-27b-it` | `005ad3404e59d6023443cb575daa05336842228a` | reporting model (per Macar et al.) | rented GPU (80 GB) | bf16 weights and bf16 compute |
| `google/gemma-3-4b-it` | `093f9f388b31de276ce2de164bdc2081324b9767` | same family, smaller | laptop CPU | bf16 storage, fp32 math (D1) |
| `google/gemma-3-1b-it` | `dcc83ea841ab6100d6b47a070329e1ba4cf78752` | **code testing only (dry runs); no results** | laptop CPU | D1 |

**Text-only path:** 4B and 27B are image+text checkpoints. They are loaded as published, but only
text is ever passed in. The decoder blocks are the language model's (`model.model.language_model.layers`,
as in Macar et al.'s code); 1B is text-only. "Block ℓ" = the output of decoder block ℓ (0-based),
the convention of both our earlier studies and Macar et al.'s code.

## 2. Macar et al.'s materials (fetched at run time, not redistributed)

Their repository (github.com/safety-research/introspection-mechanisms) carries **no license**, so we
do not copy its text into this public repository. At run time the code clones it at the pinned
commit **`5d5d9b485e293dc766c3a056cfe50283a7e435b6`** and reads these constants by parsing the
source (never executing it); their SHA-256 hashes are logged with the results:
- the introduction text of the detection prompt (`_build_introspection_messages`);
- the 50 default test concepts (`DEFAULT_TEST_CONCEPTS`) and the 100 baseline words
  (`DEFAULT_BASELINE_WORDS`);
- the judge's detection criteria (`CLAIMS_DETECTION_CRITERIA`).

## 3. Layers (fixed here by rule; L = number of blocks)

| | rule | 27B (L = 62) | 4B (L = 34) | 1B (L = 26, tests only) |
|---|---|---|---|---|
| injection (Gate 1 and fingerprint calibration) | Macar et al.'s block 37 for 27B; round(L × 37/62) otherwise | **37** | **20** | 16 |
| fingerprint layer **(a)** | same relative depth as Study 16's block 21 of 32: round(L × 21/32) | **41** | **22** | 17 |
| fingerprint layer **(b)** | just before the gate region: Macar et al. place gates in blocks 45–61 of 27B (Hasik: block 45 causally important), so (b) = 44 for 27B and round(L × 45/62) − 1 otherwise | **44** | **24** | 18 |

**Layer (a) is primary** (relative depth matched to Studies 15–16, so cross-model comparisons are
like for like). Layer (b) gets the same pre-registered tests, reported but not used for prediction
scoring.

## 4. Gate 1 — does the model report real injections? (Macar et al.'s protocol)

- **Prompt:** their introduction, "Ok.", then "Trial {k}: Do you detect an injected thought? If so,
  what is the injected thought about?"; Gemma chat template, no system turn (as in their code).
- **Concept vectors (their recipe):** last-token activation at the injection block for the
  chat-formatted prompt "Tell me about {word}", minus the mean over their 100 baseline words; not
  normalized.
- **Injection:** strength **4.0** × vector, added at the injection block's output from the token
  before "Trial {k}" through every generated token.
- **Generation:** temperature 1.0, up to 100 new tokens, fixed seeds.
- **Trials:** **200 injected** (their 50 concepts × trials k = 1–4) and **200 controls** (no
  injection; trials k = 1–10, 20 samples each).
- **Judge (decided by the researcher: Gemma-27B judge).** Macar et al. used an OpenAI model (GPT-4.1-mini); we
  spend no money. Proposed: **Gemma-3-27B-it as a local judge for both models' responses**, given
  their exact criteria (detection must be claimed affirmatively and coherently, and **before** the
  concept word appears), greedy, up to 500 tokens, parsed for "Answer: YES/NO" as in their code
  (unparseable = no claim, counted and reported). The 4B responses are judged on the GPU too, so
  both models have the same judge.
  - *Cross-check (reported, not used for the bars):* a fixed keyword rule (§9) with its agreement
    rate; agreement below 80% is flagged.
- **Measure:** report = hits − false alarms (% of injected trials judged as claiming detection minus
  % of control trials), 95% bootstrap CI (resampling trials within each group, 10,000 times, seed
  20261071).
- **Bars:** **REPORTS** if ≥ 20 points with CI lower bound > 0; **DOESN'T REPORT** if ≤ 5 points;
  otherwise **WEAK**.
- **Stop rule:** if 27B is not REPORTS, its fingerprint arm is **not run**; the study reports the gate
  result and stops for 27B. The 4B fingerprint arm runs regardless; its gate result sets its label.

## 5. Fingerprint (calibration split)

- **Calibration prompts:** the yes/no form of the detection prompt: their introduction, "Ok.",
  "Trial {k}: Do you detect an injected thought? Answer yes or no." with **k = 201–300** (100 draws,
  each of the 50 concepts twice). Each draw is run clean and with the Gate 1 injection (strength 4,
  from the token before "Trial" through the answer position).
- At the **answer position** (last prompt token) at layers (a) and (b):
  - **FP-B ("response only") — primary (decided by the researcher):**
    mean(injected − clean − 4·v_concept). Under this protocol the answer position itself is
    injected, so its state contains the injected vector. FP-B subtracts that exactly, leaving only
    what the model computed in response, as in Studies 15–16 (where the answer position was never
    injected).
  - **FP-A ("raw"):** mean(injected − clean), which includes the injected vectors themselves.
  - FP-A is run as a secondary condition (logit contrasts only).
- Each fingerprint is unit-normalized. N(layer) = mean clean norm at that layer. Hashes are committed
  before the dose calibration.

## 6. Doses (positive control only; as Study 16)

The unit (yes − no) unembedding direction (Study 10 yes/no token sets) × f · N, added at the layer,
answer position only, on the detection yes/no prompt, clean, with **k = 301–420**:
- **Stage 1:** 20 prompts, 40 geometric f from 0.02 to 16; f_hi = smallest f with yes ≥ 90%, f_lo =
  largest f below it with yes < 25%. **Stop** (for that model and layer) if 90% is never reached.
- **Stage 2:** 100 prompts, 40 linear f from 0.8 f_lo to 1.25 f_hi (from f_hi / 4 if f_lo = 0).
  Doses s25, s50, s90 = smallest f with yes ≥ 25 / 50 / 90%. Collapsed doses are run once and
  reported; no re-calibration.
- Run separately for layers (a) and (b).

## 7. Test (clean trials)

- **n = 200 test draws, k = 1–200** (disjoint from calibration). Each draw has the detection yes/no
  prompt and one control prompt, "Trial {k}: {control question} Answer yes or no.", with the 10
  Study 10 control questions cycling (all true answer "no").
- **Conditions** at each dose and each layer, answer position only:
  - (a) detection + primary FP;
  - (b) control + primary FP;
  - (c) detection + 5 random unit directions (seed 20261072);
  - (d) detection + (yes − no);
  - secondary: detection and control + the other FP.
- **Measures:** greedy first token (yes / no / neither); yes − no logit.
- **Note on variety:** test prompts differ only in the trial number and the control question,
  unlike Studies 14–16 (which varied ten sentences). Results describe this prompt family; the
  report says so.

## 8. Outcomes and bars

Per model and layer, at the top dose (as Study 16):
- **V1:** > 20% "neither" at a dose → excluded.
- **V2:** control "no" ≥ 80% at dose 0.
- **V3:** positive control ≥ 50% "yes" at s90 on test data, else inconclusive and Q1–Q3 not scored.
- **Q1** (detection rise): a ≥ 20 / b 5–20 / c < 5.
- **Q2** (diff = detection rise − control rise): a = diff ≥ 10 & control < 10 / b = diff < 10 &
  control ≥ 10 / c = both ≥ 10 & diff ≥ 10 / d = diff < 10 & control < 10.
- **Q3** (diff = detection rise − random rise): a = −10 < diff < 10 / b = diff ≥ 10 / c = diff ≤ −10.
- **Secondary logit contrasts** (95% bootstrap CIs over test draws, seed 20261073):
  - L_det − L_ctl: "detection-specific lean" / "leans more on unrelated questions" / "equal lean" /
    "no lean";
  - L_det − L_rand: "stronger than" / "weaker than" / "no different from random".
- **Per-question:** control "yes" by question, the single-question flag, and leave-one-question-out
  Q2.

**KEY COMPARISON (pre-registered).** C = L_det − L_ctl at s90 (primary FP), per model, at each layer.
"Positive" = CI entirely > 0.
- Reference (existing data): 3B (Study 15) C = −0.43 [−0.46, −0.39]; 8B (Study 16) C = −1.13
  [−1.19, −1.07]: both not positive.
- Verdict per layer:
  - **SUPPORTED** if 27B (gate REPORTS) has C positive **and** 4B has C not positive;
  - **NOT SUPPORTED, positive everywhere** if both 27B and 4B have C positive;
  - **NOT SUPPORTED, not positive in the reporter** if 27B's C is not positive;
  - **NOT TESTABLE** if 27B fails Gate 1 or V3.
  - If 4B's gate is also REPORTS, the verdict is stated as "both Gemma models report" and rests on
    the 3B/8B references (different protocol), flagged.
- **Caution, stated in advance:** 3B and 8B used a different injection (sentence-only, layer 2) and
  prompt. The **27B vs 4B** comparison (same protocol) is the clean one.

## 9. Fixed details

- Yes/no tokens: first tokens of "Yes", "yes", " Yes", " yes" / "No", "no", " No", " no" (must be
  disjoint).
- Keyword cross-check for Gate 1: claims detection if the response's first sentence, lower-cased,
  contains "i detect", "i do detect", "i notice", "yes" before any of "no", "not", "don't", "do not".
  Otherwise no claim. Reported only.
- Sanity checks before data (each model): zero vector changes nothing; prefix/last-token runs match a
  full pass (CPU < 10⁻³; GPU bf16 < 5 × 10⁻² with the same greedy token); batched = single runs;
  injection start position = the token before "Trial"; disjoint trial-number splits; hashes of
  Macar et al.'s constants recorded.
- Seeds: generation 20261074 (+ batch index); bootstrap as above.

## 10. Run plan, budget and stop rules

- **Laptop (4B):** after this pre-registration (with predictions) is committed. Timing dry run first;
  if the estimate is over 12 h, n is reduced (test ≥ 100, Gate 1 ≥ 100 + 100) and logged before
  data. Each stage saves and resumes.
- **GPU (27B; RUNBOOK_gpu.md):** one rented 80 GB GPU, budget **$40**. One command runs: vectors →
  Gate 1 generation → judge (27B and 4B responses) → gate verdict → (only if REPORTS) fingerprint →
  doses → test → analysis. Results are downloaded and the machine is shut down. The Hugging Face token
  lives only in an environment variable on that machine.
- The analysis script is committed before any real data.
- Sanity failure → stop, fix, log. Deviations are logged before the runs they affect.

## 11. Prediction scoring rules and the researcher's predictions — verbatim

**Scoring:** each prediction is correct if it names the pre-registered outcome above. P3 is scored
against the verdict at layer (a), including "NOT TESTABLE" as an outcome. P4–P6 are **not scorable**
if the 27B fingerprint arm does not run (Gate 1 not REPORTS) or its V3 fails. Layer (b) results are
not scored.

Given 2026-10-26, as multiple-choice answers, exactly:

| # | question | researcher's answer |
|---|---|---|
| P1 | Gate 1, 27B: REPORTS / WEAK / DOESN'T REPORT | **Weak (5-20 points)** |
| P2 | Gate 1, 4B: REPORTS / WEAK / DOESN'T REPORT | **Doesn't report (<= 5)** |
| P3 | Key comparison at layer (a): SUPPORTED / NOT SUPPORTED, positive everywhere / NOT SUPPORTED, not positive in the reporter / NOT TESTABLE | **Supported** |
| P4 | 27B Q1 at layer (a) (a / b / c) | **b: a little (5-20)** |
| P5 | 27B Q2 at layer (a) (a / b / c / d) | **a: specific** |
| P6 | 27B Q3 at layer (a) (a / b / c) | **b: random much weaker** |

Design choices given at the same time: primary fingerprint **"Response only"**; Gate 1 judge
**"Gemma-27B judge"**.

*Note (Claude, not a change):* under the rules above, P1 = WEAK would stop the 27B fingerprint arm,
in which case P3's outcome is NOT TESTABLE and P4–P6 are not scorable.
