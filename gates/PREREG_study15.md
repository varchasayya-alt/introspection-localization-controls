# Study 15 pre-registration — "push the button", with pushes that can move "yes" (3B)

Written 2026-10-25, **before any Study 15 data**. Design set by the researcher (a combination of
Study 14's follow-up options A and B); this file fills in the details. As in Study 14, the
researcher gave the design, bars and predictions together, so they are committed together, before
any data.

## 0. Plain-language version

Study 14 asked whether adding the "injected" pattern by hand makes the model say "yes". Nothing
moved, not even our positive control (a push built from the model's own "yes"/"no" words), so the
pushes were too weak to answer the question. Study 15 fixes that in two ways:
1. **A later layer** (block 18 of 28), closer to where the model forms its answer.
2. **Pushes sized by the positive control.** On practice trials only, we find how big a "yes"-word
   push has to be to make the model say "yes" about 25%, 50% and 90% of the time. Then the injected
   pattern, random patterns and the unrelated-question condition all get pushes of exactly those
   sizes.

If the "yes"-word push works on the real test trials but the injected pattern doesn't, the injected
pattern is not a "yes" button. If the injected pattern also makes the model say "yes" to "Is Paris
the capital of Germany?", it's a generic "yes" button, not a detection signal.

## 1. Question (unchanged from Study 14)

On **clean** (non-injected) trials, does adding the "injected" direction make Llama-3.2-3B-Instruct
say "yes" to the detection question, and is that specific to detection? Now the pushes are large
enough that the positive control can move "yes".

Option C (patching real injected activations) is **not** used: a real injection never flipped an
answer in Study 14's calibration (0/100), so patching real activations can't answer the question.

## 2. Model and reused setup

- `meta-llama/Llama-3.2-3B-Instruct` @ `0cb88a4f764b7a12671c53f0838cd831a0843b95`; bf16 storage,
  fp32 math (D1).
- **Identical to Study 14:** prompts (Study 10's yes/no setup, Llama-3.2 chat format, date line fixed
  to `24 Oct 2026`), the detection question, the 10 control questions (true answer "no"), the yes/no
  token sets, the greedy first-token measure, the batched last-token steering (D7), and the sentence
  split (Study 14's calibration half and test half of the 100-sentence pool).

## 3. Steering layer (fixed here, before any Study 15 data)

The researcher's rule: the 3B layer with the largest detection-specific yes-lean (D) in existing
Study 11/13 logit-lens data; **if no 3B logit-lens data exists, use layer 18.**

**No 3B logit-lens data exists.** Study 11's logit lens was run only on 8B models (Tülu SFT, DPO,
final, Llama-3.1-8B-Instruct). Study 13 saved 3B answer-position states but computed no logit-lens
values. So the steering layer is **block 18** (the output of `model.model.layers[18]` =
`hidden_states[19]`; 28 blocks, so about two-thirds of the way through). Computing a new lens from
Study 13's saved states would be a new analysis, so it is not done.

## 4. Directions (from existing calibration data; no yes-effects used)

- **Injection direction:** v = mean(injected) − mean(clean) at `hidden_states[19]` over Study 14's
  100 calibration draws (calibration-half sentences; Study 13's 3B injection, layer 2, α 7.937;
  states saved by Study 14, outside the repository), then **unit-normalized**. Its SHA-256 is
  committed before the dose calibration.
- **Typical activation norm** N = mean ‖h‖ at `hidden_states[19]`, answer position, over the same
  100 **clean** Study 14 calibration states.
- **Positive control:** the unit (yes − no) unembedding direction (mean of the yes-token rows minus
  mean of the no-token rows; Study 14's definition).
- **Random:** 5 Gaussian unit directions, seed 20261053 (new).

## 5. Dose calibration (positive control, calibration trials only)

- **Material:** 100 fresh dose-calibration draws (10 sentences each from Study 14's calibration
  half, seed 20261051), detection question, clean.
- **Grid:** 40 push sizes f · N, with f geometric from 0.02 to 16 (f_i = 0.02 × 800^(i/39)). The
  positive-control direction × f · N is added at block 18, answer position only.
- **Rule:** for each target t ∈ {25%, 50%, 90%}, dose s_t = the **smallest grid f whose
  calibration yes-rate is ≥ t**. The doses are **{0, s25, s50, s90}** (as fractions of N).
- **Calibration stop rule:** if the positive control never reaches 90% "yes" on the grid, stop and
  report (the design can't run). One attempt; any re-run is a logged deviation.
- Only the positive control is run in calibration. The injection direction's effect on "yes" is
  never looked at before the test.

## 6. Test

- **Material:** 400 fresh clean test draws (10 sentences each from Study 14's test half, seed
  20261052), each with the detection question and one control question (cycling through the 10).
- At each dose s ∈ {s25, s50, s90} (vector norm s · N), added at block 18, answer position only:

| condition | question | added vector |
|---|---|---|
| (a) detection | detection | unit v × s · N |
| (b) control | the trial's control question | unit v × s · N |
| (c) random | detection | each of 5 random unit directions × s · N |
| (d) positive control | detection | unit (yes − no) × s · N |

- Dose 0 is the shared clean run. **Primary:** greedy first token, yes / no / neither; yes-rate = % of
  all trials. **Secondary:** yes − no logit shift. "Neither" is recorded everywhere.
- **Sanity before the test:** zero vector changes nothing; last-token runs match a full pass
  (< 10⁻³) at the top dose for (a) and (d); batched runs match single runs (< 10⁻³, same greedy
  token); calibration and test sentences disjoint; direction hash matches.

## 7. Bars and scoring (Study 14's, with its two pre-data fixes)

**Rise** = yes-rate at the top dose (s90) minus yes-rate at dose 0 (points, paired by trial; 95%
bootstrap CI, 10,000 resamples over test draws, seed 20261054). The random rise is, per trial, the
mean over the 5 directions. Categories use point estimates; CIs are reported beside them, and a CI
crossing a boundary is stated.

- **V1 (broken doses):** a condition at a dose with > 20% "neither" answers is excluded from the bars
  (reported anyway; random pooled over its 5 directions). A question is scored at the highest dose
  where all its conditions are valid; if none, it is not scorable.
- **V2:** the control questions must be answered "no" in ≥ 80% of clean test trials, else Q2 is not
  scorable.
- **V3 (positive control, researcher's rule):** on **test** data, the positive control must reach
  **≥ 50% "yes" at the top dose (s90)**. If it doesn't, **the study is inconclusive and Q1–Q3 are
  not scored** (reported descriptively only).

**Q1 — does "yes" rise on the detection question?** rise(a): **a** ≥ 20 points (rises a lot);
**b** 5–20 (a little); **c** < 5 (nothing).

**Q2 — is it specific to detection?** diff = rise(a) − rise(b):
**a** = diff ≥ 10 and rise(b) < 10 (specific: detection rises, control doesn't);
**b** = diff < 10 and rise(b) ≥ 10 (generic "yes button");
**c** = rise(a) ≥ 10, rise(b) ≥ 10 and diff ≥ 10 (both rise, detection more);
**d** = diff < 10 and rise(b) < 10 (neither moves).

**Q3 — is the injected direction special?** diff = rise(a) − mean random rise(c):
**a** = −10 < diff < 10 (random about as strong); **b** = diff ≥ 10 (random much weaker: the
direction is special); **c** = diff ≤ −10 (random much stronger).

Secondary (descriptive, not scored): yes-rate and logit curves for all conditions at all doses; the
dose-calibration curve.

## 8. Run plan and stop rules

- Order: this pre-registration (commit + push) → code + timing dry run (5 trials; placeholder
  doses; calibration-half sentences; outputs discarded) → frozen code, direction and norm (commit) →
  dose calibration → doses (commit) → test → analysis with the committed script → report.
- **Timing:** if the dry-run estimate is > 6 hours, n is reduced and logged as a deviation before any
  data.
- Sanity failure → stop, fix, log. Calibration stop rule as in §5.
- **3B only;** stop and report after 3B.

## 9. Researcher's predictions — verbatim

Given 2026-10-25 together with the design, before any Study 15 data:

> My predictions: same as Study 14: Q1 = a, Q2 = b, Q3 = b.

Plain meanings (as in Study 14):
- **Q1 = a:** the injected pattern makes the model say "yes" to the detection question a lot more
  (≥ 20 points).
- **Q2 = b:** it does about as much for unrelated questions: a generic "say yes" button.
- **Q3 = b:** random patterns of the same size push "yes" much less (the injected pattern is
  special).
