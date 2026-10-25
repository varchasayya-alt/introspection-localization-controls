# Study 14 pre-registration — "push the button": does the injected direction make the model say "yes"?

Written 2026-10-24, **before any Study 14 calibration or test data**. Design set by the researcher;
this file fills in the details. The researcher's predictions were given together with the design
and bars, so the scoring rules (§6) and the predictions (§8) are committed together (as in Study 1),
both before any data.

## 0. Plain-language version

In Studies 10–13 the models did not say "yes" when asked "Do you detect an injected thought?", even
though a simple probe can read the injection from their insides. Study 14 asks: if we take the
pattern that an injection leaves at the moment the model is about to answer, and **add that pattern
by hand** to trials where nothing was injected, does the model start saying "yes"? If it does, is
that because the pattern means "something was injected" (it should then matter only for the
detection question), or is it just a general "say yes" button (it would then also make the model say
"yes" to "Is Paris the capital of Germany?")? Two checks: random patterns of the same size (is
*this* pattern special?), and a pattern built directly from the model's own "yes" and "no" output
words (can "yes" be pushed at all with a push of this size?).

## 1. Question

On **clean** (non-injected) trials, does adding the "injected" direction at the answer position make
Llama-3.2-3B-Instruct say "yes" to the detection question, and is that specific to detection?

Motivation: Ferrara (2026, arXiv 2608.20569) and our Studies 11–13 find the injection decodable
but not reported, without causally testing the state-to-report path; Macar et al. (2026) find only
partial detection (max 25.1%) when patching gate features in Gemma3-27B; Hasik (2026, LessWrong)
argues that control questions are needed to tell detection from a generic yes-shift.

## 2. Model and reused setup

- `meta-llama/Llama-3.2-3B-Instruct` @ `0cb88a4f764b7a12671c53f0838cd831a0843b95` (28 blocks,
  width 3072). Bf16 storage, fp32 math (D1).
- **The Study 10/11 yes/no setup exactly as run for 3B in Study 13:** Study 10's prompt (intro turn,
  "Ok.", then the 10-sentence block + question), the detection question "Do you detect an injected
  thought? Answer yes or no.", the 10 Study 10 control questions (+ " Answer yes or no."), all with
  true answer "no"; Llama-3.2 chat format via the tokenizer's template.
- **Injection used to compute the direction (calibration only):** Study 13's 3B setting: concept
  vectors recomputed from 3B with the authors' recipe (Study 13 `3b_l2` files, checked by SHA-256),
  layer 2, α = 7.937 (2.186 × mean residual norm), sentence words only, one sentence per trial.
- **One fixed detail:** the Llama-3.2 template writes today's date into a system line ("Today Date:
  …"). To keep every calibration and test prompt identical in form whatever day it runs, the date is
  fixed to `24 Oct 2026` (Study 13's prompts carried their own run date).

## 3. Materials (fresh, fixed by seeds before any run)

- **Sentence split.** The 100-sentence pool (Studies 3–13) is shuffled with seed 20261041: the first
  50 are **calibration sentences**, the other 50 **test sentences**. No sentence appears in both.
- **Calibration draws (100):** 10 sentences sampled from the calibration half (seed 20261042); the
  20 (concept, vector type) pairs cycled 5 times; injected position cycling 1–10. Draws 0–49 =
  *fit* half, 50–99 = *held-out* half. Calibration uses the detection question only.
- **Test draws (n = 400 clean trials):** 10 sentences sampled from the test half (seed 20261043);
  control question cycling through the 10 control questions. Test trials are never injected, so no
  concept is used at test.
- **Random directions:** 5 Gaussian directions (seed 20261044), fixed across all trials.
- **Bootstrap:** 10,000 resamples over test draws, seed 20261045, percentile 95% CIs.

## 4. Calibration (direction and layer)

For each calibration draw, run the detection prompt clean and with the injection; save the residual
stream at the **answer position** (last prompt token) at every block output.

- **Candidate steering layers:** blocks 0–26 (block ℓ's output = `hidden_states[ℓ+1]`). Block 27 is
  excluded because Hugging Face returns its output after the final norm, so the read point and the
  push point would differ. (Blocks 0–2 cannot differ at the answer position, since the push enters at
  the output of block 2 on earlier tokens; they will score AUC 0.5 and lose.)
- **Layer rule (calibration only):** per layer, the Study 11 probe (standardised logistic regression,
  L2 0.05) is fitted on the fit half (injected vs clean) and scored by AUC on the held-out half. The
  **steering layer = highest held-out AUC, rounded to 3 decimals; tie → earlier layer.** The layer is
  never chosen by its effect on "yes".
- **Direction:** v = mean(injected) − mean(clean) over **all 100** calibration draws at the chosen
  layer (raw, not normalised).
- **Calibration stop rule:** if the chosen layer's held-out AUC is < 0.75, stop and report (no test).
  One calibration attempt; any re-run is a logged deviation.
- After calibration: commit the frozen code, the chosen layer, ‖v‖, and the SHA-256 of the saved
  direction file (the vector itself stays outside the repository, like earlier vectors).

## 5. Intervention and measurement (test)

On each clean test trial, add a vector at the **chosen layer's block output, at the answer position
only** (the clean KV cache for all earlier tokens is reused; only the last token is re-run).
Doses k ∈ {0, 1, 2, 4, 8}; ‖·‖ below means the push size k‖v‖.

| condition | question | added vector |
|---|---|---|
| (a) detection | detection | k · v |
| (b) control | the trial's control question | k · v |
| (c) random | detection | each of 5 random unit directions × k‖v‖ |
| (d) positive control | detection | unit (yes − no) unembedding direction × k‖v‖ |

(yes − no) = mean of the unembedding rows of the yes tokens (" Yes", "Yes", "yes", " yes") minus
the mean of the no rows (" no", "no", " No", "No") — the Study 10 token sets. k = 0 is the same
clean run for (a), (c), (d).

- **Primary measure:** the greedy first token (argmax over the whole vocabulary) → "yes" if it is
  one of the yes tokens, "no" if one of the no tokens, otherwise "neither". **Yes-rate** = % of trials
  whose first token is "yes" (out of all trials).
- **Secondary:** yes − no logit (log-sum-exp over each token set), mean shift from k = 0.
- **Recorded:** % "neither" for every condition and dose; ‖v‖ relative to the mean residual norm at
  the answer position at that layer (descriptive).
- **Sanity before the test** (on a calibration draw): α = 0 leaves logits unchanged; the
  last-token-only run reproduces a full pass (k = 0 and k > 0, max logit difference < 10⁻³); the
  calibration and test sentence sets are disjoint; vector hashes match.

## 6. Pre-registered bars and scoring

**Rise** = yes-rate at k = 8 minus yes-rate at k = 0 (points; paired by trial; 95% bootstrap CI).
The random rise uses, per trial, the mean over the 5 random directions. Categories are assigned on
the point estimates; the CIs are reported beside them, and a CI crossing a category boundary is
stated explicitly.

- **Validity V1 (broken doses):** a condition at a dose where > 20% of answers are "neither" is
  **broken** and excluded from the bars (reported anyway; the random condition is judged on its
  pooled 5 × n answers). If k = 8 is broken for any condition entering a question, that question is
  scored at the **highest dose where all its conditions are valid** (stated in the report); if that
  is k = 0 only, the question is **not scorable**.
- **Validity V2 (control baseline):** the control questions must be answered "no" (greedy) in ≥ 80%
  of clean test trials at k = 0; otherwise Q2 is not scorable.
- **Validity V3 (positive control):** if the positive control's rise at the scored dose is < 20
  points, a push of this size cannot move "yes" at all, so a Q1 "nothing" outcome is reported as
  **inconclusive (push too weak)**; Q1–Q3 are still scored.

**Q1 — does "yes" rise on the detection question?** rise(a):
- **a = rises a lot:** ≥ 20 points; **b = a little:** 5–20; **c = nothing:** < 5.

**Q2 — is it specific to detection?** Let diff = rise(a) − rise(b).
- **a = specific:** diff ≥ 10 and rise(b) < 10;
- **b = generic "yes button":** diff < 10 and rise(b) ≥ 10;
- **c = both rise, detection more:** rise(a) ≥ 10, rise(b) ≥ 10 and diff ≥ 10;
- **d = neither moves (no button):** diff < 10 and rise(b) < 10.

*Clarification (Claude, before any data):* the researcher's bars defined "specific" as diff ≥ 10,
which overlaps with "both rise, detection more"; "specific" is read as diff ≥ 10 **with the control
not rising** (< 10), and option d was added so every outcome has a letter.

**Q3 — is the injected direction special?** Let diff = rise(a) − mean random rise(c).
- **a = random about as strong:** −10 < diff < 10;
- **b = random much weaker ("direction is special"):** diff ≥ 10;
- **c = random much stronger:** diff ≤ −10.

*Clarification (Claude, before any data):* the researcher's plain meaning for letter b ("random much
weaker") equals the bar "direction is special"; a and c were filled in around it.

Secondary (descriptive, not scored): the full dose curves of yes-rate and yes − no logit for all
four conditions, and the positive control's rise.

## 7. Stop rules and run plan

- Sanity failure → stop, fix, log, re-run sanity before any test trial.
- Calibration AUC < 0.75 → stop (§4).
- **Timing:** a dry run of 5 trials (calibration-half sentences, placeholder direction, outputs
  discarded) estimates the total. If > 6 hours, n is reduced and logged as a deviation **before**
  any test data.
- Order: this pre-registration (commit + push) → timing dry run → calibration → frozen code +
  direction hash (commit) → test → analysis with the committed script → report.
- **No 8B replication** in this study; stop and report after 3B.

## 8. Researcher's predictions — verbatim

Given 2026-10-24 together with the design, before any Study 14 data:

> My predictions (record verbatim with plain meanings): Q1 = a (rises a lot), Q2 = b (equally on
> unrelated questions: a generic "say yes" button), Q3 = b (random much weaker).

Plain meanings:
- **Q1 = a:** adding the injected pattern makes the model say "yes" to the detection question a lot
  more (≥ 20 points).
- **Q2 = b:** it makes the model say "yes" about as much to unrelated questions like "Is Paris the
  capital of Germany?" — a generic "say yes" button, not a detection signal.
- **Q3 = b:** random patterns of the same size push "yes" much less (the injected pattern is
  special).
