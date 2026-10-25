# Study 14 report — "push the button" on Llama-3.2-3B-Instruct

Pre-registration: [`PREREG_study14.md`](PREREG_study14.md) (`1800c98`). Frozen code `001e20a`;
calibration and direction hash `12e430b`. Generated summary:
[`results/study14_summary.md`](../results/study14_summary.md).

## Plain-language summary

The question: in earlier studies the model "knew" inside that something had been injected (a
probe could read it), but it didn't say "yes" when asked. So we took the pattern an injection leaves
at the moment the model is about to answer and **added it by hand** to normal, non-injected trials,
at 1, 2, 4 and 8 times its natural size. Would the model start saying "yes"?

**It didn't. Not once, in 400 trials, at any size.** The same was true for unrelated questions
("Is Paris the capital of Germany?"), for random patterns of the same size, and **even for the
positive control**, a pattern built from the model's own "yes" and "no" output words, which should
have pushed it toward "yes" if anything could. The model said "no" on every trial.

Because the positive control failed too, the honest reading is: **pushes of this size, at this one
spot, are too weak to change the answer at all.** So this study cannot tell us whether the injected
pattern is a "detection signal" or a generic "say yes" button. By the rules we wrote down
beforehand, the result is "nothing", marked **inconclusive (push too weak)**.

One useful number: on these prompts the model leans toward "no" by about 2.8 nats (odds of roughly
17 to 1). The biggest push changed that lean by less than 0.1 nats. A real injection changes it by
about 0.16 nats on average, also far too little to flip an answer.

## What was run

- **Model:** `meta-llama/Llama-3.2-3B-Instruct` @ `0cb88a4`; Study 10's yes/no prompts (date line
  fixed to 24 Oct 2026); Study 13's 3B injection (layer 2, α 7.937) used only for calibration.
- **Calibration (100 draws, calibration-half sentences):** probe AUC (fit 0–49 → held-out 50–99)
  was 0.5 at blocks 0–2, 0.959 at block 3, 0.999 at block 4 and **1.000 at blocks 5–11**; by the
  tie rule the **steering layer is block 5**. Direction v = mean(injected) − mean(clean) at the
  answer position: ‖v‖ = 0.123, **0.025 × the mean residual norm** there (4.88). So k = 8 adds a
  vector about 0.2 × that norm. Direction file SHA-256 `706f0278…b475`.
- **Test:** 400 clean draws (test-half sentences, disjoint from calibration), each with the detection
  question and one control question; the vector was added at block 5's output at the answer position
  only; doses k ∈ {0, 1, 2, 4, 8}; conditions (a) detection + v, (b) control + v, (c) detection + 5
  random directions, (d) detection + the (yes − no) unembedding direction, all with norm k‖v‖.
- **Sanity:** all passed (zero vector changes nothing; last-token-only runs match a full pass to
  ≤ 1.6 × 10⁻⁵; batched runs (D7) match single runs to 1.5 × 10⁻⁵ with the same greedy token).
- **Run time:** calibration 34 min, test 4 h 15 min, one attempt, no restarts.

## Results

Greedy first token, % of 400 trials (random: 2,000 answers pooled).

| condition | k = 0 | k = 1 | k = 2 | k = 4 | k = 8 | "neither" at any dose |
|---|---|---|---|---|---|---|
| (a) detection + injected direction | 0% yes | 0% | 0% | 0% | 0% | 0% |
| (b) control questions + injected direction | 0% yes | 0% | 0% | 0% | 0% | 0% |
| (c) detection + random directions | 0% yes | 0% | 0% | 0% | 0% | 0% |
| (d) detection + (yes − no) direction | 0% yes | 0% | 0% | 0% | 0% | 0% |

No dose was broken (V1); the control questions were answered "no" in 100% of clean trials (V2 ok).

**Secondary: yes − no logit** (nats; mean at k = 0, and the mean [min, max] shift at k = 8):

| | baseline (k = 0) | shift at k = 8 |
|---|---|---|
| (a) detection + v | −2.83 (range −3.49 to −1.78) | −0.001 [−0.033, +0.025] |
| (b) control + v | −3.16 (range −5.28 to −0.29) | −0.020 [−0.128, +0.057] |
| (c) detection + random (5 directions) | −2.83 | −0.005 to −0.049 per direction |
| (d) detection + (yes − no) | −2.83 | −0.044 [−0.077, −0.004] |
| *for reference: a real injection (calibration, layer 2 on one sentence)* | −2.82 | **+0.158** (SD 0.234); 0/100 greedy "yes" |

## Pre-registered outcomes

| question | rise or difference (k = 8) [95% CI] | outcome | researcher |
|---|---|---|---|
| Q1: does "yes" rise on detection? | 0.0 [0.0, 0.0] | **c = nothing**, flagged **inconclusive (push too weak)** by V3 (positive-control rise 0.0 < 20) | a ❌ |
| Q2: specific to detection? | diff 0.0 [0.0, 0.0]; control rise 0.0 | **d = neither moves** | b ❌ |
| Q3: is the direction special? | diff vs random 0.0 [0.0, 0.0] | **a = random about as strong** (both zero) | b ❌ |

**Predictions: 0 / 3.** Every CI is [0, 0] because no trial changed its answer, so no CI crosses a
category boundary.

## Interpretation

- **What the pre-registered result says:** at these doses, adding the injection's answer-position
  footprint does not make the 3B model say "yes", to either question. The positive control failed
  too, so this is **not** evidence that the injected direction lacks a causal role. A push of 0.2 ×
  the residual norm, at one layer and one position, does not move this answer for any direction we
  tried.
- **Descriptive, not pre-registered:**
  - The steering moved the yes − no logit by about **1% of what a real injection does**
    (−0.001 vs +0.158 nats). The answer-position footprint at block 5 is perfectly decodable
    (AUC 1.000) but, at up to 8× its natural size, it doesn't carry the injection's small yes-lean.
    That lean probably arrives through other routes, such as later layers attending back to the
    pushed sentence's tokens, which a one-position push leaves untouched.
  - The positive control's shift had the wrong sign (−0.04). An unembedding direction added at an
    early layer (5 of 28) isn't guaranteed to survive 22 more layers as "yes". So V3 failing means
    "too weak *or* the wrong place for that control", not only "too weak".
  - The tie rule picked the **earliest** perfect-AUC layer (5). Perfect decodability appears
    early, but the yes/no decision may be formed much later.
- **Relation to other work:** Macar et al. (2026) got only partial detection (max 25.1%) from
  patching gate features in Gemma3-27B. Hasik (2026) notes that gate features are shared with
  general yes/no answering. Our study adds a negative causal result at 3B: the decodable footprint,
  pushed in at one position, does nothing to the report at these doses. It does not settle the
  "detection signal vs yes button" question.

## Limitations

- Doses were set relative to ‖v‖, which turned out small (2.5% of the residual norm), so the
  largest push was modest. The positive control shows the doses were too weak to test the main
  question.
- One layer (chosen by a calibration-only rule), one position, one model (3B), one wording.
- Greedy first token is a coarse measure; the logit shifts above are the more sensitive view.

## What would answer the question (not run; needs a new pre-registration and the researcher's decision)

- Doses set relative to the **residual norm** (e.g. pushes of 0.5–4 × the norm), with the
  positive control required to move "yes" before the main comparison is read; and/or
- steering at a **later layer** (where the yes/no lean formed in Study 11, about two-thirds of the
  way through) or at **all positions after the sentences**; and/or
- **patching** whole injected states (every position at a layer) into clean runs instead of adding
  a mean vector at one position.
