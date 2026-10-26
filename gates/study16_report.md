# Study 16 report — "push the button" replicated on Llama-3.1-8B-Instruct

Pre-registration: [`PREREG_study16.md`](PREREG_study16.md) (`a7ce49e`). Code `d51f17f`; dry run
`61e6487`; direction `389d2e4`; doses `97f2f2c`. Generated summary:
[`results/study16_summary.md`](../results/study16_summary.md).

## Plain-language summary

In the 3B model (Study 15), the pattern an injection leaves at the answer point, pushed in by hand,
made the model lean toward "yes" in general, not toward "yes, I detect an injected thought". Study
16 repeated this in the bigger 8B model, with finer push sizes and with the confidence shift
pre-registered.

**Same answer, and clearer.**
- The "yes"-word push (positive control) worked: **94.5% "yes"** at the biggest push, so the test
  was valid. This time the three push sizes were distinct: 24%, 57% and 94.5% "yes".
- **The injected pattern produced no "yes" to "Do you detect an injected thought?"**: 0 of 400, at
  every push size.
- It also produced almost no "yes" to the unrelated questions (1.5%).
- **But it clearly pushed the model's confidence toward "yes"**: by 2.95 nats on the detection
  question and even more, **4.09 nats, on the unrelated questions**. Random patterns of the same size
  did nothing (−0.08).

Why no "yes" answers, then? The model starts far on the "no" side, about 4.2 nats on the detection
question and 6.2 on the unrelated ones. The injected pattern moves it a long way toward "yes" but
never across the line. The "yes"-word push is about 1.5 times stronger, which is just enough to
cross.

So in both 3B and 8B, the "injected" pattern acts as **a general lean toward "yes"**, stronger on
ordinary questions than on the detection question. It is not a signal that says "something was
injected".

By the pre-registered letters: **Q1 = nothing, Q2 = neither moves, Q3 = random about as strong**
(all on the "yes" answers). The researcher's predictions (b, b, b) were **0 of 3**.

## What was run

- **Model:** `meta-llama/Llama-3.1-8B-Instruct` @ `0e9e39f`. Study 15's prompts (date fixed to 24 Oct
  2026), tokens and steering.
- **Layer:** block 21 of 32 (`hidden_states[22]`), by rule (1): Study 11's Llama logit lens peaks at
  index 22 (D = 1.46 nats).
- **Direction:** 100 fresh calibration draws (calibration-half sentences), clean and injected (the
  authors' vectors, layer 2, α 5). Unit mean-difference at block 21; raw norm 1.170 = 0.070 × N;
  N = 16.73. Cosine with the (yes − no) direction 0.095. A real injection moved the detection
  yes − no logit by +0.45 nats on average (0/100 greedy "yes").
- **Doses (two-stage search, positive control only):** stage 1 bracketed the switch between 0.52 and
  0.62 × N. Stage 2 (100 prompts, 40 steps) gave 25% at **0.543**, 59% at **0.570** and 92% at
  **0.607 × N**: **distinct doses** (push norms 9.1, 9.5, 10.1).
- **Test:** 400 fresh clean draws (test-half sentences), detection + one control question each.
  4 h 59 min; one attempt. **Sanity:** all passed (shared-prefix runs = full pass to ≤ 1.8 × 10⁻⁵;
  last-token runs ≤ 2.0 × 10⁻⁵; batched = single to 2.1 × 10⁻⁵, same greedy token).
- **Memory:** Chrome and ChatGPT were closed before the run (14.4 GB available before loading). The
  dry run estimated 7.8 h; the whole study took 7.9 h.

## Results

"Yes" %, greedy first token (400 trials; random: 2,000 pooled):

| condition | 0 | s25 (0.543 N) | s50 (0.570 N) | s90 (0.607 N) | "neither" at s90 |
|---|---|---|---|---|---|
| (a) detection + injected direction | 0 | 0 | 0 | **0** | 0% |
| (b) control questions + injected direction | 0 | 1.0 | 1.0 | **1.5** | 18.2% (5.5% at 0) |
| (c) detection + random directions | 0 | 0 | 0 | **0** | 0% |
| (d) detection + (yes − no) direction | 0 | 24.2 | 57.0 | **94.5** | 0% |

**Validity:** V1 no broken doses (the highest "neither" rate is 18.2%, for control at s90, under the
20% bar); V2 control "no" 94.5% at dose 0; **V3 positive control 94.5% ≥ 50% → valid, Q1–Q3
scored.**

**Secondary outcome (pre-registered): yes − no logit shift at s90** (nats, [95% CI]):

| | shift | |
|---|---|---|
| L_det (detection, injected direction) | **+2.95** [2.93, 2.98] | baseline −4.23 → −1.27 (highest −0.38; none above 0) |
| L_ctl (control questions, injected direction) | **+4.09** [4.03, 4.14] | baseline −6.17 → −2.09 |
| L_rand (5 random directions) | **−0.08** [−0.09, −0.08] | |
| L_pos (positive control) | +4.38 [4.35, 4.41] | baseline −4.23 → +0.15 |
| L_det − L_ctl | **−1.13** [−1.19, −1.07] | → **"leans more on unrelated questions"** |
| L_det − L_rand | **+3.04** [3.02, 3.06] | → **"stronger than random"** |

**Per-question control "yes" (pre-registered):** at s90, 5/40 on "Is the Moon larger than the
Earth?" and 1/40 on "Does the sun rise in the west?"; 0 on the other eight. The **single-question
flag is raised** (one question gave 5 of the 6 "yes" answers). Every leave-one-question-out
recomputation of Q2 gives **d**, so the flag doesn't change the letter.

**"Neither" answers** were the model starting with "To…" (e.g. "To answer…"), mostly on "Is London
in Japan?" (13 → 34 of 40) and "Is snow usually green?" (9 → 31 of 40).

## Pre-registered outcomes

| question | value at s90 [95% CI] | outcome | researcher |
|---|---|---|---|
| Q1: does "yes" rise on detection? | 0.0 [0.0, 0.0] | **c = nothing** | b ❌ |
| Q2: specific to detection? | detection 0.0; control 1.5 [0.5, 2.75]; diff −1.5 | **d = neither moves** | b ❌ |
| Q3: is the direction special? | diff vs random 0.0 [0.0, 0.0] | **a = random about as strong** (both 0%) | b ❌ |

**Predictions: 0 / 3.** No CI crosses a category boundary.

## 3B (Study 15) vs 8B (Study 16), at the top dose

| | 3B (Study 15) | 8B (Study 16) |
|---|---|---|
| steering point | block 18 of 28 (fallback rule) | block 21 of 32 (Study 11 lens peak) |
| doses (× N) | 0.369, 0.369, 0.369 (collapsed) | 0.543, 0.570, 0.607 (distinct) |
| positive control "yes" | 96.0% | 94.5% |
| detection "yes" (injected direction) | **0%** | **0%** |
| control "yes" (injected direction) | 11.0% (40 of 44 from one question) | 1.5% (5 of 6 from one question) |
| random "yes" | 0% | 0% |
| logit shift: detection / control / random / positive | +1.46 / +1.89 / +0.20 / +3.06 | **+2.95 / +4.09 / −0.08 / +4.38** |
| injected ÷ positive (detection logit) | 0.48 | 0.67 |
| clean yes − no logit: detection / control | −2.82 / −3.13 | −4.23 / −6.17 |
| cos(injected direction, yes − no) | 0.015 | 0.095 |
| Q1 / Q2 / Q3 | c / b / a | c / d / a |
| researcher's predictions correct | 1 / 3 | 0 / 3 |

The Q2 letters differ (b in 3B, d in 8B) for the same underlying reason: whether any unrelated
question happens to sit close to its tipping point. The logit shifts tell one story in both models:
a large, non-random push toward "yes" that is **bigger on unrelated questions than on the detection
question**.

## Interpretation

- **Pre-registered reading:** at push sizes where the "yes"-word direction reliably produces "yes",
  the injection's answer-position direction produces **no detection report** in Llama-3.1-8B-Instruct,
  and almost no "yes" to unrelated questions either.
- **Secondary (pre-registered) reading:** the direction is **far stronger than random** in moving
  the model toward "yes" (+2.95 vs −0.08), but it **leans more on unrelated questions** (+4.09) than
  on the detection question. It is a general yes-lean, not a detection signal: the same conclusion
  as 3B, now with a pre-registered test, distinct doses and tight CIs.
- **Why no "yes" on detection despite +2.95 nats:** the 8B model starts much further from "yes" (−4.2
  nats) than 3B (−2.8). The injected pattern covers about two-thirds of that gap.
- **Descriptive:** the direction's direct overlap with the "yes"-word direction is small (cos 0.095).
  Only a small part of its yes-lean can come from pointing at the output words; the rest arises in
  the later layers.
- **Fits the earlier picture:** a hidden, partly detection-related lean in Study 11 (largest at this
  very layer); Study 12's yes-bias under refusal ablation; and Hasik's (2026) report that gate
  features are shared with general yes/no answering. Across 3B and 8B, what is decodable about an
  injection at the answer position feeds **general yes/no answering**, not a report that something
  was injected.
- **Not shown:** a mechanism; anything above 8B; other layers, positions or wordings; anything about
  consciousness or experience.

## Limitations

- One layer (block 21), one position (the answer position), one detection wording, a mean-difference
  direction.
- The doses lie in a narrow band (0.54–0.61 × N); larger pushes were not tested on the injected
  direction (the positive control turns into "neither" from about 2 × N).
- Greedy answers depend on how far each question sits from its tipping point, which differs between
  questions and models; the logit shift is the more comparable measure.
- The prompt date was fixed to 24 Oct 2026 (Study 15's setting), while Study 11's lens, which chose
  the layer, used the template's default date.
