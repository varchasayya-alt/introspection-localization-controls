# Study 15 report — "push the button" with pushes sized by the positive control (3B)

Pre-registration: [`PREREG_study15.md`](PREREG_study15.md) (`9a7fa84`). Frozen setup `eb0dc7f`;
dose calibration `fff0b87`. Generated summary: [`results/study15_summary.md`](../results/study15_summary.md).

## Plain-language summary

Study 14's pushes were too small to change any answer. Study 15 pushed at a later layer (18 of 28)
and sized the push so that a "yes"-word push really works. On the test trials it made the model say
"yes" **96%** of the time, so this time the test is valid.

At that same size, the **injected pattern** did this:
- **On "Do you detect an injected thought?" it never produced a "yes"** (0 of 400).
- **On unrelated questions it produced "yes" 11% of the time**, but almost all of that is one
  question, "Does a week have ten days?", which flipped in all 40 of its trials. That question was
  already the one the model was least sure about.
- **Random patterns** of the same size: 0% "yes".

Looking at the model's confidence instead of its final word gives a clearer picture. The injected
pattern moves the model toward "yes" by about **half as much** as the "yes"-word push. It does this
by about the same amount for unrelated questions as for the detection question (slightly more for
the unrelated ones). Random patterns barely move it.

So the injected pattern acts like **a weak, general "lean toward yes"**, not a "something was
injected" signal. It is too weak to flip the detection answer, and it isn't specific to detection.
By the pre-registered letters: **Q1 = nothing, Q2 = generic "yes button", Q3 = random about as
strong** (both zero). The researcher's Q2 prediction was right; Q1 and Q3 were not.

## What was run

- **Model:** `meta-llama/Llama-3.2-3B-Instruct` @ `0cb88a4`; Study 14's prompts, tokens, batched
  last-token steering and sentence split.
- **Layer:** block 18 (`hidden_states[19]`), by the researcher's rule: no 3B logit-lens data existed
  (Study 11's lens was 8B only; Study 13 computed none), so the fallback, layer 18, applies.
- **Injection direction:** mean(injected) − mean(clean) at block 18 over Study 14's 100 calibration
  draws, unit-normalized (raw norm 0.639 = 0.039 × N). Typical norm **N = 16.21**. SHA-256
  `d98cb7e3…c734`.
- **Dose calibration (positive control only, 100 calibration-half prompts, 40 grid sizes):**
  "yes" was 0% up to 0.26 × N, 3% at 0.31 × N, **93% at 0.37 × N**, then 100% up to 2.4 × N (at
  2.9–5.7 × N every answer was "neither"). Because the curve jumps from 3% to 93% between two
  neighbouring grid points, the pre-registered rule gave **one dose for all three targets: s25 = s50
  = s90 = 0.3685 × N** (push norm 5.97, about 9× the raw injection footprint). The bars use the top
  dose, so scoring is unaffected; the intermediate dose points are lost.
- **Test:** 400 clean draws (test-half sentences), detection question and one control question each.
  Run time 4 h 40 min, one attempt. **Sanity:** all passed (last-token runs = full pass to
  ≤ 1.2 × 10⁻⁵; batched = single to 1.7 × 10⁻⁵, same greedy token).

## Results (at the dose 0.3685 × N)

| condition | "yes" % (dose 0 → dose) | "neither" % | yes − no logit: before → shift [95% CI] |
|---|---|---|---|
| (a) detection + injected direction | 0 → **0** | 0 | −2.82 → **+1.46** [1.46, 1.47] |
| (b) control questions + injected direction | 0 → **11.0** | 0.2 | −3.13 → **+1.89** [1.86, 1.92] |
| (c) detection + 5 random directions | 0 → **0** | 0 | −2.82 → +0.20 (per direction −0.12 to +0.50) |
| (d) detection + (yes − no) direction | 0 → **96.0** | 0 | −2.82 → **+3.06** [3.05, 3.08] |

- **Validity:** V1 no broken doses; V2 control answered "no" in 100% of clean trials; **V3 positive
  control 96% ≥ 50% → valid, Q1–Q3 scored.**
- **The 11% on control questions is one question.** "Does a week have ten days?" flipped in 40/40
  trials; its clean yes − no margin was the smallest of the ten (−1.12 nats). The other nine
  flipped in 0–3 of 40. Every control question's logit moved by +1.6 to +2.6 nats.
- **Nearest the detection answer came:** after the push, its yes − no logit was −1.36 on average
  (highest −0.64), still on the "no" side for every trial.
- **Alignment:** the injected direction is almost orthogonal to the (yes − no) word direction
  (cosine 0.015; the five random directions −0.024 to +0.006). Its yes-lean doesn't come from
  pointing at the "yes" output words; it comes from how later layers process it.

## Pre-registered outcomes

| question | value at top dose [95% CI] | outcome | researcher |
|---|---|---|---|
| Q1: does "yes" rise on detection? | rise 0.0 [0.0, 0.0] | **c = nothing** | a ❌ |
| Q2: specific to detection? | detection 0.0, control 11.0 [8.0, 14.25]; diff −11.0 [−14.25, −8.0] | **b = generic "yes button"** (control rise ≥ 10, diff < 10). **The control CI crosses the 10-point boundary.** | b ✅ |
| Q3: is the direction special? | diff vs random 0.0 [0.0, 0.0] | **a = random about as strong** (both 0%) | b ❌ |

**Predictions: 1 / 3.**

## Interpretation

- **Pre-registered reading:** at a push size where the "yes"-word direction reliably produces "yes",
  the injection's answer-position direction doesn't produce a detection report. Whatever "yes" it
  produces shows up on unrelated questions instead. It is **not a detection-specific button**.
- **Caveat on Q2:** "generic yes button" rests on one control question near its tipping point,
  and the control CI crosses the bar. The more robust evidence for "generic, not detection-specific"
  is the logit shift, which is as large or larger for every control question as for the detection
  question.
- **Descriptive, not pre-registered (Q3 in logit terms):** on the greedy answer, random and injected
  directions tie at zero. On the logit, the injected direction pushes toward "yes" about 3–14× more
  than any random direction (+1.46 vs −0.12 to +0.50). So the direction is "special" in strength,
  but what it carries is a **general yes-lean**, not detection.
- **Fits earlier results:** Study 11 found only a small, sub-threshold detection-specific lean
  in 8B models; Study 12 found that refusal ablation adds a yes-bias rather than detection; Hasik
  (2026) reports that gate features are shared with general yes/no answering. Our 3B result is the
  causal version of the same picture: the decodable injection footprint, pushed in by hand, acts on
  "yes/no answering" in general, not on "was something injected?".
- **Not shown:** a mechanism; anything about larger models; anything about other layers, positions
  or wordings; consciousness or experience.

## Limitations

- **One dose in practice:** the positive control's dose curve is a step, so s25 = s50 = s90.
- One layer (18, by the fallback rule), one position (the answer position), one model (3B), one
  detection wording, a mean-difference direction (not patched states).
- The greedy measure depends on how close each question sits to its tipping point; the logit
  shifts are the more sensitive view.
- The positive control is an output-word direction; matching its norm doesn't guarantee that other
  directions are equally "strong" in their effect on later layers.
