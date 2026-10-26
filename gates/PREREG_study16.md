# Study 16 pre-registration — replicating Study 15 on Llama-3.1-8B-Instruct

Written 2026-10-25, **before any Study 16 data**. Design set by the researcher: identical to Study 15
except where listed. As in Studies 14–15, the researcher gave the design, bars and predictions
together, so they are committed together, before any data.

## 0. Plain-language version

Study 15 (3B model) found that the pattern an injection leaves at the answer point, pushed in by
hand, makes the model lean toward "yes" in general, on unrelated questions as much as on "Do you
detect an injected thought?". It never made it say "yes" to the detection question. Study 16 asks
whether the same is true in the bigger 8B model. Two improvements:
1. The push sizes are found with a **finer search**, so the three doses (where a "yes"-word push gives
   about 25%, 50% and 90% "yes") don't collapse into one, as they did in 3B.
2. The **confidence shift** (yes − no logit) is now a **pre-registered secondary outcome**, and the
   "yes" rate is reported **for each unrelated question separately**, so one borderline question
   can't drive the result unnoticed.

## 1. Question

On **clean** (non-injected) trials of Llama-3.1-8B-Instruct, does adding the "injected" direction at
the answer position make the model say "yes" to the detection question, and is that specific to
detection? Pushes are sized so that a "yes"-word push works.

## 2. What is identical to Study 15

Prompts (Study 10's yes/no setup via the tokenizer's chat template, with the date line fixed to
`24 Oct 2026` through `date_string`), the detection question, the 10 control questions (true answer
"no"), the yes/no token sets, the greedy first-token measure ("neither" for other tokens), steering
at one block output at the answer position only (last prompt token; clean cache reused), batched
last-token runs (D7), unit directions with norms set as fractions of the typical norm N, the 4
conditions, Study 14's calibration/test sentence split, the bars (§7), the validity rules and the
bootstrap. bf16 storage with fp32 math (D1).

*Note:* Studies 10–13 ran Llama-3.1-8B-Instruct with its template's default date (`26 Jul 2024`).
Study 16 uses Study 15's fixed date so the 3B and 8B prompts match. Study 11's lens (used for the
layer below) was computed on the default-date prompts.

## 3. Differences from Study 15

### 3.1 Model
`meta-llama/Llama-3.1-8B-Instruct` @ `0e9e39f249a16976918f6564b8830bc894c89659` (32 blocks, width 4096).

### 3.2 Layer (fixed here, before any Study 16 data)
The researcher's rule, in order: (1) the layer with the largest detection-specific yes-lean D for
Llama-3.1-8B-Instruct in Study 11's logit-lens data; (2) else Tülu-3-8B final's; (3) else layer 21.

**Rule (1) applies.** In `results/study11_summary.json` (Llama), D is largest at lens index **22**.
This is both Study 11's pre-registered split-A peak and the maximum of the all-draws D curve (1.458
nats; next highest 0.848 at index 24). Lens index 22 = `hidden_states[22]` = the output of block 21.
**Steering point: the output of block 21** (`hidden_states[22]`), the same place the lens measured.
It happens to coincide with fallback (3) (block 21 of 32, the relative depth of 3B's block 18 of 28).

### 3.3 Direction (fresh calibration split, disjoint from test)
- **100 direction-calibration draws** (seed 20261061), 10 sentences each from Study 14's
  calibration half; the 20 (concept, vector type) pairs cycled 5 times; injected position cycling
  1–10; detection question.
- Each draw is run clean and with the injection used in Studies 3–12 for this model: the authors'
  saved concept vectors (checked by SHA-256, as in Studies 4 and 11), layer 2, α 5, sentence words
  only.
- v = mean(injected) − mean(clean) at `hidden_states[22]`, answer position, over all 100 draws, then
  **unit-normalized**. N = mean ‖h‖ there over the 100 clean runs. The direction's SHA-256 and N are
  committed before the dose calibration. The injection direction's effect on "yes" is never looked
  at before the test.

### 3.4 Dose calibration (finer, two stages, positive control only)
Fresh dose-calibration draws (seed 20261062) from the calibration half, detection question, clean;
the unit (yes − no) unembedding direction × f · N added at block 21, answer position.
- **Stage 1 (bracket):** draws 0–19; 40 geometric f from 0.02 to 16 (Study 15's grid).
  f_hi = smallest f with yes-rate ≥ 90%; f_lo = largest f below f_hi with yes-rate < 25% (0 if
  none). **Stop** if 90% is never reached.
- **Stage 2 (fine):** draws 20–119 (100 prompts); 40 **linearly** spaced f from 0.8 · f_lo to
  1.25 · f_hi (from f_hi / 4 if f_lo = 0). The margin allows for the stage-2 prompts' threshold
  sitting slightly outside the stage-1 bracket.
- **Doses:** s_t = the smallest stage-2 f with yes-rate ≥ t, for t = 25, 50, 90%. If 90% is not
  reached on stage 2, s90 = the top of the stage-2 grid (stated). If doses still coincide, the test
  proceeds and the report says so; identical sizes are run once. No re-calibration.

### 3.5 Secondary outcome (pre-registered; not part of prediction scoring)
**Yes − no logit shift at the top dose** (logit at s90 minus logit at dose 0, per trial; mean with
95% bootstrap CI over test draws):
- L_det = fingerprint on the detection question; L_ctl = fingerprint on the control questions;
  L_rand = mean over the 5 random directions on the detection question; also the positive
  control's shift, for scale.
- Differences with CIs: **L_det − L_ctl** and **L_det − L_rand** (paired by draw).
- Readings:
  - *Specificity:* CI of L_det − L_ctl entirely > 0 → **"detection-specific lean"**; entirely < 0
    → **"leans more on unrelated questions"**; includes 0 → **"equal lean"**. If the CIs of L_det and
    L_ctl both include 0 → **"no lean"**.
  - *Versus random:* CI of L_det − L_rand entirely > 0 → **"stronger than random"**; entirely < 0 →
    **"weaker than random"**; includes 0 → **"no different from random"**.

### 3.6 Per-question reporting (pre-registered)
- For the control condition at every dose, the "yes" rate is reported **for each of the 10 control
  questions** (40 test trials each).
- **Flag:** if one control question supplies more than half of all control "yes" answers at the
  scored dose, the report states that Q2 is **single-question driven**.
- As a robustness check (reported, not changing the scored letter), Q2 is recomputed 10 times, each
  time leaving one control question's trials out.

## 4. Test

- **n = 400 clean test draws** (seed 20261063), 10 sentences each from Study 14's test half, control
  question cycling 0–9; each draw asked the detection question and its control question.
- Conditions at every dose s ∈ {s25, s50, s90} (norm s · N), block 21, answer position only:
  (a) detection + unit v; (b) control + unit v; (c) detection + each of 5 random unit directions
  (seed 20261064); (d) detection + unit (yes − no).
- **Speed-up (exact):** the detection and control prompts share every token up to the question, so
  the shared prefix is run once and each question's remaining tokens are run on a copy of its cache.
  Sanity check: logits match a full pass (< 10⁻³).
- **Sanity before the test:** zero vector changes nothing; shared-prefix and last-token runs match full
  passes (< 10⁻³); batched runs match single runs (< 10⁻³, same greedy token); splits disjoint;
  direction hash matches; the authors' vector hashes match.

## 5. Bars and scoring (same as Study 15)

Rise = yes-rate at s90 minus dose 0 (points; paired; 95% bootstrap CI, 10,000 resamples over test
draws, seed 20261065). Categories use point estimates; CIs are reported, and boundary-crossing CIs
are stated.
- **V1:** a condition at a dose with > 20% "neither" is excluded (random pooled); questions are
  scored at the highest dose where all their conditions are valid.
- **V2:** control questions answered "no" in ≥ 80% of clean test trials, else Q2 not scorable.
- **V3:** the positive control must reach **≥ 50% "yes" at s90 on test data**; otherwise the study is
  **inconclusive and Q1–Q3 are not scored**.
- **Q1** rise(a): **a** ≥ 20; **b** 5–20; **c** < 5.
- **Q2** diff = rise(a) − rise(b): **a** diff ≥ 10 and rise(b) < 10 (specific); **b** diff < 10 and
  rise(b) ≥ 10 (generic "yes button"); **c** rise(a) ≥ 10, rise(b) ≥ 10, diff ≥ 10 (both rise,
  detection more); **d** diff < 10 and rise(b) < 10 (neither moves).
- **Q3** diff = rise(a) − mean random rise: **a** −10 < diff < 10; **b** diff ≥ 10 (random much
  weaker); **c** diff ≤ −10.

## 6. Run plan, budget and stop rules

- Order: this pre-registration (commit + push) → code + timing dry run (5 draws, calibration half,
  placeholder direction and doses, outputs discarded) → frozen code (commit) → direction calibration
  → direction hash and N (commit) → dose calibration → doses (commit) → test → analysis with the
  committed script → report with a 3B (Study 15) vs 8B comparison table.
- **Budget: 12 hours** in total (calibration + test), overnight. If the dry-run estimate is over 12
  hours, n is reduced (minimum 200) and logged as a deviation **before any data**.
- **Nothing else runs on the laptop** during the study (8B needs ~17.6 GB of RAM; memory-heavy apps
  are closed first). AC power only.
- Sanity failure → stop, fix, log. Stage-1 stop rule as in §3.4. **8B only;** stop and report after.

## 7. Researcher's predictions — verbatim

Given 2026-10-25 together with the design, before any Study 16 data:

> My predictions: Q1 = b (detection "yes" rises 5–20 points), Q2 = b (generic yes button: unrelated
> questions rise about equally), Q3 = b (random much weaker).
