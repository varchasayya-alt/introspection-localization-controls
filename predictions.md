# Predictions (pre-registration)

Predictions are recorded **verbatim** from the researcher, committed, and only then is the
corresponding experiment run. Nothing in this file is edited after the run it refers to.

## Gate 1 — positive control (allenai/Llama-3.1-Tulu-3-8B, sentence localization)

Recorded 2026-10-15, **before** any Gate 1 data was collected (no model weights downloaded yet).

### Original questions (from the Gate 0 report)

1. **Calibration sweep:** best cell accuracy (any layer × strength), in %?
2. **Held-out test** at the pre-selected cell (300 trials, chance 10%): accuracy in %? Will it
   pass the ≥ 30% rule (yes / no)?
3. **Random-vector control** (same size, random direction): accuracy in %?
4. **Text-only reader control** (a clean model sees only the text the injected model
   "perceived"): accuracy in %? Will it come within 10 points of the injected model?
5. **Layer pattern:** will localization only work at early layers (≤ 4 of 32)?
6. **Transcription:** in what fraction of trials will the injected sentence be copied back
   *exactly*?
7. **Overall:** if Gate 1 passes, will it look like "noticing" or like "reading changed text"?

### Researcher's predictions — verbatim

The researcher answered a simplified multiple-choice version (their teacher rephrased the 7
questions as a "whisper game"). Letter answers and mapping, exactly as given:

> Q1 (fresh-test accuracy): "c" = 3 or more in 10, so it PASSES the 30% bar. (Best practice setting: not predicted separately; implied >=30%.)
> Q2 (random-vector accuracy): "a" = random vectors work just as well as real concepts.
> Q3 (clean text reader): "b" = the reader does NOT match the injected model, i.e. not within 10 points; real noticing.
> Q4 (layers): "c" = works best when injected LATE in the network, not only early (<=4).
> Q5 (injected sentence copied back exactly): "b" = about half the time.
> Q6/Q7 (noticing vs reading changed text, if it passes): "b" = a little; mostly noticing, but weak.

Mapping to the original questions: whisper-game Q1 → original Q2 (and Q1 implied ≥ 30%);
Q2 → original Q3; Q3 → original Q4; Q4 → original Q5; Q5 → original Q6; Q6/Q7 → original Q7.

### How each prediction will be scored (set by Claude before the run, not by the researcher)

These rules are written down now so the predictions can't be re-judged after seeing results.

| Prediction | Counts as correct if… |
|---|---|
| Q1 "c" | Phase B accuracy ≥ 30% (≥ 90 / 300). Implied part: best Phase A cell ≥ 30% (≥ 9 / 30). |
| Q2 "a" | Random-vector accuracy (C3) is within 10 points of concept accuracy at the same cell (same threshold as the C3 labelling rule). |
| Q3 "b" | Perceived-text reader (C1) accuracy is **more** than 10 points below the injected model's accuracy on the same 150 trials. |
| Q4 "c" | The best Phase A cell is at a layer > 4 (i.e. layer 6 or 8). **Limitation:** the approved sweep only covers layers 1–8 of 32 (the first quarter of the network), so truly late layers are *not tested* in Gate 1; this prediction can only be checked within that range. |
| Q5 "b" | The injected (target) sentence is copied back exactly (every token is the model's top prediction) in 35–65% of the C1 trials. |
| Q6/Q7 "b" | Gate 1 passes **and** Gate 2 passes ("mostly noticing") **and** Phase B accuracy is below 50% ("weak"). Not scored if Gate 1 fails. |

---

## Study 2 — text-intact search (plan: `gates/text_intact_plan.md`)

Scoring rules written 2026-10-15, **before** the researcher's Study 2 predictions were recorded
and before any Study 2 data exists. Each rule says which answer letter the outcome corresponds
to, so a prediction is correct if its letter matches.

Definitions used below:
- *Text-intact setting* = clean reader ≤ 4/30 in practice (all 30 reader trials run; a setting
  whose reader trials stopped early at 5 correct is not intact).
- *Best text-intact setting* = the setting the selection rule picks among text-intact settings
  (highest injected accuracy; ties → lower r, then lower layer), pooled over all practice
  attempts that were run.

| # | Question | Outcome → letter |
|---|---|---|
| P1 | How many of the 15 attempt-1 settings are text-intact? | a = 0; b = 1–5; c = 6–10; d = 11–15 (attempt 1 only) |
| P2 | Injected accuracy at the best text-intact setting (practice) | a ≤ 4/30 (about chance); b = 5–7/30; c = 8–15/30; d ≥ 16/30. If no setting is text-intact, P2 is "not scorable". |
| P3 | Does practice find a passing setting? | a = attempt 1 passes; b = attempt 1 fails and attempt 2 passes; c = both fail (stop) |
| P4 | Fresh test outcome | a = fresh test passes all three rules; b = injected < 50/200 (if several rules fail, b takes precedence); c = injected ≥ 50/200 but the reader > 30/200 or the margin < 10 points; d = fresh test not run |
| P5 | Where is the best text-intact setting? | a = layer 2 or 4; b = layer 8; c = layer 12 or 16; **d = no text-intact setting reaches 8/30 injected** (includes the case of no text-intact settings). If d applies, a–c are wrong. |
| P6 | Exact copy-back of the target at the chosen setting | Measured on the practice copy trials of the best text-intact setting: a < 25%; b = 35–65%; c > 75%; 25–35% and 65–75% match no letter. Not scorable if no setting is text-intact. |
| P7 | Sentences 1–2 at text-intact settings | Pool all injected practice trials with target in position 1 or 2 at every text-intact setting (plus fresh-test trials if run): a = 0 or 1 correct (essentially never); b = 2 or more correct. Not scorable if no setting is text-intact. |

### Researcher's Study 2 predictions — verbatim

Recorded 2026-10-15, after the scoring rules above were committed (229a20d) and before any
Study 2 data. The researcher answered a simplified version (their teacher rephrased P1–P7 in the
same order). Exactly as given:

> P1: c (about half of the 15 settings are text-intact)
> P2: a (at chance at the best text-intact setting)
> P3: c (no passing setting; we stop and write up). [I first answered "b", then changed it to "c" before the run because "b" contradicted my P2/P4/P5.]
> P4: d (we won't reach the fresh test)
> P5: d (no text-intact setting finds the sentence)
> P6: b (injected sentence copied back exactly about half the time)
> P7: b (sentences 1-2 sometimes found)

Researcher approvals recorded with these predictions: design approved; fresh-test reader
≤ 15% condition added; 200 random-vector trials at the selected setting included.

## Study 3 — replicate the "Detecting the Disturbance" headline + text control (plan: `gates/study3_plan.md`)

Scoring rules written 2026-10-16, **before** the researcher's Study 3 predictions were recorded
and before any Study 3 data. Counts are out of the 400 trials unless stated. "Injected" = the
injected model's accuracy; "reader" = the clean reader's accuracy on the same 400 trials.

| # | Question | Outcome → letter |
|---|---|---|
| Q1 | Injected accuracy | a ≥ 75% (≥ 300); b = 50–75% (200–299); c = 25–50% (100–199); d < 25% (< 100) |
| Q2 | Reader accuracy | a ≤ 15% (≤ 60); b = 61–160 (15–40%); c = 161–280 (40–70%); d ≥ 281 (> 70%) |
| Q3 | Main test | d if R1 fails (injected < 200); otherwise a = reader > injected (more correct trials); b = injected − 40 ≤ reader ≤ injected; c = reader < injected − 40 |
| Q4 | Exact copy-back of the injected span ("SENTENCE i: …" reproduced exactly) | a < 25%; b = 35–65%; c > 75%; 25–35% and 65–75% match no letter |
| Q5 | Vector type (200 trials each) | a = "avg" ahead by > 10 points (> 20 trials); b = "last" ahead by > 10 points; c = otherwise |
| Q6 | Injected accuracy vs Tülu Gate 1 (53.7%) | a > 63.7%; b = 43.7–63.7% inclusive; c < 43.7% |
| Q7 | Injected-model accuracy when the target is sentence 1 or 2 (80 trials) | a ≤ 2 correct ("never or almost never"); b ≥ 3 correct |

### Researcher's Study 3 predictions — verbatim

Recorded 2026-10-16, after the scoring rules above were committed (b1b0336) and before any
Study 3 data. The researcher's teacher rephrased Q1–Q7 simply, in the same order. Exactly as given:

> Q1: b (50-75%, replication passes)
> Q2: c (reader 40-70%)
> Q3: a (the reader beats the injected model)
> Q4: a (injected span almost never copied back exactly)
> Q5: b ("last-token" vectors localize better)
> Q6: b (within 10 points of Tulu's 53.7%)
> Q7: a (sentences 1-2 never or almost never found)
> Note: I first answered Q1 "c" and Q4 "c", then changed them to "b" and "a" before the run because they contradicted Q3 and Q2.

Researcher approvals recorded with these predictions: design approved (including the
label-included injection and both vector types); download of the authors' saved vectors
(safe mode + match check); download of meta-llama/Llama-3.1-8B-Instruct.

## Study 4 — sentence-only vs label-only injection (plan: `gates/study4_plan.md`)

Scoring rules written 2026-10-17, **before** the researcher's Study 4 predictions were recorded
and before any Study 4 data. Counts are out of 200 trials per arm unless stated. "Exact-copy
trials" = arm-A trials whose injected span is copied back exactly (with the D3 copy fix).

| # | Question | Outcome → letter |
|---|---|---|
| P1 | Arm A injected accuracy | a ≥ 75% (≥ 150); b = 50–75% (100–149); c = 25–50% (50–99); d < 25% (< 50) |
| P2 | Arm A main test | d if fewer than 40 exact-copy trials; otherwise a = injected ≥ 50% **and** reader ≤ 15% in those trials; b = injected < 50% (whatever the reader does); c = injected ≥ 50% but reader > 15% |
| P3 | Arm B injected accuracy | same bins as P1 |
| P4 | Arm A vs arm B (injected) | a = A − B > 20 trials (> 10 points); b = B − A > 20 trials; c = otherwise |
| P5 | Arm C (random) vs arm A (injected) | a = within 20 trials (≤ 10 points); b = C lower by > 20 trials; c = C higher by > 20 trials |
| P6 | Arm A exact copy-back of the injected span | a < 25%; b = 35–65%; c > 75%; 25–35% and 65–75% match no letter |
| P7 | Overall conclusion (table in plan §4) | a = A passes and B < 50%; b = A passes and B ≥ 50%; c = A does not pass and B ≥ 50%; d = A does not pass and B < 50%. "Not testable" counts as "does not pass". |

### Researcher's Study 4 predictions — verbatim

Recorded 2026-10-17, after the scoring rules above were committed (ac653cb) and before any
Study 4 data. The researcher's teacher rephrased P1–P7 simply, in the same order. Exactly as given:

> P1: a (arm A: 75% or more)
> P2: a (arm A exact-copy test passes: model >= 50%, reader <= 15%)
> P3: d (arm B label-only: below 25%)
> P4: a (A beats B by more than 10 points)
> P5: b (random vectors clearly worse than arm A)
> P6: c (arm A exact copies: more than 75%)
> P7: a (conclusion: noticing - A passes, B below 50%)

Researcher approvals recorded with these predictions: design approved (arms A, B, C, the rules,
fresh draws and the D3 copy fix); arm B **with** copy+reader (~9 hours).

## Study 5 — question-specificity control (plan: `gates/study5_plan.md`)

Scoring rules written 2026-10-17, **before** the researcher's Study 5 predictions were recorded
and before any Study 5 data. Counts are out of 200 trials per arm. "Pull" (points) = 100 ×
(trials answered with the injected sentence − trials whose no-injection answer for that draw and
question is the injected sentence) / 200 (= accuracy − 10 points). Q0 = original question,
Q1 = "longest", Q2 = "like most"; "concept"/"random" = vector type of the arm.

| # | Question | Outcome → letter |
|---|---|---|
| P1 | Q0 concept accuracy | a ≥ 85% (≥ 170); b = 60–85% (120–169); c < 60% (< 120) |
| P2 | Q1 concept pull | a < 20 points; c ≥ 20 points **and** ≥ half of Q0-concept pull; b = otherwise |
| P3 | Q2 concept pull | same bins as P2 |
| P4 | Overall conclusion (plan §4) | a = spotlight; b = question-specific; c = mixed; "not scorable" if the validity check fails (Q0-concept pull < 50 points) |
| P5 | Q0: random vs concept accuracy | a = within 20 trials (≤ 10 points); b = random lower by > 20 trials; c = random higher by > 20 trials |
| P6 | Q1: random vs concept accuracy | same bins as P5 |
| P7 | Q1 correctness (answer = the truly longest sentence) with push vs without | with push = share of the 200 Q1-concept trials answered with the longest sentence; without = share of the 20 no-injection Q1 answers that are the longest sentence. a = with push ≥ without − 10 points; b = with push lower by more than 10 points |

### Researcher's Study 5 predictions — verbatim

Recorded 2026-10-17, after the scoring rules above were committed (ab0e61c) and before any
Study 5 data. The researcher's teacher rephrased P1–P7 simply, in the same order. Exactly as given:

> P1: a (Q0: 85% or more)
> P2: a (Q1 "longest": small pull, under 20 points)
> P3: a (Q2 "like most": small pull)
> P4: b (overall: question-specific)
> P5: a (random vs concept on Q0: within 10 points)
> P6: a (random vs concept on Q1: within 10 points)
> P7: a (the model still picks the truly longest sentence about as often with the push on)

Researcher approvals recorded with these predictions: design approved (neutral intro for Q1/Q2,
6-word + 4-word draws, the rules, arms 1-5); optional arm 6 (random vectors on Q2) included.

## Study 6 — Tülu-3-8B with the Study 4 arm-A protocol (plan: `gates/study6_plan.md`)

Scoring rules written 2026-10-17, **before** the researcher's Study 6 predictions were recorded
and before any Study 6 data. Counts are out of 200 trials per arm. "Exact-copy trials" = arm-A
trials whose copied target sentence is exactly right (D3 fix). "Relative size" = 5 / average
residual-stream norm at the output of layer 2 over sentence-word tokens (plan §4).

| # | Question | Outcome → letter |
|---|---|---|
| P1 | Arm A injected accuracy | a ≥ 75% (≥ 150); b = 50–75% (100–149); c = 25–50% (50–99); d < 25% (< 50) |
| P2 | Number of arm-A exact-copy trials | a ≥ 150; b = 40–149; c < 40 |
| P3 | Arm A main test | a = PASS; b = FAIL with injected < 50% in exact-copy trials; c = FAIL with injected ≥ 50% but reader > 15%; d = NOT TESTABLE (< 40 exact-copy trials) |
| P4 | Arm C (random) vs arm A (injected) | a = within 20 trials (≤ 10 points); b = C lower by > 20 trials; c = C higher by > 20 trials |
| P5 | Injection size: Tülu's relative size ÷ Llama's | a = 0.9–1.1 (about the same); b > 1.1 (relatively bigger in Tülu); c < 0.9 (relatively smaller in Tülu) |
| P6 | Mean cosine, Tülu vectors vs the authors' Llama vectors (20 pairs) | a ≥ 0.9; b = 0.5–0.9; c < 0.5 |

If a sanity check stops the study before trials, P1–P4 are "not scorable"; P5 and P6 are scored
if their measurements were made.
