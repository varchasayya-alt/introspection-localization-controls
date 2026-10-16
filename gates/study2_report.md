# Study 2 report — text-intact search

Model: `allenai/Llama-3.1-Tulu-3-8B` @ `6669437`. Run: 2026-10-15 22:19 → 2026-10-16 12:10, AC
power, one uninterrupted run. Plan: [`text_intact_plan.md`](text_intact_plan.md). Predictions
and scoring rules: [`../predictions.md`](../predictions.md) (rules `f6eb6c3`, predictions
`3bdd3f2`, both before any data). Tables: [`../results/study2/summary.md`](../results/study2/summary.md)
(made by the analysis script committed before the run).

This was a **new** pre-registered question. Gate 2's FAIL stands: at the Gate 1 setting,
localization is explained by text corruption.

---

## The short version

**Question:** is there an injection setting where the text stays intact (a clean reader of the
copied text is at chance) but the injected model still finds the injected sentence?

1. **Practice PASSED.** 11 of 15 weak/late settings kept the text intact. At the best one
   (layer 8, half-strength) the injected model found the sentence 14/30 times while the reader
   was at 3/30.
2. **Fresh test FAILED.** On new concepts and sentences: injected **36/200 (18%)**, reader
   **20/200 (10%)**, random vectors 28/200 (14%). The rules needed injected ≥ 50/200 and a gap of
   ≥ 20; the gap was 16. The text *was* intact (198/200 exact copies, reader at chance).
3. **Answer:** in this model, at the settings we tried, we did **not** find localization that
   survives with the text left intact. Per the stop rule, we stop and write up.

---

## What happened

### Practice (attempt 1 of max 2) — passed

Strength is relative to normal activity at that layer (¼×, ½×, 1×). Chance = 3/30.

| layer | ¼× | ½× | 1× |
|---|---|---|---|
| 2 | 4/30 ✔ | 4/30 ✔ | 21/30 ✘ |
| 4 | 4/30 ✔ | 7/30 ✔ | 22/30 ✘ |
| 8 | 9/30 ✔ | **14/30 ✔** | 20/30 ✘ |
| 12 | 9/30 ✔ | 9/30 ✔ | 8/30 ✘ |
| 16 | 6/30 ✔ | 4/30 ✔ | 5/30 ✔ |

✔ = text-intact (reader ≤ 4/30; every one of these had the reader at exactly 3/30 and 28–30
exact copies). ✘ = not intact (the reader found the sentence 5 times within the first 7–13
trials; copies came back changed).

The pattern repeats Gate 1: **full strength scrambles the text** at layers 2–12, and that's where
the high scores are. At half strength, layer 8 still scored 14/30 with perfect copies, and was
chosen by the pre-set rule. Attempt 2 was not needed.

### Fresh test (new material, 200 trials) — failed

| condition | accuracy |
|---|---|
| Injected (concept) | **18.0%** (36/200; 95% range 13–24%) |
| Clean reader of the copied text | 10.0% (20/200) |
| Injected (random vector, same size) | 14.0% (28/200) |

| rule | needed | got | |
|---|---|---|---|
| injected | ≥ 50/200 | 36 | ✘ |
| reader (text intact) | ≤ 30/200 | 20 | ✔ |
| gap injected − reader | ≥ 20 | 16 | ✘ |

The text stayed intact (198/200 copied back exactly), but the injected model rarely found the
sentence. The practice score (14/30) did not carry over: it was the best of 15 small (30-trial)
practice scores, and the best of many noisy scores tends to be higher than the true value.

---

## The "default answer" problem (descriptive)

With nothing injected, the model has a favourite answer for each sentence set: "5" for the
practice set, "3" for fresh set 1, and "6" for fresh sets 2–4. At weak strengths it almost always
gives that favourite anyway: "6" was the answer in 157 of 200 fresh trials, and "5" in 292 of 450
practice trials.

Because the target cycles through all positions, a favourite answer is right exactly once per
cycle. That is why the reader scored **exactly** 3/30 at every intact practice setting and
20/200 in the fresh test: with intact text it just gives its favourite. So the honest baseline is
"the favourite answer", and the injected model's 36/200 is only 16 above it.

**The "never answers 1 or 2" bias** (checked as you asked): it affected this search fully. At
every text-intact setting, and in the whole fresh test, the injected model got **0** trials right
when the target was sentence 1 or 2 (it answered "1" or "2" only 1 time in 450 practice trials and
never in the fresh test). So the practical ceiling was 80%, and weak injections never overcame
this default. It did not decide the result: the failure was far below the ceiling.

### Exploratory only (noticed after seeing the data; not a result)

In the fresh test, the injected model moved away from its favourite answer in 34 of 200 trials,
and in **18 of those 34** it moved to the injected sentence. If its moves were random, about 1 in 9
would land on the right sentence. This hints at a weak signal hiding under a strong habit. But this
analysis was chosen *after* looking at the data, so it can't change the pre-registered verdict. At
most it is an idea for a new, pre-registered study.

**Same check on the random vectors (added 2026-10-16, also exploratory):** with a random direction
of the same size, the model moved off its favourite answer in 25 of 200 fresh trials, and **11 of
those 25 (44%)** landed on the injected sentence. Concept vectors gave 18 of 34 (53%). So the
"occasional move lands on the target" pattern is **not concept-specific**: any push of this size
sometimes nudges the answer toward the pushed sentence. That matches Gate 1, where random vectors
localized as well as concepts. In practice, at layer 8 ½×, the concept vectors moved the answer in
16 of 30 trials and 11 of those landed on the target.

---

## Your predictions, scored with the rules committed before you answered

| # | Your answer | What happened | Outcome letter | Correct? |
|---|---|---|---|---|
| P1 | c — about half text-intact | 11 of 15 were intact | d | ❌ |
| P2 | a — at chance at the best intact setting | 14/30 at layer 8, half strength | c | ❌ |
| P3 | c — no passing setting; stop | passed on attempt 1 | a | ❌ |
| P4 | d — won't reach the fresh test | reached it; injected below 50/200 | b | ❌ |
| P5 | d — no intact setting finds the sentence | layer 8 reached 14/30 in practice | b | ❌ |
| P6 | b — exact copy about half the time | 30/30 exact (100%) | c | ❌ |
| P7 | b — sentences 1–2 sometimes found | never found (0) | a | ❌ |

0 of 7 by the letter rules. Your **overall** expectation, that text-intact localization would
not hold up, matched the final outcome. It failed one step later than you predicted: practice
passed, then the fresh test failed. The rules score each step, so the letters don't match.

---

## Caveats

- **One model** (Tülu-3-8B), one prompt format, 15 settings, layers 2–16 only.
- **Small practice samples** (30 trials per setting) make the "best of 15" pick optimistic. That
  is exactly why the fresh test exists, and it did its job.
- **Accuracy is a weak measure when the model has a strong favourite answer.** At weak strengths,
  most of its score is the favourite answer. A future design could shuffle sentence order per
  trial, or score how much the injection *moves* the answer away from the no-injection favourite.
  That would need its own pre-registration.
- The copy test uses the model's best guess for each word. At intact settings it copied perfectly,
  so this caveat matters less here than in Gate 2.

## Where the project stands

| Step | Result |
|---|---|
| Gate 1 — positive control | PASS: 53.7% localization at layer 2, α 5 |
| Gate 2 — cheap tricks | FAIL: explained by text corruption (reader 74.7% > injected 45.3%) |
| Study 2 — text-intact search | Practice PASS, fresh test **FAIL** (18% vs reader 10%; needed ≥ 25% and a ≥ 10-point gap) |

Together: for this 8B model, **we found no evidence of sentence localization that doesn't come
from the text itself being changed.** The training-stage ladder question can't be answered with
this task until a version of the task is shown to measure something other than text change. Per
the stop rule, this study ends here, and next steps are your decision.
