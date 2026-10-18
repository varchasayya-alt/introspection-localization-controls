# Gate 1 + Gate 2 report — positive control and cheap-trick controls

> **Update (Study 6, 2026-10-18):** this result holds for *our* protocol (our prompt, injection span
> and strength). Under the authors' protocol, Tülu-3-8B localizes 93% with the text intact
> ([`study6_report.md`](study6_report.md)), so this failure is protocol-specific, not a model property.

Model: `allenai/Llama-3.1-Tulu-3-8B` @ `6669437`. Run: 2026-10-15, 01:52 → 11:32, on AC power,
one uninterrupted run (no crashes, no restarts). Plan: [`gate1_plan.md`](gate1_plan.md).
Predictions: [`../predictions.md`](../predictions.md) (committed before the run).
All numbers: [`../results/gate1/summary.md`](../results/gate1/summary.md) (made by the analysis
script that was committed *before* any result existed); per-trial data in `results/gate1/*.jsonl`.

---

## The short version

1. **Gate 1 PASSED.** When we inject a concept into one of ten sentences, Tülu-3-8B points to
   the injected sentence **53.7%** of the time on 300 fresh trials (161/300; chance is 10%;
   95% range 48–59%). So the measurement "works": the model *can* localize the injection.
2. **Gate 2 FAILED.** The localization is fully explained by the injected sentence **reading
   differently**. When we let the injected model copy out the ten sentences, the injected one
   came back garbled every time (0 of 150 exact). A clean copy of the model that saw *only that
   copied text* found the odd sentence **74.7%** of the time, *better* than the injected model
   itself (45.3% on the same 150 trials).
3. **Random vectors work just as well as concepts** (53.3% vs 53.7%), so whatever the model
   detects is not specific to the concept.
4. **Stop rule triggered (as pre-registered):** we stop calling this "introspection". Running the
   training-stage ladder with this setup would measure "how much the injection garbles the text",
   not "noticing", so Gate 3 does **not** go ahead as planned.

---

## What happened, step by step

### Sanity checks — passed
The prompt matched the model's own chat format; "1"…"10" were single tokens; injecting zero
changed nothing (difference exactly 0); the speed-up that reuses the start of the prompt gave the
same answer scores as a full pass (max difference 0.00002).

With **nothing injected**, the model names sentence 5, 5, 4, 10, 7, 7 for the six sentence sets
(it has favourite answers). Because the target cycles through all ten positions, a favourite
answer can only score 10% on its own.

### Phase A — calibration sweep (720 trials) — passed
Correct out of 30 (chance = 3):

| layer \ strength | 2 | 5 | 8 | 14 |
|---|---|---|---|---|
| 1 | 23 | 15 | 21 | 18 |
| 2 | 14 | **26** | 25 | 22 |
| 3 | 7 | 25 | 25 | 25 |
| 4 | 7 | 24 | 26 | 22 |
| 6 | 10 | 22 | 25 | 16 |
| 8 | 9 | 18 | 22 | 16 |

Almost every setting is far above chance. Layer 2, strength 5 was chosen by the pre-set rule
(tied with layer 4 strength 8; lower strength wins) — the same cell where *Detecting the
Disturbance* found its best result. For scale: the model's normal activity at layer 2 has size
about 2.4, so strength 5 is roughly **twice** the normal signal. That is a big push.

### Phase B — fresh test (300 trials) — passed
**161/300 = 53.7%** (pass bar 90/300). By concept: glacier 78%, tornado 74%, origami 64%,
lighthouse 52%, saxophone 48%, **cactus 6%** (cactus basically doesn't work).

**Something odd:** the model *never* answered "1" or "2" in all 300 trials, so it scored 0 when
the injection was in sentence 1 or 2 (positions 3–10 were 53–87%). We don't know why yet (the
first two sentences come right after the instructions). It means the highest possible score in
this setup was really 80%, not 100%.

*Follow-up check (added 2026-10-15, after the run, descriptive only):* this is a strong
**default against answering 1 or 2**, not a broken prompt. With nothing injected the model never
names 1 or 2 either. It does say "1" or "2" when the evidence is strong, and then it is almost
always right: 23 times across the Phase A settings, 11 times for the clean reader and 20 times
in the typed-word condition. When the target was sentence 1 or 2 at layer 2 α 5, the correct
digit ranked about 8th of 10.

### Gate 2 — the cheap-trick controls

| condition | accuracy | what it tells us |
|---|---|---|
| Injected model (the 150 trials the reader also did) | 45.3% | the thing we're testing |
| **C1: clean reader, given only the text the injected model "perceived"** | **74.7%** | the text change alone is enough — and works *better* |
| C3: random vectors of the same size | 53.3% | not concept-specific |
| C6: concept word literally typed into the sentence, no injection | 89.3% | what plain reading can do |

- **C2 (copy fidelity):** the injected sentence was copied back exactly in **0 of 150** trials
  (on average only 44% of its words were right); the other nine sentences were copied perfectly.
  The concept leaks into the words: with *lighthouse* injected, "The store was crowded…" came
  back as "The **lhouse** crowded…", and many sentences started with "The **l**…".
- **C7 (how much the injection shifts the model's word predictions):** concept vectors shift
  them ~4× more than random vectors (KL 8.5 vs 2.0), yet both give the same localization score.
  Within each condition, bigger shifts did *not* give higher accuracy (if anything, lower).
  These are descriptive only.
- **C4 (was the concept already related to the sentence?):** sentences least similar to the
  concept had lower accuracy (37%) than the others (62%). Descriptive only.

**Pre-registered rule:** pass needed injected − reader ≥ 15 points *and* reader < 30%.
Result: −29.4 points, reader 74.7% → **FAIL.** Label from C3: **"disturbance localization"**.

---

## Your predictions, scored with the rules fixed before the run

| # | Your answer | What happened | Correct? |
|---|---|---|---|
| Q1 | "c" — passes the 30% bar | 53.7% (and best practice cell 87%) | ✅ yes |
| Q2 | "a" — random vectors work just as well | 53.3% vs 53.7% | ✅ yes |
| Q3 | "b" — reader does *not* match; real noticing | reader 74.7%, *above* the injected 45.3% | ❌ no |
| Q4 | "c" — works best when injected late | best cell was layer 2 (only layers 1–8 were tested) | ❌ no |
| Q5 | "b" — sentence copied back exactly about half the time | 0% exact | ❌ no |
| Q6/Q7 | "b" — mostly noticing, but weak | Gate 2 failed: it looks like reading changed text | ❌ no |

2 of 6 correct. Your Q2 call was the important one: you predicted that random vectors would work
as well as concepts, and they did.

Side note on Q4 (not a re-score — the rule says "no"): within the layers we tested, localization
was well above chance at *every* layer including 6 and 8, so it is not limited to the very first
layers — but the best was early.

---

## What this means, in plain language

Imagine slipping a smudge into one line of a page and asking someone which line "felt different".
If they point to the smudged line, that doesn't show they have some inner sense — they may just
have *seen the smudge*. That is what happened here. The injection at layer 2 is strong enough to
change how the model reads the sentence (the copy test shows the words themselves get scrambled),
and anyone reading the scrambled text — even a fresh copy of the model with nothing injected —
can spot the odd sentence, more easily than the injected model does.

So the "88%"-style localization results for this model size and these settings **do not by
themselves show introspection**. This is the gap we flagged in Gate 0: neither localization
paper ran a text-only reader control.

## Caveats (honest limits)

- **The copy test may exaggerate the garbling.** We took the model's single best guess for each
  word while showing it the correct previous words. A sentence can look more scrambled this way
  than in what the model "perceives". But even so, the rule was fixed in advance, and the
  typed-word control (89%) shows plain reading is a very strong route.
- **One model, one cell, early layer.** We tested Tülu-3-8B at layer 2, strength 5. A smaller
  push, or later layers, might leave the text intact; that was not tested here.
- **Answers "1" and "2" were never given**, which caps accuracy at 80% and is unexplained.
- **Cactus** barely worked (6%), so results depend a lot on the concept.

## What the rules say happens next

Gate 2 FAIL → **stop the "introspection" framing** (pre-registered). The planned Gate 3 ladder
(base → SFT → DPO → final) is **not** run with this setup, because it would only measure how much
the injection scrambles the text.

Any next step is a **new** question that needs its own pre-registration. Options to discuss:
1. **Write this up as a negative / methods result.** Localization at the published setting is
   explained by text change; the text-only reader control should be standard.
2. **Search for a "text-intact" regime.** Find layer/strength settings where the copy test shows
   the sentence intact (e.g. ≥ 90% exact copies) *and* the reader is near chance, then ask
   whether the model still localizes there. Only if that works would a ladder comparison make
   sense.
3. **Investigate the "never answers 1 or 2" bias** before any further localization runs.
