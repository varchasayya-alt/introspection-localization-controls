# Study 4 — noticing vs label-digit highlighting (APPROVED 2026-10-17)

Approved by the researcher on 2026-10-17: arms A, B and C, the rules, fresh draws, the D3 copy fix,
and arm B **with** copy+reader (~9 h). Code `scripts/study4.py`, analysis `scripts/study4_analyze.py`,
stimuli `stimuli/study4_stimuli.json` (20 draws, seed 20261018; random vectors seed 20261019;
SHA-256 of the 20 vector files).

**New pre-registered study.** Study 3's results stand. The hypothesis was logged by the researcher
during Study 3 (`study3_plan.md` §7, 2026-10-16 16:04).

**Question:** in Study 3, the injection covered "SENTENCE N: <sentence>", including the label digit
N, the very number the model answers with. Does the model still find the injected sentence when we
push only on the sentence words (and the text stays intact)? And does pushing only on the label
already explain the result?

## 1. Same setup as Study 3

Model `meta-llama/Llama-3.1-8B-Instruct` @ `0e9e39f`; layer 2 (their convention); α = 5; their
exact localization prompt; their 20 saved layer-2 vectors (10 concepts × "avg"/"last"),
checked by SHA-256 against the files used in Study 3. Answer = argmax over "1"…"10".

**Fresh sentence draws:** 20 new draws of 10 sentences from their pool (seed 20261018), one per
concept × vector type. **The same 20 draws are used in every arm**, so the arms are compared on
identical sentences.

## 2. Arms (200 trials each = 10 concepts × 2 vector types × 1 draw × 10 positions)

| arm | what gets the push | copy-back + clean reader? | time |
|---|---|---|---|
| **A — sentence only** | the sentence's words only; **not** " SENTENCE", " ", "N", ":" | yes (same trials) | 0.8 h + 2.8 h |
| **B — label only** | only the label tokens " SENTENCE N:"; **not** the words | yes (same trials; recommended, see §6) | 0.8 h + 2.8 h |
| **C — random vectors, sentence only** (your optional arm) | as A, but a random direction of the same size (20 random unit vectors, seed 20261019, one per draw) | no (injected only) | 0.8 h |

Token check from the tokenizer: "SENTENCE 3: The grass grows tall." is ' SENT', 'ENCE', ' ', '3',
':', ' The', ' grass', ' grows', ' tall', '.'. Arm A pushes on ' The' … '.'; arm B on
' SENT' … ':'.

## 3. Fix for the copy-scoring quirk (logged before running)

In Study 3 the copied span 2 always came back as "ENCE 2: …". Diagnosis: span 2's first token
(' SENT') is predicted from the full stop at the end of sentence 1. That is the first place in the
copy where the model guesses how the list continues, and it guesses a line break, which vanishes
when the text is trimmed. **Fix:** when building the "perceived" text, the **first token of each
copied span** (' SENT', which carries only the list formatting) is taken from the real text. Every
other token, including the label digit and ':', still uses the model's own prediction. This is
the same for all arms, so no arm gets an advantage.

**New sanity check S7:** a clean copy (no injection) must reproduce all 10 spans exactly in at least
19 of the 20 draws. If not, stop and report. That would be a procedure problem, not a result.

## 4. Pre-registered rules

- **Sanity:**
  - **S1–S5:** as in Study 3.
  - **S6′:** the vector files match Study 3's by SHA-256 (no slow recompute).
  - **S7:** the copy-fix check above.
  - **S8:** arm A and arm B spans don't overlap and together cover the whole Study 3 span.
- **Main test (arm A, text intact):** take the arm-A trials where the injected span was copied back
  exactly. **PASS** if, in those trials, **injected ≥ 50%** and **clean reader ≤ 15%**. It is only
  testable if there are **at least 40** such trials; if fewer, the result is "not testable".
- **Label test (arm B):** label highlighting counts as a **real alternative explanation** if arm B's
  injected accuracy is **≥ 50%** (≥ 100/200; 5× chance).
- **Secondary, for comparison with Study 3:** the arm-A loophole rule over all 200 trials (reader
  ≥ injected − 10 points → loophole confirmed); arm B's reader accuracy and how often its label is
  copied back changed.
- **Arm C (descriptive):** if random vectors are within 10 points of arm A, the sentence-only
  effect is **not concept-specific** ("disturbance", as with Tülu in Gate 1).
- **How the two tests combine (fixed now):**

| main test (A) | label test (B) | conclusion |
|---|---|---|
| PASS | below 50% | The sentence words alone are enough, with intact text, and the label push isn't. This is the strongest evidence in this project for **noticing**, though not proof. |
| PASS | ≥ 50% | **Both routes work.** Noticing is not ruled out, but label highlighting is a real alternative. |
| FAIL | ≥ 50% | **Study 3's effect is best explained by label-digit highlighting.** |
| FAIL | below 50% | Neither route alone is enough: **inconclusive** (the effect may need both, or text change). |

- **Descriptive:** per concept, per vector type, per position, favourite-answer analysis, KL.
- No tuning, no re-runs, no other layers or strengths.

## 5. Run time (from Study 3's measured times: injected 15 s, copy+reader 51 s per trial)

| step | time |
|---|---|
| model load + sanity (S6′ is a quick file check) | ~15 min |
| clean passes for 20 draws (localization + copy) | ~20 min |
| arm A: 200 injected + 200 copy+reader | ~3.6 h |
| arm B: 200 injected + 200 copy+reader | ~3.6 h |
| arm C: 200 injected | ~0.8 h |
| **total** | **~8.5–9 h (one night)** |

Shorter option: skip arm B's copy+reader → **~5.9 h**. Then the label test still works, but we
couldn't tell whether a label push visibly changes the copied label.

## 6. Approvals needed

1. The design above (arms A, B, C; rules; fresh draws; the copy fix).
2. Arm B with copy+reader (~9 h total, recommended) or without (~5.9 h).

## 7. Implementation notes (fixed before any Study 4 data)
- Per draw, all arms are run position by position (A, A-reader, B, B-reader, C), so a crash leaves
  the arms balanced.
- Arm spans come from character offsets: label = "SENTENCE N:", sentence = the words after
  "SENTENCE N: ". S8 checks they're disjoint and add up to the Study 3 span.
- Dry run on a tiny random model with the real tokenizer and fake vectors: every path ran, including
  resume. Only in the dry-run script, S7's threshold was set to 0 (a random model can't copy). The
  real run requires 19/20.
