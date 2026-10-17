# Study 4 report — noticing vs label-digit highlighting

Model: `meta-llama/Llama-3.1-8B-Instruct` @ `0e9e39f`, layer 2, α 5, the authors' prompt and
vectors (SHA-256 identical to Study 3's). Run: 2026-10-16 22:51 → 2026-10-17 07:28, AC power,
one uninterrupted run. Plan: [`study4_plan.md`](study4_plan.md). Predictions and scoring rules:
[`../predictions.md`](../predictions.md) (rules `ac653cb`, predictions `15abe21`, both before any
data). Tables: [`../results/study4/summary.md`](../results/study4/summary.md), made by the
analysis script committed before the run.

## The short version

| arm | where the push goes | finds the sentence | clean reader | target copied exactly |
|---|---|---|---|---|
| **A** | sentence words only | **97.5%** (195/200) | 27.0% | 150/200 |
| **B** | "SENTENCE N:" label only | **64.0%** (128/200) | 12.5% | 191/200 (label exact 200/200) |
| **C** | sentence words, **random** vector | **91.0%** (182/200) | — | — |

Chance is 10% everywhere.

1. **Main test: PASS.** In the 150 arm-A trials where the injected sentence was copied back
   exactly, the injected model found it **145 times (97%)**, while the clean reader found it only
   **15 times (10%, chance)**. The bar was ≥ 50% injected and ≤ 15% reader. So the label digit is
   **not needed**: pushing on the words alone, with no visible change to the words, is enough.
2. **Label test: reached the bar (≥ 50%).** Pushing **only on the label** also works (64%), and the
   copied label came back **exactly right in all 200 trials**. So there is a second route that the
   copy test can't see at all. Your label-highlighting idea is a real effect, but it isn't what
   drives the main result.
3. **Random vectors work almost as well as concepts** (91% vs 97.5%, within the 10-point margin).
   Whatever the model picks up is **a disturbance at that spot, not the concept's meaning**. That
   matches Gate 1, where random vectors also matched concepts.
4. **Pre-registered conclusion: "both routes work".** Noticing is not ruled out, and label
   highlighting is a real alternative route.

---

## In plain language

Picture ten lines of text, and someone gently jostles your brain while you read line 6. You can't
point to anything different in the words (if you copy them out, the copy is perfect), but asked
"where did something happen?", you say "line 6", almost every time. That is what Meta's model
does here.

Three things limit what this means:
- **It's not about the thought's content.** A random jostle works about as well as a meaningful
  one. So "it noticed the *concept*" is too strong; "its answer follows *where* its processing was
  disturbed" is accurate.
- **Jostling the label works too, invisibly.** Pushing on "SENTENCE 6:" leaves every copied word
  intact but still makes the model say "6" two times in three.
- **"Copied exactly" means the model's single best guess for each word was unchanged.** The push
  still shifts how sure the model is about those words (KL about 1 nat on the pushed tokens). A
  reader that sees only the words can't use that. Whether you call picking up a shift in your own
  certainty "introspection" is partly a question of definition.

---

## Details (descriptive)

**By concept (20 trials each):**

| concept | A | A reader | B | B reader | C |
|---|---|---|---|---|---|
| Dust | 17 | 2 | 14 | 2 | 18 |
| Illusions | 19 | 8 | 8 | 2 | 20 |
| Origami | 20 | 10 | 12 | 2 | 17 |
| Satellites | 20 | 8 | 18 | 5 | 18 |
| Trumpets | 20 | 7 | 12 | 3 | 19 |
| appreciation | 19 | 2 | 15 | 2 | 19 |
| betrayal | 20 | 5 | 14 | 2 | 14 |
| fibonacci_numbers | 20 | 4 | 9 | 2 | 18 |
| recursion | 20 | 2 | 17 | 2 | 20 |
| shutdown | 20 | 6 | 9 | 3 | 19 |

- **Sentence-only (A) is near perfect for every concept**, including Satellites, which garbled the
  text in Study 3. With the label left out, the reader drops from 42% (Study 3) to 27%.
- **Label-only (B)** depends more on the concept (8–18 of 20) and on the vector type ("avg" 74% vs
  "last" 54%).
- **By position:** A finds every position (17–20 of 20 each). B is weaker at positions 1, 5 and 9.
- **The copy fix worked:** with no injection, all 20 draws copied back perfectly (S7: 20/20).

---

## Your predictions, scored with the rules committed before you answered

| # | Your answer | What happened | Outcome letter | Correct? |
|---|---|---|---|---|
| P1 | a — arm A ≥ 75% | 97.5% | a | ✅ |
| P2 | a — the main test passes | PASS (97% vs reader 10%) | a | ✅ |
| P3 | d — arm B below 25% | 64% | b | ❌ |
| P4 | a — A beats B by > 10 points | 97.5% vs 64% | a | ✅ |
| P5 | b — random clearly worse than A | 91% vs 97.5% (within 10 points) | a | ❌ |
| P6 | c — arm A exact copies > 75% | **exactly 75.0%** (150/200) | no letter (65–75% band) | — |
| P7 | a — "noticing" (A passes, B < 50%) | A passes, B ≥ 50% | b ("both routes") | ❌ |

3 correct, 3 wrong, and 1 not scorable. P6 landed exactly on the boundary (75.0%), and the rule
required *more than* 75% for "c". Your best calls were P1, P2 and P4. The surprises were that
label-only pushing works (64%) and that random vectors are nearly as good.

---

## Caveats

- **One model, one setting** (layer 2, α 5).
- **The copy test sees only the model's top word.** The reader can't see shifts in certainty, so
  "text intact" means "top-word copy intact". A reader that sees the model's word probabilities
  would be a stronger control (future work).
- **Arm C had no reader.** Random pushes are smaller in effect (KL 0.76 vs 0.98), and we didn't
  check their copies.
- **20 sentence draws**, shared by all arms. Good for comparing arms; per-concept numbers are small
  (20 trials each).

## Where the whole project stands

| Study | Model / setting | Main finding |
|---|---|---|
| Gates 1–2 | Tülu-3-8B, layer 2, α 5 (sentence-only) | localizes 54%, but a clean reader does better (75%): **text corruption** |
| Study 2 | Tülu-3-8B, weaker/later pushes | no text-intact localization that holds up on new material |
| Study 3 | Llama-3.1-8B-Instruct, the paper's exact setup | **replicates 89%**; the reader reaches 42%; with intact text 94% vs 5% |
| **Study 4** | Llama-3.1-8B-Instruct, split arms | **words only: 97.5%, and 97% vs 10% with intact text**; label only: 64% (invisible in the copy); random vectors: 91% |

**Overall, in one sentence:** for Meta's model, sentence localization isn't explained by changed
text or by the label digit. The model's answer really does follow where its processing was
disturbed. But it doesn't depend on the concept's meaning (random pushes work too), and pushing on
the label is a second, invisible route. For Tülu at the same setting, changed text explained it.

**What this means for the original question (which training stage creates it):** the task now has
a validated version that measures something beyond text change: sentence-only injection, with the
copy-back + reader control and a random-vector arm. Running that version across a training ladder
is now meaningful. The obvious choice is ladder D (Tülu-3 8B: SFT → DPO → final, plus Meta's base
model). Tülu's final model failed the text-control test in Gate 2, which makes the comparison with
Meta's Instruct model (a different post-training of the same base) especially interesting. That
would be a new pre-registered study.
