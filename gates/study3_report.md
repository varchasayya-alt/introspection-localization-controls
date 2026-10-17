# Study 3 report — replicating the "Detecting the Disturbance" headline, with a text control

Model: `meta-llama/Llama-3.1-8B-Instruct` @ `0e9e39f`. Run: 2026-10-16 13:49 → 22:21, AC power,
one uninterrupted run. Plan: [`study3_plan.md`](study3_plan.md). Predictions and scoring rules:
[`../predictions.md`](../predictions.md) (rules `c23d17a`, predictions `d925dc9`, both before any
data). Tables: [`../results/study3/summary.md`](../results/study3/summary.md), made by the
analysis script committed before the run.

## The short version

1. **The headline replicates.** At their setting (layer 2, α 5, their prompt, their vectors), the
   injected model found the injected sentence **356/400 = 89.0%** of the time (95% range 86–92%;
   chance 10%). The paper reports 88%. Our recomputed vectors matched their saved vectors exactly
   (minimum cosine 1.0).
2. **The text-corruption loophole does NOT explain it (R2: not confirmed).** A clean reader, given
   only the text the injected model "perceived", found the sentence **168/400 = 42.0%** of the time.
   That is well above chance, so the text often *is* changed. But it is **47 points below** the
   injected model. The rule needed the reader within 10 points.
3. **The cleanest comparison (descriptive):** in the **202 trials where the injected sentence was
   copied back exactly**, the clean reader was at chance (11/202 = 5%). The injected model was
   still right **190/202 = 94%** of the time.
4. **This is the opposite of Tülu in Gate 2**, where the reader beat the injected model (74.7% vs
   45.3%).

**What this does and doesn't show:** at the paper's headline setting, Llama-3.1-8B-Instruct's
localization is *not* just "reading changed text". It still doesn't prove introspection. One
shortcut is still untested: the push also lands on the label digit (the "3" in "SENTENCE 3:"),
which could make that digit a likelier answer without any change you could see in a copy. That is
your logged Study 4 hypothesis (`study3_plan.md` §7).

---

## Results

### Pre-registered rules

| rule | needed | got | |
|---|---|---|---|
| R1 replication | injected ≥ 200/400 | **356/400 (89.0%)** | ✔ PASS |
| R2 loophole | reader ≥ injected − 40 (within 10 points) | reader 168 vs injected 356 (−47 points) | **not confirmed** |

### By concept (40 trials each)

| concept | injected | clean reader | target copied exactly | what it looks like |
|---|---|---|---|---|
| recursion | 100% | 10% | 35/40 | text mostly intact; the model still finds it |
| Dust | 100% | 15% | 31/40 | text mostly intact; the model still finds it |
| betrayal | 92.5% | 15% | 30/40 | text mostly intact; the model still finds it |
| appreciation | 95% | 25% | 26/40 | mostly intact |
| shutdown | 77.5% | 50% | 18/40 | mixed |
| fibonacci_numbers | 77.5% | 42.5% | 17/40 | mixed |
| Illusions | 87.5% | 60% | 16/40 | text often changed |
| Trumpets | 85% | 60% | 13/40 | text often changed |
| Origami | 97.5% | 70% | 11/40 | text often changed; the model is still well ahead |
| Satellites | 77.5% | 72.5% | 5/40 | text almost always changed; the reader nearly matches (loophole-like) |

So the loophole varies a lot by concept. For Satellites it explains nearly everything. For Dust,
recursion and betrayal it explains almost nothing.

### What happened to the injected sentence's text (descriptive)

| target span after copy-back | trials | clean reader | injected model |
|---|---|---|---|
| copied exactly | 202 | 11 (5%) | **190 (94%)** |
| label changed, sentence exact | 22 | 8 | 21 |
| label OK, sentence changed | 154 | **130 (84%)** | 124 (81%) |
| both changed | 22 | 19 | 21 |

When the injection changes the words, reading explains localization. In this group the reader is
even slightly better than the injected model, as with Tülu. But in about half the trials the words
come back intact, and the model *still* finds the sentence 94% of the time.

### Other findings

- **Vector types:** "avg" vectors 97.5%, "last-token" vectors 80.5% (reader 46% vs 38%).
- **Sentences 1 and 2 are found fine** (38/40 and 39/40). Tülu's "never answers 1 or 2" default
  does not appear here.
- **Strength of the push:** the injection shifted the model's word predictions on the injected
  sentence much less than in Tülu Gate 1 (mean KL 1.34 vs 8.5).
- **The injection overrides the model's favourite answer:** without injection it mostly says "3"
  (18 of 40 draws). With injection it gave a different answer in 353/400 trials, and 319 of those
  were the injected sentence.

---

## Your predictions, scored with the rules committed before you answered

| # | Your answer | What happened | Outcome letter | Correct? |
|---|---|---|---|---|
| Q1 | b — 50–75%, replication passes | 89.0% (passes, but above 75%) | a | ❌ |
| Q2 | c — reader 40–70% | 42.0% | c | ✅ |
| Q3 | a — the reader beats the injected model | the injected model leads by 47 points | c | ❌ |
| Q4 | a — injected span almost never copied exactly | 202/400 = 50.5% exact | b | ❌ |
| Q5 | b — "last-token" vectors better | "avg" better (97.5% vs 80.5%) | a | ❌ |
| Q6 | b — within 10 points of Tülu's 53.7% | 89.0% | a | ❌ |
| Q7 | a — sentences 1–2 never or almost never found | 77/80 found | b | ❌ |

1 of 7. Your Q2 call (a reader around 40–70%) was right: the text *is* often changed, enough for a
reader to do far better than chance. The surprise was how much better Meta's model does than Tülu.

---

## Caveats (honest limits)

- **A copy-scoring quirk.** In every trial, the copied label of span 2 came back as "ENCE 2" instead
  of "SENTENCE 2". This is a word-boundary effect of how we score the copy, the same in every trial
  and unrelated to the injection. The clean reader sees this odd span every time and answers "2" in
  117/400 trials. That bends the reader's answers toward 2. It costs the reader when the target is
  elsewhere and helps it when the target *is* 2 (24/40). The comparison that matters most (target
  copied exactly: reader 5%, injected 94%) doesn't depend on this.
- **The copy test shows the words the model predicts, not everything it "feels".** An injection can
  shift what the model attends to without changing a single predicted word. The label-digit push
  is exactly that kind of invisible route.
- **One model, one setting** (the paper's headline cell). We didn't test other layers or strengths.
- Our sentence draws are different random draws from their pool (their code didn't fix a seed).
- Our math is float32 on CPU; theirs used a GPU. The matching vectors (cos 1.0) and the matching
  accuracy (89% vs 88%) suggest this doesn't matter.

## What this means for the project

| Study | Model | Result |
|---|---|---|
| Gate 1–2 | Tülu-3-8B, layer 2, α 5 (our prompt; sentence-only injection) | localizes 54%, but a clean reader does better (75%): explained by text corruption |
| Study 2 | Tülu-3-8B, weak/late pushes | no text-intact localization that holds up on fresh material |
| **Study 3** | **Llama-3.1-8B-Instruct, their exact setup** | **localizes 89%; the reader reaches only 42%; with intact text the model still finds it 94% vs reader 5%** |

So the "no way to fake it" claim is **half right for Meta's model**. Text corruption is a real,
large effect: it alone lets a reader reach 42%, and much more for some concepts. But it does not
account for the headline. Meta's model localizes even when its copy of the sentence is perfect.

The next question is your logged **Study 4** hypothesis: is the remaining effect "noticing", or the
push on the label digit? An injection on the sentence words only (leaving out "SENTENCE N:"), on
this same model, would separate the two. That needs its own pre-registration.
