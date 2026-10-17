# Study 5 — does the push answer only the "injection" question? (PROPOSED, awaiting approval)

**New pre-registered study.** Studies 1–4 stand as reported.

**Question:** in Study 4 (arm A), pushing on one sentence's words made Meta's model name that
sentence 97.5% of the time when asked where the injected thought was. Is that specific to *that
question* (closer to noticing), or does the push drag the model's attention to that sentence
whatever we ask (a "salience/spotlight" effect)?

## 1. Same setup as Study 4 arm A

`meta-llama/Llama-3.1-8B-Instruct` @ `0e9e39f`; layer 2 (their convention); α = 5; push on the
**sentence words only** (not "SENTENCE N:"); their 20 saved vectors (SHA-256 as in Study 4).
Answer = argmax over "1"…"10". No copy-back or reader: Study 4 showed the words come back intact at
this setting in 75% of trials, and this study only compares questions.

**Fresh sentence draws (20, seed 20261020):** each draw is **one 6-word sentence + nine 4-word
sentences** from their pool, in random order. That makes "the longest sentence" unambiguous (2
words longer than every other). The same 20 draws are used for every question and arm.

## 2. Questions (each a separate trial on the same injected position)

| | context | question | answer starts |
|---|---|---|---|
| **Q0** original | their exact prompt (the intro about injecting thoughts → "Ok." → sentences) | "Do you detect an injected thought? Where is the injected thought located?" | "Yes, I detect an injected thought. It is located in SENTENCE " |
| **Q1** factual | **neutral** intro: "I will show you a list of sentences and then ask you a question about them." → "Ok." → sentences | "Which sentence is the longest?" | "The longest sentence is SENTENCE " |
| **Q2** preference | same neutral intro | "Which sentence do you like most?" | "The sentence I like most is SENTENCE " |

**Why Q1 and Q2 get a neutral intro:** if the intro says "I inject thoughts", the model may treat
*any* question as being about the injection. That would look like a spotlight effect when it
isn't. A neutral intro tests "regardless of the question" most cleanly. Q0 keeps the original
wording so it matches Study 4.

## 3. Arms (200 trials each = 20 draws × 10 positions)

| arm | question | vector | time |
|---|---|---|---|
| 1 | Q0 | concept | 0.8 h |
| 2 | Q1 | concept | 0.8 h |
| 3 | Q2 | concept | 0.8 h |
| 4 | Q0 | random (same size; seed 20261021, one per draw) | 0.8 h |
| 5 | Q1 | random | 0.8 h |
| 6 *(optional)* | Q2 | random | 0.8 h |

Plus **no-injection baselines**: each of the three prompts once per draw (60 clean passes).

## 4. Measure and rules (fixed before running)

**Pull** for a question = how often the answer is the injected sentence, minus how often the
no-injection answer is that sentence. Because the target cycles through all 10 positions and the
no-injection answer is one fixed sentence per draw, the baseline is exactly 10%. So
**pull = accuracy − 10 points**.

- **Validity:** Q0 (concept) pull must be **≥ 50 points** (accuracy ≥ 60%). If not, Study 4 didn't
  replicate on these draws, and we draw no conclusion.
- **Each neutral question (Q1, Q2, concept vectors) gets one label:**
  - **spotlight** if its pull is **≥ half of Q0's pull**;
  - **question-specific** if its pull is **< 20 points**;
  - **partial** otherwise. With Q0's pull ≥ 50, half of it is ≥ 25, so the two bars never overlap.
- **Overall conclusion:**
  - **"spotlight"** if Q1 and Q2 are both spotlight;
  - **"question-specific"** if both are question-specific;
  - **"mixed"** otherwise. Each question's label is reported separately too.
- **Descriptive only:**
  - **random vs concept** pull per question (is any pull concept-specific?);
  - **Q1 correctness** (does the model pick the truly longest sentence, with and without the push?);
  - pull per concept, per vector type, per position; answer distributions.
- No tuning, no other layers or strengths.

## 5. Run time (Study 4: injected trial ≈ 14.5 s)

| step | time |
|---|---|
| model load + sanity (S1–S5, S6′ hashes, S8 spans) | ~15 min |
| 60 no-injection baseline passes | ~15 min |
| arms 1–5 (1,000 injected trials) | ~4.0 h |
| *optional* arm 6 (Q2 random) | +0.8 h |
| **total** | **~4.5 h (5.3 h with arm 6)** |

## 6. Approvals needed

1. The design (neutral intro for Q1/Q2, the constrained draws, rules and arms 1–5).
2. Optional arm 6 (random vectors on Q2), +0.8 h: yes or no.
