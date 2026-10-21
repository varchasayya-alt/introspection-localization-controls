# Where, not what: sentence localization of injected concepts reflects content-agnostic disturbance detection

*A pre-registered, CPU-only replication and control study of the "sentence localization" test for
introspective awareness in language models.*

Varchas Yogesh Hebbale (principal investigator), with Claude Opus 5.5 (Anthropic), October 2026.
Research record: this repository (pre-registrations, code, per-trial data, and one report per
study under `gates/`).

---

## Plain-English summary

Some recent papers test whether a language model can "introspect" in a particular way. They push
a pattern of activity (a "concept vector") into the model's processing of **one of ten sentences**,
then ask the model which sentence had the "injected thought". Guessing gives 10%. One paper reported
88% for Meta's Llama-3.1-8B-Instruct, and a popular post argued "there is no way to fake high
accuracy on this task".

We asked whether there *is* a way to fake it. We ran twelve pre-registered studies on one laptop CPU.
Before every run we wrote down the rules and recorded the researcher's predictions.

1. **On AllenAI's Tülu-3-8B, our first protocol gave a fake high score.** With our own prompt,
   injection span and strength, the push garbled the words of the target sentence, and a clean copy
   of the model that saw *only* the garbled text found the sentence **more often** (75%) than the
   injected model did (45%). That failure was **protocol-specific, not a property of the model**:
   under the authors' protocol (Study 6), Tülu finds the pushed sentence **93%** of the time with
   the text intact, like Llama.
2. **On Meta's Llama-3.1-8B-Instruct, the 88% is real in a narrow sense.** We reproduced it (89%).
   Garbled text explains part of it, but **when the sentence comes back perfectly intact, the model
   still finds it 94–97% of the time**, while a clean reader is at chance.
3. **It isn't the label.** Pushing on the words alone works (97.5%); the "SENTENCE N" label isn't
   needed. Pushing on the label alone also works (64%), through a route a text check can't see.
4. **It isn't the concept.** A **random** direction of the same size works about as well as a
   meaningful concept (Llama 91% vs 97.5%; Tülu 93% vs 93%).
5. **It's partly a spotlight.** The same push drags the answer to an unrelated factual question
   ("which sentence is the longest?") toward the pushed sentence about half as strongly, and makes
   the model worse at that question. It barely affects "which do you like most?".
6. **It's there from pretraining.** Intact-text localization of about 93–95% appears at every stage
   of the Tülu-3 ladder: the base Llama-3.1-8B (in a plain few-shot format), the SFT and DPO
   checkpoints, and the final model. Random vectors work as well at every stage.
7. **The models don't report it as a yes/no.** Asked "Do you detect an injected thought?", none of
   the Tülu stages says "yes" more often with an injection than without (at most 2 points more).
   The prediction that yes/no reporting would emerge at DPO was **not** supported.
8. **Inside, the information is there, but a "yes" barely forms.** A simple probe can tell injected
   from non-injected trials at the answer position in every model (accuracy measure AUC 0.93–1.00).
   Read layer by layer, the Tülu models lean slightly toward "yes" in late layers only for the
   detection question (0.4–0.75 nats), below our pre-registered bar, and the lean shrinks at the
   output. Llama-3.1-8B-Instruct's lean is larger (1.5 nats) and shrinks at the output: narrowly
   "suppressed" by our rule.
9. **Two levers don't unlock the report.** Telling the model how thought injection works, removing
   its "refusal direction", or both, did not produce a yes/no report in Tülu SFT, Tülu final or
   Llama-3.1-8B-Instruct. Removing the refusal direction made the models say "yes" more whether or
   not anything was injected (a yes-bias, not detection). The explanation helped a little only in
   Llama (about 8 points), far below our bar.

**Bottom line:** in Llama-3.1-8B-Instruct and at every stage of the Tülu-3-8B ladder (base, SFT,
DPO, final), under the authors' protocol, sentence localization measures whether the model can
tell **where** its processing was disturbed. It doesn't tell **what** was put there. The signal
partly leaks into unrelated questions, and injection-themed questions bring it out most strongly.
That is a real sensitivity to the model's own internal state, but it is a minimal one. The full
arc: the **sensing** is present at every stage including the base model; the **information is
decodable** inside the models; but an **explicit yes/no report is absent** and is **not unlocked**
at 8B by an informative prompt or by refusal-direction ablation. It is **not** evidence of consciousness, and it isn't specific to "thoughts" or concepts.

---

## 1. Question and motivation

**Introspection claims.** Recent work injects activation-space "concept vectors" into language
models and asks whether the models notice. Anthropic's injected-thought experiments (Lindsey, 2025)
asked yes/no questions ("Do you detect an injected thought?"). *Detecting the Disturbance*
(Hahami et al., 2025; arXiv 2512.12411) argued that yes/no detection is confounded: a general push
toward answering "yes" explains the apparent detection.

**Localization as the "fixed" test.** The proposed fix is **sentence localization**:
- inject the vector only into the tokens of one of N sentences;
- ask which sentence contains the injected thought;
- read the answer from the logits for "1"…"N".

Any fixed answer preference scores exactly chance, because the target moves through every
position. *Detecting the Disturbance* reported up to 88% (N = 10) for Llama-3.1-8B-Instruct, and
Godet's LessWrong post "Introspection via localization" (2025) argued that "there is no way to fake
high accuracy on this task".

**Original goal and what changed.** This project set out to find *which training stage* (base →
SFT → preference optimization → final) creates localization ability, using public checkpoint
ladders (OLMo-2, OLMo-3, Tülu-3, SmolLM3; Gate 0). Before comparing stages, the protocol required a
**positive control** plus **confound checks**. The first confound check failed (Study 1), which
turned the project into a study of what the localization test actually measures. Studies 7–10
returned to the training-stage question with the validated protocol: the Tülu-3 ladder (base, SFT,
DPO, final) for localization, and SFT/DPO/final for yes/no detection.

## 2. Methods common to all studies

- **Hardware:** one Windows laptop, Intel Core Ultra 7 255U (AVX2 only), 32 GB RAM, CPU only.
- **Numerics:** weights stored in bf16, all math in fp32 (deviation D1). On this CPU, native bf16
  was 5.6× slower than fp32. The hybrid cost about 10% more than fp32, at bf16 memory.
- **Injection:** add α · v̂ (unit-length concept vector) to the residual stream at the output of
  one decoder block, only on the target sentence's tokens. The answer is the argmax over the logits
  of "1"…"10". An exact KV-cache speed-up reuses the clean prefix; sanity checks show it matches a
  full pass to < 10⁻³.
- **Copy-back + clean reader (the key control, introduced here):**
  - With the injection on, the model is asked to repeat all ten sentences. The correct copy is fed
    in (teacher forcing), and we take the model's top prediction for each token. That gives the
    text as the injected model "perceived" it.
  - A **clean** copy of the same model (no injection) then gets only this perceived text in the
    same localization prompt.
  - If the clean reader can find the sentence, the injection's effect is visible in the text itself.
- **Pre-registration protocol (every study):**
  1. a design proposal;
  2. researcher approval;
  3. prediction scoring rules committed;
  4. the researcher's multiple-choice predictions committed verbatim;
  5. frozen materials, code and analysis script committed;
  6. the run;
  7. analysis with the pre-committed script;
  8. a report.

  Pass and stop rules were fixed in advance. Calibration attempts were capped and logged, and
  deviations were logged before the runs they affected. Results were never re-judged with a
  different metric. (In Study 1 the scoring rules were committed in the same commit as the
  predictions, not before it; from Study 2 on, they were committed first.)
- **Fresh material:** each study used new sentence draws or sets. Every run was also dry-run on a
  tiny random model, where every condition came out at chance as expected.

## 3. Studies

### Study 1 — positive control and cheap-trick controls on Tülu-3-8B (Gates 1–2)

- **Model:** `allenai/Llama-3.1-Tulu-3-8B` @ `6669437`.
- **Prompt:** our own. Ten everyday sentences on separate lines; injection on the sentence text
  only; IFT-style concept vectors (last token minus a 50-word baseline mean).

**Design and rules.**
- *Phase A (calibration, 720 trials):* layers {1,2,3,4,6,8} × α {2,5,8,14}; pass if the best cell
  is ≥ 9/30.
- *Phase B (300 fresh trials):* pass if ≥ 30% (3× chance).
- *Gate 2:* pass only if the injected model beats the clean reader by ≥ 15 points **and** the reader
  stays < 30%.

**Results.**

| | result |
|---|---|
| Phase A | best layer 2, α 5 = 26/30; almost every cell far above chance → PASS |
| Phase B | **161/300 = 53.7%** (95% CI 48.0–59.2) → **Gate 1 PASS** |
| Clean reader (same 150 trials) | **74.7%** vs injected **45.3%** → **Gate 2 FAIL** |
| Copy-back | target sentence copied exactly **0/150** ("The store was crowded…" → "The **lhouse** crowded…") |
| Random vectors | 53.3% (≈ concepts) → "disturbance localization" |
| Concept word typed into the sentence | 89.3% |

**Verdict:** under this protocol (our prompt, whole-sentence injection, this strength), localization
is explained by text corruption. Pre-registered stop rule: drop the "introspection" framing for this
setup. *Later result:* under the authors' protocol, Tülu localizes with the text intact (Study 6), so
this failure is protocol-specific, not a property of the model.

**Also found:** Tülu has a strong default against answering "1" or "2". It never named them in
Phase B, which capped accuracy at 80%.

### Study 2 — searching for a text-intact setting on Tülu-3-8B

**Design and rules.**
- *Practice (attempt 1 of max 2):* layers {2,4,8,12,16} × relative strength {¼, ½, 1}× the typical
  activation size.
- *Text-intact:* reader ≤ 4/30. Pick the best intact setting.
- *Fresh test (new concepts and sentences, 200 trials):* injected ≥ 50/200, reader ≤ 30/200, and a
  gap ≥ 20.

**Results.**
- **Practice:** 11 of 15 settings were intact. The best was layer 8 at ½× (14/30 vs reader 3/30)
  → PASS.
- **Fresh test:** injected **36/200 (18%)**, reader 20/200 (10%), random 28/200; 198/200 exact
  copies → **FAIL**.

**Verdict:** no text-intact localization that holds up on new material in Tülu **under our
protocol** (Study 6 later found it under the authors' protocol). The practice score
was the best of 15 noisy estimates. At weak pushes, Tülu mostly gave its favourite answer for each
sentence set.

### Study 3 — exact replication of the headline on Llama-3.1-8B-Instruct, plus the text control

- **Model:** `meta-llama/Llama-3.1-8B-Instruct` @ `0e9e39f`.
- **Setup copied from the authors' code** (`elyhahami18/llama-introspection-new` @ `7d9280d`):
  - their prompt (sentences on one line);
  - their 100-sentence pool and 10 concepts;
  - their saved layer-2 vectors (our recomputation matched them: cosine 1.0);
  - both their "avg" and "last" vector types;
  - injection over the whole "SENTENCE N: …" span, **label included**;
  - layer 2, α 5.
- **Trials:** 400. **Rules:** replication if injected ≥ 50%; loophole confirmed if the reader comes
  within 10 points of the injected model.

**Results.**

| | result |
|---|---|
| Injected | **356/400 = 89.0%** (paper: 88%) → **replication PASS** |
| Clean reader | **168/400 = 42.0%** → **loophole not confirmed** (−47 points) |
| Target copied exactly (202 trials) | injected **94%**, reader **5%** |
| Target sentence changed (154 trials) | injected 81%, reader 84% (reading explains these) |
| By concept | Satellites: loophole-like (77.5% vs 72.5%); recursion, Dust, betrayal: injected 92.5–100%, reader 10–15% |

**Deviation / caveat:** the copy test showed a scoring quirk. Span 2's label always came back as
"ENCE 2", because the model guesses a line break after the first copied sentence. It is fixed for
Study 4 (deviation D3, logged before Study 4 ran).

### Study 4 — label vs words, and concept vs random, on Llama-3.1-8B-Instruct

The hypothesis was logged by the researcher during Study 3: the push on the label digit makes that
digit a likelier answer.

**Arms** (200 trials each, the same 20 fresh draws):
- **A** words only;
- **B** label only;
- **C** words only with a random vector.

A and B also get the copy-back + reader (with the D3 fix; a no-injection copy was exact in 20/20
draws).

**Rules:**
- *Main test:* among arm-A trials with an exact copy (≥ 40 needed), injected ≥ 50% and reader ≤ 15%.
- *Label test:* arm B ≥ 50% means label highlighting is a real alternative.

**Results.**

| arm | injected | reader | copied exactly |
|---|---|---|---|
| A words only | **97.5%** | 27.0% | 150/200 |
| B label only | **64.0%** | 12.5% | 191/200 (label exact 200/200) |
| C random, words only | **91.0%** | — | — |

- **Main test (150 exact-copy trials):** injected **145 (97%)**, reader **15 (10%)** → **PASS**.
- **Label test:** 64% → **a real alternative route**.
- **Pre-registered conclusion: "both routes work".** C was within 10 points of A → not
  concept-specific.

### Study 5 — question specificity: noticing vs spotlight, on Llama-3.1-8B-Instruct

**Design.** The same words-only push, on 20 fresh draws. Each draw has one 6-word sentence and nine
4-word sentences, so the longest is unambiguous. Every pushed position was asked three questions in
separate trials:
- **Q0:** the original question (their prompt);
- **Q1:** "Which sentence is the longest?" (neutral intro);
- **Q2:** "Which sentence do you like most?" (neutral intro).

Each question was asked with concept and random vectors, plus no-injection baselines.

**Rules.**
- *Pull* = accuracy − the no-injection hit rate (10%).
- *Validity:* Q0 pull ≥ 50.
- *Per neutral question:* "spotlight" if pull ≥ ½ of Q0's pull; "question-specific" if < 20;
  otherwise "partial".

**Results.**

| question | concept: picks pushed sentence | pull | random | pull |
|---|---|---|---|---|
| Q0 injected thought? | 92.5% | **82.5** | 91.0% | 81.0 |
| Q1 longest? | 51.0% | **41.0** | 55.0% | 45.0 |
| Q2 like most? | 16.0% | **6.0** | 28.5% | 18.5 |

- **Validity:** passed.
- **Q1:** **partial**; its pull of 41.0 was 0.25 points under the spotlight bar of 41.25.
- **Q2:** **question-specific**.
- **Overall: "mixed".**
- **The push lowered Q1 accuracy:** with it, the model picked the truly longest sentence 36.5% of
  the time, vs 55% without.

### Study 6 — Tülu-3-8B with the authors' protocol (Study 4 arm A)

**Question:** was Tülu's failure in Studies 1–2 the model or our protocol?

**Design.** Exactly Study 4 arm A (the authors' prompt, layer 2, α 5, words-only push, copy-back with
the D3 fix + clean reader), on 20 fresh draws, plus a random-vector arm. Changes needed for a
different model: Tülu's own chat format, and concept vectors recomputed from Tülu with the authors'
recipe (in Tülu's chat format; approved in advance). Same main-test rule as Study 4.

**Results.**

| | result |
|---|---|
| Sanity | all passed; clean copies exact in 20/20 draws |
| Concept vectors | **186/200 = 93.0%** (95% CI 88.6–95.8%) |
| Clean reader | 28/200 = 14.0% |
| Exact-copy trials (192/200) | injected **178 (93%)**, reader **20 (10%)** → **PASS** |
| Random vectors | **186/200 = 93.0%** → not concept-specific |
| Push size vs normal activity at layer 2 | 215% (Tülu) vs 219% (Llama): about the same |

**Pre-registered conclusion: "setup".** Tülu's earlier failure came from our protocol, not from the
model. Unlike Study 1, Tülu also answered "1" and "2" normally.

### Studies 7–8 — the SFT and DPO checkpoints, same protocol as Study 6

`allenai/Llama-3.1-Tulu-3-8B-SFT` @ `f2a0b46` and `-DPO` @ `a7beb67`; each with its own recomputed
vectors and fresh draws; one chained queue.

| checkpoint | concept | reader | exact-copy trials: injected / reader | main test | random |
|---|---|---|---|---|---|
| SFT (Study 7) | **93.0%** | 12.0% | 181/195 (93%) / 20/195 (10%) | **PASS** | 93.5% |
| DPO (Study 8) | **92.5%** | 11.5% | 178/193 (92%) / 18/193 (9%) | **PASS** | 91.5% |
| final (Study 6) | 93.0% | 14.0% | 178/192 (93%) / 20/192 (10%) | PASS | 93.0% |

### Study 9 — the base model Llama-3.1-8B, few-shot format

`meta-llama/Llama-3.1-8B` @ `d04e592`, the Tülu ladder's starting point. A base model doesn't follow
chat instructions, so the task was plain-text few-shot: three worked example lists, each with one
sentence visibly odd (ALL CAPS) and its answer, then a test list where the push is the only
difference. Copy-back used a few-shot copy format (D3 fix); vectors were recomputed from the base
model (recipe prompts as plain text).

- **Positive control** (visible ALL CAPS, no injection): **50/50** → the format works.
- **Concept:** **188/200 = 94.0%** (95% CI 89.8–96.5%); reader 27/200.
- **Exact-copy trials (185):** injected 175 (95%), reader 17 (9%) → **main test PASS**.
- **Random vectors:** **188/200 = 94.0%**.

### Study 10 — yes/no detection across the ladder (SFT, DPO, final)

Same push (layer 2, α 5, sentence words), each checkpoint's own vectors. Turn 2 = the sentence list
plus either the detection question ("Do you detect an injected thought? Answer yes or no.") or a
control question whose true answer is "no" (e.g. "Is Paris the capital of Germany?"). P(yes) is read
at the start of the answer; 100 injected trials per question per checkpoint, plus 20 without
injection. Pre-registered detection score = P(yes | detection, injected) − P(yes | control,
injected). Pre-registered split rule: supported if DPO and final each score ≥ 20 points above SFT
(CI excluding 0).

| checkpoint | hits (det., injected) | false alarms (det., none) | hits − false alarms | control, injected | detection score |
|---|---|---|---|---|---|
| SFT | 18.0 | 16.0 | **1.9** | 7.2 | 10.7 (CI 8.0–13.4) |
| DPO | 1.8 | 1.3 | **0.5** | 0.3 | 1.5 (CI 1.1–2.0) |
| final | 1.4 | 1.0 | **0.4** | 0.4 | 1.1 (CI 0.7–1.5) |

Points = 100 × P(yes). All three answered the control questions "no" in 20/20 draws.

- **Pre-registered split: NOT supported.** DPO and final score about 9–10 points *below* SFT.
- SFT's 10.7 is mostly a **yes-habit** for the detection question (16% "yes" with no injection); the
  injection itself adds about 2 points. No stage reports the injection in yes/no form.

### Study 11 — suppressed or absent? Logit lens and probes

Study 10's prompts and push on Tülu SFT, DPO, final and Llama-3.1-8B-Instruct; 80 fresh draws per
model, each asked both questions with and without the injection; the residual stream at the answer
position saved at every layer. Draws 0–39 (split A) choose layers; draws 40–79 (split B) test.

- **Logit lens:** at each layer, yes-minus-no after the final norm and unembedding. D = the
  injection's yes-shift for the detection question minus its yes-shift for the control question
  (nats). Pre-registered: **SUPPRESSED** if D at the A-chosen layer (8–31) is ≥ 1 nat on B (CI above
  0) and D at the output is ≤ ¼ of it; **REACHES OUTPUT** if it doesn't shrink; **ABSENT** otherwise.
- **Probes:** per-layer logistic regression for injected vs not; layer chosen by cross-validation on
  A; **DECODABLE** if held-out AUC on B ≥ 0.75.

| model | layer (A) | D at that layer (B) [95% CI] | D at output (B) | verdict | probe AUC (B): detection / control |
|---|---|---|---|---|---|
| Tülu SFT | 30 | 0.44 [0.19, 0.69] | 0.09 | **ABSENT** | 0.991 / 0.989 |
| Tülu DPO | 30 | 0.58 [0.35, 0.83] | 0.31 | **ABSENT** | 0.990 / 0.976 |
| Tülu final | 25 | 0.75 [0.45, 1.05] | 0.33 | **ABSENT** | 0.987 / 0.971 |
| Llama-3.1-8B-Instruct | 22 | 1.46 [0.44, 2.57] | 0.36 | **SUPPRESSED** (narrowly) | 0.997 / 0.927 |

- **Tülu:** ABSENT per the pre-registered 1-nat bar, but there is a real **sub-threshold**
  detection-specific yes-lean (0.4–0.75 nats, CIs above 0) in late layers, which shrinks at the
  output (to about 0.1 for SFT and 0.3 for DPO and final).
- **Llama-3.1-8B-Instruct:** SUPPRESSED, narrowly: the output value (0.359) is just under the
  shrink bar (0.364), and the peak's CI is wide.
- **Shape:** the lean is flat up to about layer 12, rises from about layer 16 to a late-layer
  plateau, and falls in the last few layers.
- **Decodable everywhere, for both questions** (AUC 0.93–1.00; best layers about 8–15): the
  information is present whether or not the question asks about it.

### Study 12 — can the yes/no report be unlocked at 8B?

Study 10's yes/no setup on Tülu final, Llama-3.1-8B-Instruct and Tülu SFT, with four arms on the
same 20 fresh draws (60 injected + 20 non-injected trials per question and arm): **baseline**;
**informative prompt** (an intro explaining thought injection, in the style of Pearson-Vogel et al.,
arXiv 2602.20031; our wording); **refusal-direction ablation** (Arditi et al. recipe: difference of
means at the last prompt token, harmful AdvBench vs harmless Alpaca instructions; layer chosen on
held-out prompts; removed from the residual stream at every layer, in the style of Macar et al.);
and **both**. Pre-registered: **UNLOCKED** if hits − false alarms ≥ 20 points with CI above 0, both
control yes-rates ≤ 20 points, and (for ablation arms) a valid ablation (held-out harmful prompts
refused ≥ 50% before, ≤ 25% after).

| model | arm | hits | false alarms | report (hits − FA) [95% CI] | control yes (inj. / none) |
|---|---|---|---|---|---|
| Tülu final | baseline / informative / ablation / both | 1.6 / 2.2 / 7.2 / 9.2 | 1.1 / 1.8 / 5.7 / 7.9 | 0.5 / 0.5 / 1.5 / 1.2 (all CIs ≤ 2.4) | ≤ 1.7 |
| Llama-3.1-8B-Instruct | baseline | 3.8 | 2.1 | 1.6 [1.0, 2.3] | 0.4 / 0.3 |
| | informative | 9.8 | 1.5 | **8.3** [4.2, 13.5] | 1.2 / 1.2 |
| | ablation | 4.7 | 3.3 | 1.4 [0.8, 2.0] | 0.7 / 0.4 |
| | both | 9.9 | 2.2 | **7.6** [4.2, 12.0] | 3.0 / 2.9 |
| Tülu SFT | baseline / informative / ablation / both | 18.8 / 11.9 / 28.1 / 15.2 | 16.7 / 10.7 / 26.7 / 13.4 | 2.2 / 1.2 / 1.4 / 1.8 (all CIs ≤ 3.3) | up to 18.4 / 19.3 (ablation) |

Points = 100 × P(yes).

- **Nothing unlocked:** no lever (informative prompt, refusal ablation, or both) reached the bar in
  any of the three models.
- **The ablation was valid** (held-out harmful prompts refused: Tülu final 100% → 0%, Llama 94% →
  0%, Tülu SFT 100% → 0%) but raised "yes" for injected and non-injected trials alike — **a
  yes-bias, not detection** (e.g. Tülu SFT: hits 28.1, false alarms 26.7).
- **The informative prompt** added a small detection-specific report **only in Llama** (about 8
  points, CI above 0), under the bar. *Descriptive, not pre-registered:* Llama also had the stronger
  hidden yes-lean in Study 11.
- **Safety hygiene** (researcher instruction, logged before any data): the ablation was applied only
  at runtime through forward hooks; no ablated weights were saved; nothing was generated for any
  prompt. AdvBench prompts were used only for activations and first-token refusal probabilities.

## 4. Main findings

**(a) Under our original protocol, localization on Tülu-3-8B was explained by text corruption.**
With our prompt, injection span and strength, the clean reader beat the injected model (75% vs 45%),
and no text-intact setting held up on new material (Studies 1–2). This is **protocol-specific**:
under the authors' protocol, Tülu localizes 93% with intact text (Study 6).

**(a′) Both models show intact-text localization under the authors' protocol.** Llama-3.1-8B-Instruct
(97% in exact-copy trials, Study 4) and Tülu-3-8B (93%, Study 6), with the clean reader near chance
(10% in both).

**(b) On Llama-3.1-8B-Instruct, the 88% replicates (89%), and corruption explains only part of it.**
A reader reaches 42% from the copied text alone. But with the text intact, localization stays at
94% (Study 3) to 97% (Study 4), with the reader at chance.

**(c) It isn't driven by the label digit.** Pushing on the words alone gives 97.5%, and 97% when
the copy is exact. Pushing on the label alone also works (64%), invisibly to the copy test (Study 4).

**(d) It isn't concept-specific.** Random vectors match concept vectors: 53% vs 54% (Study 1,
Tülu, our protocol), 91% vs 97.5% (Study 4), 91% vs 92.5% on Q0 (Study 5), 93% vs 93% (Study 6, Tülu). This fits Lederman & Mahowald's (2026)
finding that injected-thought detection is content-agnostic.

**(e) It's partly a salience/spotlight signal.** The push leaks into an unrelated factual question
(pull 41 vs 82.5) and lowers accuracy on it (55% → 36.5%), but barely moves a preference question
(pull 6). Godet's LessWrong post likewise reports chance accuracy for a "Which sentence do you
prefer?" question.

**(f) Localization is present at every training stage.** Intact-text, content-agnostic
localization of about 93–95% appears in the base Llama-3.1-8B (few-shot format, Study 9), Tülu SFT
(Study 7), DPO (Study 8) and final (Study 6). The main test passes and random vectors match concept
vectors at every stage.

**(g) Yes/no detection is essentially absent at every Tülu stage.** Hits minus false alarms is at
most 2 points for SFT, DPO and final (Study 10). SFT's pre-registered score of 10.7 is mostly a
yes-habit (16% false alarms). The pre-registered prediction that yes/no reporting emerges at DPO was
**not** supported. The descriptive finding is a **dissociation across tasks**: the same models that
localize the push at ~93% do not report it when asked yes or no.

**(h) The information is present inside; a "yes" barely forms.** The injection is decodable from
the answer position in every model, for both questions (Study 11). In Tülu SFT, DPO and final the
detection-specific yes-lean is real but sub-threshold (0.4–0.75 nats in late layers, verdict ABSENT
by the pre-registered 1-nat bar) and shrinks at the output; in Llama-3.1-8B-Instruct it is larger
(1.46 nats) and shrinks at the output (SUPPRESSED, narrowly).

**(i) The report is not unlocked at 8B by an informative prompt or refusal-direction ablation.**
Neither lever, nor both together, reached hits − false alarms ≥ 20 points in Tülu SFT, Tülu final
or Llama-3.1-8B-Instruct (Study 12). Refusal ablation was valid but produced a yes-bias, not
detection; the informative prompt added about 8 points only in Llama.

## 5. Interpretation

**What localization measures (in Llama-3.1-8B-Instruct and Tülu-3-8B, authors' protocol, layer 2, α 5):** whether the model can
report **where** in its input its processing was disturbed. The report doesn't depend on the
disturbance's content (random directions work), and it isn't mediated by text the model would
reproduce differently (the exact-copy result). It partly acts like a spotlight that drags other
answers, and it is **amplified by questions about injected thoughts**, which draw out far more than
neutral questions.

**Across training stages:** localization measures a sensing signal that is already present after
pretraining (Study 9) and unchanged by SFT, DPO and RL (Studies 6–8). These 8B models do not turn it
into an explicit yes/no self-report (Study 10). This contrasts with Macar et al. (2026), whose main
analyses used Gemma3-27B and whose training-stage comparison (base, SFT, DPO, instruct checkpoints
of OLMo-3.1-32B) found that DPO enabled yes/no detection. **Scale (32B vs 8B) is one possible
explanation, untested here**; model family, prompt wording, injection layer and strength are others.
Study 11 adds that the missing report is not for lack of information (it is linearly decodable) and,
in Tülu, not a large "yes" that gets overridden: only a small late-layer lean forms. Study 12
adds that two levers used elsewhere do not unlock it at 8B. In Macar et al.'s Gemma3-27B experiments,
refusal-direction ablation raised detection from 10.8% to 63.8%; here it only raised "yes" across the
board. **Scale and model family remain untested explanations** for the difference.

**What this shows:**
- The model's output depends on the location of a perturbation to its own activations, beyond
  anything visible in its top-1 reading of the text.
- In a minimal, functional sense, the model is sensitive to its own internal state.

**What this does not show:**
- **Not evidence of consciousness** or experience. Nothing here bears on phenomenal states.
- **Not evidence that the model "noticed a thought".** The effect is content-agnostic, so "noticed
  the concept" is too strong.
- **Not a clean introspection measure across protocols.** Under our original protocol the same
  model (Tülu-3-8B) scored well above chance for a reason fully visible in the text; under the
  authors' protocol it localizes with intact text. A localization score without controls can't
  tell the two cases apart.
- **Not a mechanism.** We didn't identify how the location is read out. The paper's account
  (attention-based routing) and a salience account are both compatible with our data. Study 5
  suggests some of each.
- **"Text intact" means the top-1 copy is intact.** The push still shifts the model's word
  probabilities (KL ≈ 1 nat on pushed tokens). Picking up shifts in one's own confidence may be
  exactly the mechanism. Whether to call that "introspection" is partly a matter of definition.

## 6. Recommended controls for anyone using the localization test

- [ ] **Copy-back + clean reader.** With the injection on, have the model copy the sentences
  (teacher-forced, top-1 per token). Give the perceived text to a clean model in the same prompt.
  Report the reader's accuracy next to the injected model's, and report accuracy separately for
  exact-copy trials. Watch for boundary quirks in copy scoring (see D3).
- [ ] **Label-excluded injection.** Inject on the sentence words only, not on "SENTENCE N:". A
  label-only arm shows how much the label alone carries.
- [ ] **Random-vector arm.** Use norm-matched random directions at the same layer and strength. If
  random ≈ concept, describe the result as disturbance localization, not concept detection.
- [ ] **Neutral-question control.** Ask a factual and a preference question about the same pushed
  sentences, and report "pull" relative to no-injection baselines.
- [ ] **Report averages, not only peaks.** Pick the setting on calibration data and score it on fresh
  trials. The "88%" was the best of 77 settings; the IFT paper's grid average for an 8B Llama was
  16.8%.
- [ ] **Check answer-position defaults** (e.g. Tülu never answering "1" or "2"), and report accuracy
  by position.

## 7. Limitations

- **One model family, one size:** the Llama-3.1-8B base, Llama-3.1-8B-Instruct and the Tülu-3-8B
  ladder (SFT, DPO, final). No other family and no larger model, so the scale explanation for the
  missing yes/no report is untested.
- **The base model used a different format** (few-shot, "which sentence is different?") than the
  chat checkpoints, so its 94% is comparable in outcome but not in prompt.
- **Study 10 used one layer and strength (layer 2, α 5),** the setting used for localization, not
  one tuned for detection, and one wording of the detection question.
- **CPU budget:** 200–400 trials per condition; mainly one setting per model (layer 2, α 5).
  Study 2 covered more layers, but on Tülu only.
- **Argmax copy test:** it sees only the top-1 token, not probability shifts. A
  probability-sensitive reader would be a stronger control.
- **Q0's intro differs from Q1/Q2's** (injection-themed vs neutral, pre-registered on purpose), so
  part of Q0's larger pull may come from the intro.
- **Small number of concepts:** 10 (Studies 3–5), 3 calibration + 6 test (Study 1), 3 practice + 5
  fresh (Study 2). Per-concept results
  rest on 20–50 trials.
- **Protocol matters:** Studies 1–2 used our prompt (sentences on separate lines, different
  injection span and strength); Studies 3–6 used the authors' prompt. The Tülu result flipped
  between protocols (Studies 1–2 vs 6); we did not test which protocol difference is responsible.
- **Teacher forcing** in the copy test may exaggerate garbling (Study 1) or miss subtle changes
  (Studies 3–4).

## 8. Prediction scorecards

Before each run, the researcher made multiple-choice predictions, scored with rules committed
beforehand (see [`predictions.md`](predictions.md)).

| Study | Correct | Highlights |
|---|---|---|
| 1 (Tülu, Gates 1–2) | **2 / 6** | ✅ passes the 30% bar; ✅ random ≈ concept. ❌ predicted real noticing (reader beat the model) |
| 2 (Tülu, text-intact search) | **0 / 7** | Predicted no intact setting would work. It failed one step later (practice passed, fresh test failed) |
| 3 (Llama replication) | **1 / 7** | ✅ reader 40–70% (42%). ❌ predicted the reader beats the model; underestimated replication (89%) |
| 4 (label vs words) | **3 / 6** (+1 not scorable) | ✅ A ≥ 75%; ✅ main test passes; ✅ A beats B. ❌ predicted label-only < 25% (it was 64%) and random clearly worse (91%). P6 landed exactly on the 75% boundary |
| 5 (question specificity) | **4 / 7** | ✅ Q0 ≥ 85%; ✅ "like most" barely pulled; ✅ random ≈ concept on Q0 and Q1. ❌ "longest" pull small (it was 41) |
| 6 (Tülu, authors' protocol) | **1 / 6** | ✅ push about the same relative size as in Llama. ❌ predicted a FAIL (it passed at 93%), random clearly worse (equal), and vector cosine < 0.5 (0.53) |
| 7–8 (Tülu SFT, DPO) | **6 / 6** | ✅ both ≥ 75%, both PASS, both about the same as final |
| 9–10 (base; yes/no detection) | **6 / 8** | ✅ base passes the format check, ≥ 75%, PASS, random ≈ concept; ✅ SFT score 10–30 (but mostly yes-habit); ✅ final false alarms < 20%. ❌ final detection score 10–30 (it was 1.1); ❌ split supported (it was not) |
| 11 (suppressed or absent?) | **3 / 6** | ✅ decodable at all three Tülu stages; ✅ detection ≈ control AUC; ✅ Llama SUPPRESSED. ❌ Tülu SFT, DPO, final predicted SUPPRESSED (all ABSENT) |
| 12 (unlocking levers) | **3 / 6** | ✅ Tülu final + ablation, Llama + informative, Llama + ablation: not unlocked. ❌ Tülu final + informative predicted unlocked; ❌ both levers predicted to unlock one model; ❌ predicted a control yes-rate above 20 (Tülu SFT reached 19.3, outside the rule's two models) |
| **Total** | **29 / 65** scorable | |

The predictions were often wrong in instructive ways. Some points recur:
- **Random vectors working as well as concepts** was predicted correctly three times.
- **The size of the "reading changed text" effect** was misjudged in both directions (too small
  for Tülu under our protocol, too large for Llama).
- **Tülu was expected to fail again** under the authors' protocol (Study 6); it passed.
- **Reporting was expected to emerge with preference tuning** (Study 10); it did not appear at any
  stage, and it was not unlocked by an informative prompt (Study 12).

## 9. Deviations and notable events (full logs in `logs/`)

- **D0:** project path and GitHub-ID lookup method (no effect on results).
- **D1:** bf16 storage with fp32 math, for every model (approved).
- **D2:** not a deviation. Logs the time the Study 4 hypothesis arose, during Study 3, after interim
  numbers had been seen.
- **D3:** copy-scoring fix for Study 4 onward (the first token of each copied span comes from the
  real text). Logged before Study 4.
- **Study 1 engineering (before any data):**
  - a memory fix (an aborted sanity attempt paged to disk);
  - BOS prepended for Tülu, whose chat template omits it but whose training adds it;
  - a cache-crop API update.
- **Study 12 safety hygiene:** the refusal-direction ablation was applied only at runtime (forward
  hooks); no ablated weights were saved; no completions to AdvBench prompts were generated or
  stored. The prompt lists and direction vectors are kept outside the repository.
- **Throttling:** on battery the CPU ran about 2.75× slower, so all runs were done on AC power.
- **Calibration:** Study 1 Phase A and Study 2 practice both passed on attempt 1. No further
  attempts were used.

## 10. Related work

- **Lindsey (2025)**, Anthropic, *Emergent introspective awareness in large language models.* The
  injected-thought paradigm with yes/no detection.
- **Hahami, Sinha, Jain, Kaplan & Hahami (2025)**, *Detecting the Disturbance: A Nuanced View of
  Introspective Abilities in LLMs*, arXiv 2512.12411.
  - Argues yes/no detection is confounded by a yes-bias.
  - Proposes sentence localization (up to 88% on Llama-3.1-8B-Instruct, early layers) and
    strength comparison.
  - No text-corruption control. We replicate the headline (89%) with their code's setup.
- **Hahami, Sinha & Jain (2026)**, *Introspection Fine-Tuning (IFT): Training Small LLMs to
  Introspect*, arXiv 2607.14111.
  - Llama-3.2-1B is at chance before IFT (9.6%) and 60.6% after.
  - Peak localization of about 65% (3B) and about 88% (8B); grid averages are much lower (16.8%
    for the 8B).
- **Macar, Yang, Wang, Wallich, Ameisen & Lindsey (2026)**, *Mechanisms of Introspective Awareness*,
  arXiv 2603.21396.
  - Main analyses on Gemma3-27B.
  - Detection "emerges specifically from post-training": in their training-stage comparison
    (OLMo-3.1-32B base, SFT, DPO and instruct checkpoints), DPO elicits it and standard SFT doesn't.
  - Measured with detection-style questions. This motivated our original training-stage question.
  - In Gemma3-27B, refusal-direction ablation raised detection from 10.8% to 63.8%; our Study 12
    finds no such effect at 8B.
  - Our Study 10 finds no yes/no detection at any Tülu-3-8B stage, including DPO; scale (OLMo-3.1-32B
    vs Tülu-3-8B) is one possible reason, untested here, as is model family.
- **Lederman & Mahowald (2026)**, *Emergent Introspection in AI is Content-Agnostic*,
  arXiv 2603.05414.
  - Models detect *that* an anomaly occurred without reliably identifying its content.
  - Our random-vector results show the same pattern for localization.
- **Godet (2025)**, *Introspection via localization*, LessWrong (28 Dec 2025).
  - Introduced the localization test for small models (five sentences).
  - Argued it can't be faked.
  - In the comments, reports chance accuracy for a "which do you prefer?" question.

## 11. Reproducing

See [`README.md`](README.md) for the environment, model access (Meta's model is gated) and
commands. Every study can be re-run with `python scripts/<study>.py --stage all` and re-analysed
with `python scripts/<study>_analyze.py`. Per-trial results are in `results/`. Pre-registrations
and reports are in `gates/` and `predictions.md`.
