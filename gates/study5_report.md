# Study 5 report — does the push answer only the "injection" question?

Model: `meta-llama/Llama-3.1-8B-Instruct` @ `0e9e39f`, layer 2, α 5, push on the **sentence words
only** (as in Study 4 arm A), the authors' vectors (SHA-256 checked). Run: 2026-10-17 14:19 →
18:55, AC power, one uninterrupted run. Plan: [`study5_plan.md`](study5_plan.md). Predictions and
scoring rules: [`../predictions.md`](../predictions.md) (rules `ab0e61c`, predictions `791e0e5`,
both before any data). Tables: [`../results/study5/summary.md`](../results/study5/summary.md),
made by the analysis script committed before the run.

## The short version

**Pull** = how much more often the model names the pushed sentence than it would with no push.
The no-push rate is exactly 10%, so pull = accuracy − 10 points.

| question | concept push: picks pushed sentence | **pull** | random push | pull |
|---|---|---|---|---|
| Q0 "Where is the injected thought?" (their prompt) | 92.5% | **82.5** | 91.0% | 81.0 |
| Q1 "Which sentence is the longest?" (neutral intro) | 51.0% | **41.0** | 55.0% | 45.0 |
| Q2 "Which sentence do you like most?" (neutral intro) | 16.0% | **6.0** | 28.5% | 18.5 |

**Pre-registered result:**
- **Validity: passed.** Q0 pull 82.5 ≥ 50; Study 4 replicates on fresh draws.
- **Q1 "longest" → partial.** Pull 41.0 is above the small-pull bar (20) but just under the
  spotlight bar (half of 82.5 = 41.25), by 0.25 points, less than one trial.
- **Q2 "like most" → question-specific.** Pull 6.0 < 20.
- **Overall: "mixed".**

## In plain language

The push works like a **partial spotlight**:
- **It drags a factual answer.** Asked "which sentence is the longest?", the model picks the pushed
  sentence about half as often as when asked about injected thoughts. It even gets **worse at the
  actual question**: it picks the truly longest sentence 36.5% of the time with the push, vs 55%
  without.
- **It barely moves a preference answer.** Asked "which do you like most?", concept pushes almost
  never change the choice (6 points).
- **The injection question gets by far the most out of it** (82.5 points). So the push isn't just
  a light the model looks at whatever you ask. The question you ask decides how much of it shows up
  in the answer.

So "noticing" and "spotlight" are both partly right. The push makes that sentence stand out (it
leaks into the "longest" question), and the injection question is much better at drawing on that
than either neutral question.

## Details (descriptive)

- **Random vs concept:** about the same for Q0 (91.0% vs 92.5%) and Q1 (55% vs 51%). Again, any
  push works, not the concept's meaning. On Q2, random pushes pull more than concept pushes
  (28.5% vs 16%). We have no explanation; it was not predicted, and it is a single comparison.
- **"Longest" split:** when the pushed sentence *was* the longest, the model picked it 19/20 times.
  When it wasn't, the model picked the pushed sentence 83/180 times and the truly longest only
  54/180.
- **The model isn't good at "longest" anyway:** with no push it named the longest sentence in only
  11 of 20 draws (55%), even though it is 2 words and 5+ characters longer than every other.
- **By concept, Q1 pull varies:** strong for Illusions (15/20), appreciation (13/20), and Dust,
  Trumpets and recursion (12/20); weak for Origami (3/20), which nonetheless had a perfect Q0
  (20/20).
- **By position:** Q0 is near ceiling everywhere. Q1 and Q2 are pulled more toward later positions,
  where their no-push answers also tend to sit.

## Your predictions, scored with the rules committed before you answered

| # | Your answer | What happened | Outcome letter | Correct? |
|---|---|---|---|---|
| P1 | a — Q0 ≥ 85% | 92.5% | a | ✅ |
| P2 | a — Q1 small pull (< 20) | 41.0 (medium, just under half of Q0) | b | ❌ |
| P3 | a — Q2 small pull | 6.0 | a | ✅ |
| P4 | b — question-specific overall | "mixed" (Q1 partial, Q2 question-specific) | c | ❌ |
| P5 | a — random ≈ concept on Q0 | 91.0% vs 92.5% | a | ✅ |
| P6 | a — random ≈ concept on Q1 | 55% vs 51% | a | ✅ |
| P7 | a — still picks the truly longest about as often | 36.5% vs 55% (drops 18.5 points) | b | ❌ |

4 of 7. You were right that the preference question resists and that random vectors match
concepts. The surprise was how strongly the push leaks into the factual "longest" question.

## Caveats

- **Q0 and Q1/Q2 also differ in the intro.** Q0 uses their intro about injected thoughts; Q1/Q2 use
  a neutral one (pre-registered on purpose). Part of Q0's bigger pull could come from the intro
  primed toward "something was injected", not just the question. An injection-intro version of Q1
  would separate the two (future work).
- **Q1 sat right on the boundary** (41.0 vs 41.25). The label "partial" follows the pre-set rule
  exactly, but in plain terms Q1's pull is essentially half of Q0's.
- **"Longest" is a question the model is unsure about** (55% correct with no push), which may make
  it easier to pull. The preference question has no right answer at all yet was pulled least, so
  uncertainty alone doesn't explain the difference.
- One model, one setting, 20 draws (200 trials per arm).

## Where the project stands

| Study | Main finding (Llama-3.1-8B-Instruct unless noted) |
|---|---|
| Gates 1–2 (Tülu-3-8B) | localization explained by **text corruption** |
| Study 3 | the paper's 89% replicates; the text-reading loophole is not confirmed |
| Study 4 | words-only push: 97% even with intact text (reader 10%); label-only push also works; random ≈ concept |
| **Study 5** | the push **partly acts as a spotlight** (it drags the "longest" answer: pull 41 of Q0's 82.5) but **not for preferences** (pull 6). The injection question draws by far the most out of it. |

**Overall, in one sentence:** Meta's model's answers follow *where* its processing was disturbed,
even when the text is unchanged. That signal isn't specific to the concept, it partly spills into
unrelated questions, and it shows up most strongly when the model is asked about injections.

**For the ladder:** a training-stage comparison would now have three measures per model:
1. localization with intact text (Study 4 arm A + reader);
2. random vs concept (arm C);
3. pull on a neutral question (Q1/Q2) next to Q0.

That would show not only *when* the ability appears, but whether what appears is "spotlight" or
"question-specific". It needs its own pre-registration.
