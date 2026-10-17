# IntrospectionStages

**Question:** Which training stage (base → SFT → preference/DPO → final) creates a small open
model's ability to *notice* an injected concept, measured with the confound-free
**sentence localization** task (inject a concept vector into 1 of 10 sentences, ask which
sentence felt different; chance = 10%)?

## How this project works

- **Gates.** Work proceeds in gates. Each gate has a pass rule and a stop rule written down
  *before* it runs.
- **Pre-registration.** Before any experiment, predictions are recorded verbatim in
  [`predictions.md`](predictions.md) and committed. Only then does the run start.
- **Calibration** attempts are capped at 3 per gate and logged in
  [`logs/calibration_log.md`](logs/calibration_log.md).
- **Deviations** from the plan are logged in [`logs/deviations.md`](logs/deviations.md)
  *before* the deviating run.
- **Validity first.** Positive control and confound checks come before main tests. A result is
  never re-judged with a different metric after it has been seen.

## Status

| Gate | What | Status |
|---|---|---|
| 0 | Feasibility: hardware timing, checkpoint ladders, control design | Report: [`gates/gate0_report.md`](gates/gate0_report.md) |
| 1 | Positive control (one final model localizes above chance) | **PASS** — Tülu-3-8B 53.7% (161/300, chance 10%). Report: [`gates/gate1_report.md`](gates/gate1_report.md) |
| 2 | Cheap-trick controls on the positive control | **FAIL** — a clean reader of the perceived text scores 74.7% (> injected 45.3%); random vectors 53.3%. Stop rule: no "introspection" framing |
| 3 | Training-stage ladder | Not run (Gate 2 stop rule); next step needs a new pre-registration |
| S2 | Text-intact search (new pre-registered question) | Practice PASS (layer 8, ½×: 14/30, reader 3/30); **fresh test FAIL** (injected 36/200 = 18%, reader 20/200, random 28/200). Report: [`gates/study2_report.md`](gates/study2_report.md) |
| S3 | Replicate *Detecting the Disturbance* headline on Llama-3.1-8B-Instruct + text control | **R1 PASS** (89.0%, paper 88%); **R2 loophole not confirmed** (clean reader 42.0%); with the target copied exactly: injected 94% vs reader 5%. Report: [`gates/study3_report.md`](gates/study3_report.md) |
| S4 | Sentence-only vs label-only injection (Llama-3.1-8B-Instruct) | **Main test PASS**: words only 97.5%, and with the text copied exactly 97% vs reader 10%; label only 64% (label copy intact 200/200); random vectors 91%. Pre-registered conclusion: **both routes work**. Report: [`gates/study4_report.md`](gates/study4_report.md) |

## Environment

- Windows 11 laptop, Intel Core Ultra 7 255U (12 cores / 14 threads), 32 GB RAM, CPU only.
- Python venv lives **outside OneDrive** at `%USERPROFILE%\.venvs\introspection` so that
  OneDrive does not try to sync gigabytes of packages. Model weights go to the default
  Hugging Face cache (`%USERPROFILE%\.cache\huggingface`), also outside OneDrive.

## Related prior work

- varchasayya-alt, *llm-consciousness-indicators-negative-results*
  (github.com/varchasayya-alt/llm-consciousness-indicators-negative-results)
