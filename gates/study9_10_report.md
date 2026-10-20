# Studies 9–10 report — base-model localization and yes/no detection across the Tülu ladder

Plan: `gates/study9_10_plan.md` (e5841f5). Predictions: `predictions.md` (89e3f4f). Frozen setup:
f13d15b. One queued run, 2026-10-19 15:41 → 2026-10-20 00:09; every step on its first attempt, no
restarts, no deviations. Full tables: `results/study9_10_summary.md`.

## Study 9 — base Llama-3.1-8B (few-shot format)

Positive control (visible ALL CAPS, no injection) **50/50** → format valid. Clean copies exact 20/20.

| | Result |
|---|---|
| Arm A (concept) | **188/200 = 94.0%** (95% CI 89.8–96.5%) |
| Reader | 27/200 |
| Exact-copy trials (185) | injected 175 (95%), reader 17 (9%) → **main test PASS** |
| Arm C (random) | **188/200 = 94.0%** |

Intact-text, content-agnostic localization is already present in the **base model**, before any
instruction tuning.

## Study 10 — yes/no detection (points = 100 × P(yes))

| Checkpoint | Hits (det., injected) | False alarms (det., none) | Hits − false alarms | Control, injected | Control, none | Detection score (pre-registered) |
|---|---|---|---|---|---|---|
| SFT | 18.0 | 16.0 | **1.9** | 7.2 | 6.9 | 10.7 (CI 8.0–13.4) |
| DPO | 1.8 | 1.3 | **0.5** | 0.3 | 0.3 | 1.5 (CI 1.1–2.0) |
| Final | 1.4 | 1.0 | **0.4** | 0.4 | 0.4 | 1.1 (CI 0.7–1.5) |

All three answered the control questions "no" in 20/20 draws (valid). **Sensing-vs-reporting
split: not supported** — DPO and final score ~9–10 points *below* SFT, not 20 above.

SFT's 10.7 is mostly a yes-habit for the detection question, not detection: it says "yes" 16% of
the time with no injection, and the injection adds only ~2 points. No checkpoint reports the
injection in yes/no form (hits − false alarms ≤ 2 points), although all of them localize it at 93%.

## Prediction scoring (rules e5841f5)

P1 ✔ P2 ✔ P3 ✔ P4 ✔ P5 ✔ (SFT 10.7, bin 10–30) P6 ✘ (final 1.1, < 10) P7 ✘ (not supported)
P8 ✔ (false alarms 1.0%). **6/8.** Running total over Studies 1–10: **23/53**.

Note: P5 scored "correct" on the pre-registered detection score, which the false-alarm data show is
mostly a yes-habit.
