# Study 13 report — size curve in the Llama family (1B, 3B, 8B)

Plan: `gates/study13_plan.md` (d99992c). Predictions: `predictions.md` (e5ef057). Frozen setup:
8a6f67a. Deviation D6 (ff5d91d, bea42b6). One queue, 2026-10-21 21:35 → 2026-10-22 03:13; every
step on its first attempt, no errors. Tables: `results/study13_summary.md`,
`results/study13_d6_summary.md`. 8B reused from Studies 4 and 11 (same protocol).

## Settings used (push size matched at 2.186 × the mean residual norm at the injection point)

| Model | Layer | Mean norm | α |
|---|---|---|---|
| Llama-3.2-1B (primary) | 1 | 2.10 | 4.59 |
| Llama-3.2-1B (secondary) | 2 | 2.98 | 6.51 |
| Llama-3.2-3B | 2 | 3.63 | 7.94 |
| Llama-3.1-8B (reference) | 2 | 2.29 | 5.00 |

## Size table

| | 1B (layer 1) | 3B (layer 2) | 8B (layer 2) |
|---|---|---|---|
| Localization, concept | **not testable** (S7 15/20); *D6, not scored:* 81/200 (40.5%) | **189/200 (94.5%)** | 195/200 (97.5%) |
| Exact-copy trials: injected / reader | *D6:* 33/68 (48.5%) / 10/68 (14.7%) → **FAIL** (injected < 50%) | 124/129 (96.1%) / 12/129 (9.3%) → **PASS** | 145/150 (96.7%) / 15/150 (10%) → PASS |
| Localization, random | *D6:* 57/200 (28.5%) | 183/200 (91.5%) | 182/200 (91%) |
| Yes/no: hits − false alarms [95% CI] | 1.8 [1.1, 2.5] (hits 16.8, FA 15.0) | 1.2 [0.8, 1.6] (hits 7.6, FA 6.4) | 1.6 [1.1, 2.2] (hits 3.9, FA 2.2) |
| Control yes-rate (injected / none) | 8.5 / 7.1 | 5.6 / 6.0 | 0.4 / 0.3 |
| Probe AUC, detection / control | 0.998 / 1.000 | 0.991 / 0.943 | 0.997 / 0.927 |

**Secondary, 1B at layer 2:** localization *D6:* 71/200 (35.5%); exact-copy trials 26/59 (44%) vs
reader 9/59 (15%) → FAIL; random 38/200 (19%). Yes/no: hits − FA 2.1 [1.3, 3.0]; control yes 7.4 /
5.8; probe AUC 1.00 / 1.00.

## Summary

**Sensing grows with size; reporting doesn't.** Intact-text localization is weak in 1B (about 45–49%
on exact copies, just under the 50% bar, though well above its reader's 15%), and near-ceiling at 3B
and 8B (96–97%, main test PASS); random vectors follow the same pattern. Explicit yes/no reporting is
flat at about 1–2 points at every size, while the injection is linearly decodable everywhere.

**Trends (pre-registered):** localization **not testable** (1B stopped at S7); report **flat**. With
the D6 1B run (not scored), the localization trend would be **rises**.

## Prediction scoring (rules d99992c)

P1, P2 not scorable (1B stopped at S7) · P3 ✘ (3B 94.5%, predicted 50–75%) · P4 ✔ (3B PASS) · P5 ✘
(not testable) · P6 ✘ (report flat). **1/4 scorable.** Running total over Studies 1–13: **30/69**.
For information only, under D6 P1 (c), P2 (b) and P5 (a, rises) would all have been correct.
