# Study 12 report — can the yes/no report be unlocked at 8B?

Plan: `gates/study12_plan.md` (2e49802). Predictions: `predictions.md` (8e5efb9). Frozen setup:
41bf9ba. One queued run, 2026-10-20 13:31 → 2026-10-21 04:09; every model on its first attempt,
all sanity checks passed, no deviations. Full tables: `results/study12_summary.md`.

## Results (points = 100 × P(yes); report = hits − false alarms, 95% CI over draws)

| Model | Arm | Hits | False alarms | **Report** [95% CI] | Control yes (inj. / none) | Unlocked |
|---|---|---|---|---|---|---|
| Tülu final | baseline | 1.6 | 1.1 | 0.5 [0.3, 0.7] | 0.3 / 0.4 | no |
| | informative | 2.2 | 1.8 | 0.5 [0.1, 0.9] | 0.2 / 0.2 | no |
| | ablation | 7.2 | 5.7 | 1.5 [0.9, 2.1] | 1.5 / 1.7 | no |
| | both | 9.2 | 7.9 | 1.2 [0.3, 2.4] | 0.8 / 0.9 | no |
| Llama-3.1-8B-Instruct | baseline | 3.8 | 2.1 | 1.6 [1.0, 2.3] | 0.4 / 0.3 | no |
| | informative | 9.8 | 1.5 | **8.3** [4.2, 13.5] | 1.2 / 1.2 | no |
| | ablation | 4.7 | 3.3 | 1.4 [0.8, 2.0] | 0.7 / 0.4 | no |
| | both | 9.9 | 2.2 | **7.6** [4.2, 12.0] | 3.0 / 2.9 | no |
| Tülu SFT | baseline | 18.8 | 16.7 | 2.2 [1.1, 3.2] | 6.3 / 6.2 | no |
| | informative | 11.9 | 10.7 | 1.2 [0.4, 2.1] | 3.2 / 3.1 | no |
| | ablation | 28.1 | 26.7 | 1.4 [−0.0, 2.7] | 18.4 / 19.3 | no |
| | both | 15.2 | 13.4 | 1.8 [0.7, 3.1] | 11.1 / 12.2 | no |

**Ablation validity** (refused share of 32 held-out harmful prompts, without → with ablation): Tülu
final 100% → 0% (layer 16), Llama 94% → 0% (layer 12), Tülu SFT 100% → 0% (layer 16). All valid.

**Safety hygiene held:** the ablation was applied only through forward hooks at runtime; no model
weights were modified or saved; nothing was generated for any prompt. Stored per model: one
direction vector (18 KB), JSON summaries and yes/no probabilities (606 KB for the whole study; the
prompt lists stay outside the repository).

## Summary

**Neither lever unlocks a yes/no report at 8B.** Refusal ablation raises "yes" for both injected and
non-injected trials alike (a yes-bias, strongest in SFT), not detection. The informative prompt is the
only lever that adds detection-specific "yes" — in Llama-3.1-8B-Instruct only (report ≈ 8 points, CI
above 0), well below the 20-point bar.

## Prediction scoring (rules 2e49802)

P1 ✘ (Tülu final + informative: not unlocked) · P2 ✔ · P3 ✔ · P4 ✔ · P5 ✘ (both levers: unlocked in
neither) · P6 ✘ (no control yes-rate above 20 in Tülu final or Llama; Tülu SFT's ablation arm reached
19.3, not scored by the rule). **3/6.** Running total over Studies 1–12: **29/65**.
