# Study 11 report — is the yes/no report suppressed or absent?

Plan: `gates/study11_plan.md` (c7eb51d). Predictions: `predictions.md` (cdecf60). Frozen setup:
7b68386. One queued run, 2026-10-20 00:48 → 07:26; every model on its first attempt, all sanity
checks passed, no deviations. Full tables and curves: `results/study11_summary.md`.

## Results (D = detection-question yes-shift minus control-question yes-shift, in nats; peak layer chosen on split A, values on held-out split B)

| Model | Peak layer (A) | D at peak (B) [95% CI] | D at output (B) | Verdict | Probe AUC (B): detection / control | Decodable |
|---|---|---|---|---|---|---|
| Tülu SFT | 30 | 0.44 [0.19, 0.69] | 0.09 | **ABSENT** | 0.991 / 0.989 | yes / yes |
| Tülu DPO | 30 | 0.58 [0.35, 0.83] | 0.31 | **ABSENT** | 0.990 / 0.976 | yes / yes |
| Tülu final | 25 | 0.75 [0.45, 1.05] | 0.33 | **ABSENT** | 0.987 / 0.971 | yes / yes |
| Llama-3.1-8B-Instruct | 22 | 1.46 [0.44, 2.57] | 0.36 | **SUPPRESSED** | 0.997 / 0.927 | yes / yes |

- **Shape:** the detection-specific yes-lean is flat (~0) up to about layer 12, rises from about
  layer 16 to a late-layer plateau, then falls in the last few layers. In the Tülu checkpoints the
  plateau is 0.4–0.75 nats, **below the pre-registered 1-nat bar**, so the verdict is ABSENT; it
  falls to about 0.1 (SFT) or 0.3 (DPO, final) at the output. In Llama-3.1-8B-Instruct the
  held-out peak (1.46 nats, layer 22) passes the bar and the output value (0.36) is just under a
  quarter of it (0.364), so the verdict is SUPPRESSED, narrowly, with a wide CI.
- **Probes:** "injected or not" is decodable from the answer position in every model, for both
  questions (held-out AUC 0.93–1.00, best around layers 8–15). The information is there whether or
  not the question asks about it.
- **Caveat:** the Tülu lean is real but small (CIs above 0, below 1 nat), and the Llama verdict sits
  on the shrink boundary. Logit-lens values in layers 0–7 are not interpreted (set in advance).

## Prediction scoring (rules c7eb51d)

P1 ✘ P2 ✘ P3 ✘ (Tülu SFT, DPO, final: predicted SUPPRESSED, all ABSENT) · P4 ✔ (decodable at all
three) · P5 ✔ (final: detection vs control AUC within 0.10) · P6 ✔ (Llama: SUPPRESSED). **3/6.**
Running total over Studies 1–11: **26/59**.
