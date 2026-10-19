# Studies 7–8 report — Tülu-3-8B SFT and DPO checkpoints with the Study 6 protocol

Plan: `gates/study7_8_plan.md` (6bae069). Predictions: `predictions.md` (b70f297). Frozen setup:
9331c85. One queued run on 2026-10-18 (Study 7 02:20–07:50, Study 8 07:50–13:25), first attempt
each, no restarts, no deviations. Full tables: `results/study7_8_summary.md`.

## Result

The SFT and DPO checkpoints behave like the final model: **all three Tülu stages localize the push
~93% with intact text, and random vectors work just as well.** Under this protocol the ability is
already present after supervised fine-tuning; preference tuning and RL do not change it.

| Checkpoint | Arm A (concept) | Reader | Exact-copy trials: injected / reader | Main test | Arm C (random) |
|---|---|---|---|---|---|
| SFT (Study 7) | **186/200 (93.0%)** | 24/200 | 181/195 (93%) / 20/195 (10%) | **PASS** | 187/200 (93.5%) |
| DPO (Study 8) | **185/200 (92.5%)** | 23/200 | 178/193 (92%) / 18/193 (9%) | **PASS** | 183/200 (91.5%) |
| Final (Study 6) | 186/200 (93.0%) | 28/200 | 178/192 (93%) / 20/192 (10%) | PASS | 186/200 (93.0%) |

Sanity checks passed for both (clean copies exact in 20/20 draws). Activation size at layer 2 was
nearly identical across stages (2.31, 2.31, 2.32), and the recomputed vectors were similar to the
authors' Llama vectors in the same way as Study 6 (mean cosine "avg" 0.89, "last" 0.17).

## Prediction scoring (rules 6bae069)

All six predictions correct (SFT and DPO: ≥ 75%, PASS, about the same as final). **6/6.** Running
total over Studies 1–8: **17/45**.

## Limits

One layer and strength, 20 draws per checkpoint, one model family. The base model (before SFT) was
not tested here.
