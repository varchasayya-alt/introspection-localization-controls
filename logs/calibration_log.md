# Calibration log

Max **3** calibration attempts per gate. Each attempt is logged here *before* it runs
(what is being tuned and why), and its outcome is appended *after*.

## Gate 1 (positive control)

### Attempt 1 — logged 2026-10-15, before running
- What: Phase A sweep, layers {1,2,3,4,6,8} × α {2,5,8,14}, concepts volcano / violin /
  octopus, sentence set 0, 30 trials per cell (720 trials). Model Tülu-3-8B @ 6669437.
- Why these values: they bracket the published best cell for Llama-3.1-8B-Instruct (layer 2,
  α = 5) and its neighbours (layer 3 α 5, layer 4 α 8); layer 0 excluded by design.
- Pass: best cell ≥ 9/30. Selection rule: most correct; tie → lower α, then lower layer.
- If it fails, planned attempt 2 (to be re-logged before running): add layers {5, 10} and
  α {3, 20}. Planned attempt 3: contrastive-sentence concept vectors.
- Outcome (appended 2026-10-15 05:19, after the run): **PASS.** Best cells 26/30 at layer 2 α 5
  and layer 4 α 8; tie rule (lower α) selects **layer 2, α 5** for Phase B. No further
  calibration attempts needed. Full grid (correct of 30):

  | layer \ α | 2 | 5 | 8 | 14 |
  |---|---|---|---|---|
  | 1 | 23 | 15 | 21 | 18 |
  | 2 | 14 | **26** | 25 | 22 |
  | 3 | 7 | 25 | 25 | 25 |
  | 4 | 7 | 24 | 26 | 22 |
  | 6 | 10 | 22 | 25 | 16 |
  | 8 | 9 | 18 | 22 | 16 |
