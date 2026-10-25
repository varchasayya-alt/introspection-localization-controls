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

## Study 2 (text-intact search)

### Attempt 1 — logged 2026-10-15, before running
- Layers {2, 4, 8, 12, 16} × relative strength r {0.25, 0.5, 1.0} (α = r × median activation
  size at that layer on practice set 0). 30 injected + up to 30 copy+reader trials per setting
  (reader stops at 5 correct). Practice material: Gate 1 set 0; volcano, violin, octopus.
- Text-intact: reader ≤ 4/30. Pass: best text-intact setting has injected ≥ 8/30 and
  injected − reader ≥ 3/30. Selection: highest injected; tie → lower r, then lower layer.
- Outcome (appended 2026-10-16 07:02, after the run): **PASS.** 11 of 15 settings
  text-intact. Selected **layer 8, r 0.5** (α 3.1872): injected 14/30, reader 3/30.
  Attempt 2 not needed.

  | layer | r | injected | reader | text-intact | exact copies |
  |---|---|---|---|---|---|
  | 2 | 0.25 | 4/30 | 3/30 | yes | 30/30 |
  | 2 | 0.5 | 4/30 | 3/30 | yes | 30/30 |
  | 2 | 1.0 | 21/30 | 5/13 | no | 1/13 |
  | 4 | 0.25 | 4/30 | 3/30 | yes | 30/30 |
  | 4 | 0.5 | 7/30 | 3/30 | yes | 30/30 |
  | 4 | 1.0 | 22/30 | 5/7 | no | 0/7 |
  | 8 | 0.25 | 9/30 | 3/30 | yes | 30/30 |
  | 8 | 0.5 | 14/30 | 3/30 | yes | 30/30 |
  | 8 | 1.0 | 20/30 | 5/7 | no | 0/7 |
  | 12 | 0.25 | 9/30 | 3/30 | yes | 30/30 |
  | 12 | 0.5 | 9/30 | 3/30 | yes | 30/30 |
  | 12 | 1.0 | 8/30 | 5/7 | no | 0/7 |
  | 16 | 0.25 | 6/30 | 3/30 | yes | 30/30 |
  | 16 | 0.5 | 4/30 | 3/30 | yes | 30/30 |
  | 16 | 1.0 | 5/30 | 3/30 | yes | 28/30 |

### Attempt 2 — logged 2026-10-15, before running (runs automatically only if attempt 1 fails)
- Same layers × r {0.35, 0.7, 1.4}; same trials, rules and selection. Fixed now, not adjusted
  after attempt 1. If attempt 2 also fails → STOP and write up.
- Outcome: not run (attempt 1 passed).

## Study 14 — steering layer and direction (Llama-3.2-3B-Instruct)

### Attempt 1 of 1 — logged 2026-10-24, before running
- 100 calibration draws (calibration-half sentences), detection question, each clean and injected
  (Study 13 3B injection: layer 2, α 7.937, 3B concept vectors). Answer-position states at every block
  output saved outside the repository.
- Layer rule: Study 11 probe fitted on draws 0–49, AUC on draws 50–99, blocks 0–26; highest AUC
  (3 decimals), tie → earlier block. Direction = mean(injected) − mean(clean) over all 100 draws at
  that block. Stop if AUC < 0.75. No further attempts; any re-run would be a logged deviation.
- Outcome (appended 2026-10-24 22:32, after calibration, before any test data): **PASS (no stop).**
  Held-out AUC 0.5 at blocks 0–2 (as expected), 0.959 at block 3, 0.999 at block 4, **1.000 at
  blocks 5–11** (tie → earliest), 0.988–0.998 at blocks 12–26. **Steering layer = block 5.**
  ‖v‖ = 0.1231 = 0.025 × the mean clean residual norm at the answer position (4.88), so k = 8 is a
  push of about 0.2 × that norm. Direction file (`direction.pt`, outside the repository) SHA-256
  `706f02782e723062906a0cc09b57bc06fa76436a701011c192fe0555f763b475`; raw float32 bytes SHA-256
  `141ea0a84244cafd3515077cc6674cb68f72f7179d3d410591e5e4763657cfd8`
  (`results/study14/calibration.json`).

## Study 15 — dose calibration on the positive control (Llama-3.2-3B-Instruct, block 18)

### Attempt 1 of 1 — logged 2026-10-25, before running
- Direction (computed from Study 14's saved calibration states, before this log entry): unit
  injection direction at `hidden_states[19]`, SHA-256 `d98cb7e3…c734`; raw norm 0.639; typical norm
  N = 16.211 (`results/study15/direction.json`).
- 100 dose-calibration draws (Study 14 calibration half), detection question, clean; the unit
  (yes − no) direction × f · N at block 18, answer position, for 40 geometric f from 0.02 to 16.
- Doses = smallest f with yes-rate ≥ 25%, 50%, 90%. Stop if 90% is never reached. Only the positive
  control is run. No further attempts; any re-run would be a logged deviation.
- Outcome (appended 2026-10-25 16:24, after dose calibration, before any test data): **PASS (no
  stop).** The positive control's yes-rate is 0% up to f = 0.2616, 3% at f = 0.3105, **93% at
  f = 0.3685**, 100% from f = 0.4374 to 2.4283; at f = 2.88–5.72 every answer is "neither"; at
  6.79–8.06 a mix of "no" and "neither"; 100% "yes" again from f = 9.57. Because the curve jumps
  from 3% to 93% between two neighbouring grid points, the rule gives **the same dose for all three
  targets: s25 = s50 = s90 = 0.3685 × N** (push norm 5.97). The test runs as frozen (three identical
  dose slots); the bars use the top dose, so scoring is unaffected, but the intermediate points of
  the dose curve are lost. No re-calibration (it would be a second attempt).
