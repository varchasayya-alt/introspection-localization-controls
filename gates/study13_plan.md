# Study 13 plan — a size curve within the Llama family (1B, 3B, 8B)

Written 2026-10-21, before any Study 13 data. Design set by the researcher; this file fills in the
details. Scoring rules for the predictions are in `predictions.md` (Study 13 section).

## 1. Question

Do intact-text localization and yes/no reporting change with model size within one family?
- `meta-llama/Llama-3.2-1B-Instruct` @ `9213176726f574b556790deb65791e0c5aa438b6` (16 layers, width 2048)
- `meta-llama/Llama-3.2-3B-Instruct` @ `0cb88a4f764b7a12671c53f0838cd831a0843b95` (28 layers, width 3072)
- `meta-llama/Llama-3.1-8B-Instruct` (32 layers, width 4096): **reference, reused** (§5).

**Noted in advance:** the IFT paper (arXiv 2607.14111) reported Llama-3.2-1B at chance on
localization (9.6%) before introspection fine-tuning, without text-corruption controls. Study 13
tests whether that holds under our protocol and controls.

## 2. Matching across sizes

- **Layer (primary):** the same relative depth as layer 2 of 32: layer index = round(2 × N / 32)
  → **1B: layer 1**, **3B: layer 2** (8B: layer 2). As in all earlier studies, vectors are read at
  `hidden_states[ℓ]` and the push is added at the output of `model.model.layers[ℓ]`.
- **Strength:** match the push size relative to normal activity at the injection point. In
  Llama-3.1-8B, α = 5 was 2.186 × the mean residual norm over sentence-word tokens (5 / 2.287,
  Study 6). For each model, α = **2.186 × that model's mean norm** at its injection point, measured
  on the clean localization prompts of the first 5 draws before any injection trial. The actual
  ratio is reported.
- **Secondary arm (labelled secondary):** absolute layer 2 for 1B (with its own vectors and matched
  α). For 3B the primary layer already is layer 2, so no separate secondary run is needed.

## 3. Protocol per model and layer setting

**Localization = Study 4 arm A + arm C** (authors' prompt in the Llama chat format; push on the
sentence words only): concept vectors **recomputed from each model** with the authors' recipe (their
hand-built Llama chat string, which is native to Llama 3.2; mean over prompt tokens "avg" and last
token "last"; mean differences) at `hidden_states[ℓ]`; 20 fresh draws × 10 positions (seed
20261037, shared by 1B and 3B); **arm A** concept + copy-back (D3 fix) + clean reader; **arm C**
random unit vector × α (seed 20261038). Sanity as in Study 4 (S1–S5, S8, and **S7**: clean copies
exact in ≥ 19/20 draws, else the localization part stops for that model setting).

**Yes/no and probes = Study 11's design** (Study 10's prompts: detection and control questions):
80 fresh draws (seed 20261039), one injected position each, each question with and without the
push; residual stream at the answer position saved at every layer (outside the repository).
- **Hits** = mean P(yes) detection, injected; **false alarms** = detection, not injected;
  **report = hits − false alarms** (points, 95% bootstrap CI over draws, seed 20261040); control
  yes-rates (injected, not injected).
- **Probe AUC:** as Study 11 (layer chosen by 5-fold CV on draws 0–39, tested on draws 40–79);
  DECODABLE if held-out AUC ≥ 0.75.

## 4. Pre-registered tests

**Main test per model (primary layer), same as Study 4:** among arm-A trials with an exact copy of
the target sentence: < 40 → NOT TESTABLE; injected ≥ 50% **and** reader ≤ 15% → PASS; otherwise
FAIL.

**Intact-text localization score** per model = injected accuracy among exact-copy trials (defined
only if ≥ 40 such trials).

**Size trend, localization** (1B, 3B, 8B; primary layers):
- **Rises** if 8B − 1B ≥ 20 points and each step up in size is no more than 5 points lower than the
  step before (3B ≥ 1B − 5, 8B ≥ 3B − 5);
- **Flat** if all three are within 10 points of each other;
- **Mixed** otherwise; **not testable** if any model's score is undefined (< 40 exact-copy trials).

**Size trend, report (hits − false alarms):**
- **Rises** if 8B − 1B ≥ 10 points and 3B ≥ 1B − 3 and 8B ≥ 3B − 3;
- **Flat** if all three are within 5 points of each other;
- **Mixed** otherwise.

The secondary 1B layer-2 arm is reported beside the primary, but does not enter the trend rules.

## 5. The 8B reference (reused, same protocol)

- **Localization:** Study 4 arms A and C (authors' vectors, which equal the recipe's output, cosine
  1.0 in Study 3; layer 2; α = 5 = 2.186 × norm by definition): arm A 195/200, exact-copy trials 150,
  injected 145 (96.7%), reader 15 → PASS; arm C 182/200.
- **Yes/no and probes:** Study 11's Llama-3.1-8B-Instruct run (identical design and prompts).
- Different sentence draws from 1B/3B (fresh draws in every study); otherwise identical protocol.

## 6. Run time and conditions

Scaled from the 8B timings by parameter count (1B ≈ 0.2×, 3B ≈ 0.45× of 8B per forward pass):

| Step | 1B (layer 1) | 1B (layer 2, secondary) | 3B (layer 2) |
|---|---|---|---|
| vectors + strength calibration + sanity | 0.15 h | 0.15 h | 0.3 h |
| localization (A + copy/reader + C) | 0.9 h | 0.9 h | 2.0 h |
| yes/no + probes (320 runs) | 0.3 h | 0.3 h | 0.7 h |
| **total** | **1.35 h** | **1.35 h** | **3.0 h** |

**≈ 5.7 hours** in one detached overnight queue on AC power (order: 1B primary, 3B, 1B secondary),
up to 3 restarts per step (resuming), analysis at the end.
