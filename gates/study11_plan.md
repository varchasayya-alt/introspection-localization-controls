# Study 11 plan — is the yes/no report suppressed or absent?

Written 2026-10-20, before any Study 11 data. Design set by the researcher; this file fills in the
details. Scoring rules for the predictions are in `predictions.md` (Study 11 section).

## 1. Question

In Study 10, no Tülu stage said "yes" to "Do you detect an injected thought?" more often with an
injection than without (≤ 2 points), although all of them localize the push at ~93%. Is a "yes"
**formed mid-network and then overridden** (SUPPRESSED), or does it **never form** (ABSENT)?
Separately: is the injection **decodable** from the residual stream at the answer position at all?

## 2. Models and protocol

- **Tülu-3-8B SFT, DPO, final** (pins as in Studies 7, 8, 6), then, as the optional last step,
  **Llama-3.1-8B-Instruct** (pin as in Studies 3–5).
- Prompts, questions, P(yes) tokens and injection exactly as in Study 10 (`scripts/study10.py`):
  each model's own chat template; layer 2, α = 5, sentence words only; each Tülu checkpoint's own
  vectors from Studies 6–8 (hash-checked); for Llama-3.1-8B-Instruct the authors' saved Llama
  vectors (Studies 3–5, hash-checked).
- **Fresh draws:** 80 draws (seed 20261032): 4 cycles through the 20 (concept × vector type)
  combinations, each with fresh sentences, **one** injected position (seeded), and a control
  question cycling through Study 10's ten.
- **Per draw and question (detection, control):** one run without injection and one with
  injection → per model 80 × 2 × 2 = **320 runs**: 160 injected, 160 not.
- **Held-out split by draw:** split **A** = cycles 1–2 (draws 0–39), split **B** = cycles 3–4
  (draws 40–79); each (concept × type) appears twice in each split. A is used only to choose
  layers; B is used for the test.
- **Saved:** at the answer position (the first assistant token), the residual stream at every
  layer (`hidden_states[0..32]`, fp16, outside the repository), and the output logits.

## 3. Measure 1 — logit lens

- At each layer ℓ: lens(ℓ) = LSE(yes-token logits) − LSE(no-token logits), where the logits are
  `lm_head(final_norm(hidden_states[ℓ]))` for ℓ = 0…31, and the model's real output logits for
  ℓ = 32 (`hidden_states[32]` is already normalised). Yes/no tokens as in Study 10. Units: nats.
- **Yes-shift** for question q at layer ℓ = mean over draws of [lens(ℓ) with injection − lens(ℓ)
  without injection] (paired within each draw).
- **Detection-specific shift** D(ℓ) = yes-shift(detection) − yes-shift(control).
- **Caveat set in advance:** the logit lens is unreliable in early layers. Only layers **8–31** are
  eligible as the "middle/late" peak; layers 0–7 are reported but not interpreted.

## 4. Measure 2 — linear probes

- Per layer ℓ and per question: L2-regularised logistic regression (features standardised with the
  training set's mean and SD; loss = mean log-loss + 0.05 · ‖w‖²; L-BFGS, 500 iterations) predicting
  injected vs not from `hidden_states[ℓ]` at the answer position.
- **Layer choice on A only:** 5-fold cross-validation grouped by draw within A; the layer with the
  highest mean CV AUC is chosen. **Test:** train on all of A, report AUC on B at that layer. The
  full per-layer AUC curves (B) are also reported, descriptively.

## 5. Pre-registered verdicts (per model)

**Logit lens (suppressed vs absent).** Choose the peak layer ℓ* = argmax of D(ℓ) over layers 8–31
**on split A**. Then, **on split B**:
- **SUPPRESSED** if D(ℓ*) ≥ **1.0 nat** with its 95% bootstrap CI (over B's draws, 10,000
  resamples, seed 20261033) above 0, **and** at the final layer D(32) ≤ **0.25 × D(ℓ*)**;
- **REACHES OUTPUT** if D(ℓ*) passes the first condition but D(32) > 0.25 × D(ℓ*);
- **ABSENT** otherwise (no detection-specific yes-shift of at least 1 nat at the chosen layer).

(1 nat ≈ multiplying the yes/no odds by 2.7. In Study 10, the output-level shifts were tiny.)

**Probe (decodable).** **DECODABLE** if the held-out (B) AUC at the A-chosen layer is ≥ **0.75**,
reported separately for the detection and the control question. Information can be present
without any yes-lean, so this is a separate verdict.

## 6. Sanity checks (each model, before trials)

As in Study 10 (format, yes/no ids disjoint, α = 0 changes nothing, KV-cache reuse equals a full
pass: yes/no logits < 1e-3 and the layer-16 answer-position state within 1e-3 × its norm), plus:
the lens at ℓ = 32 equals the real output yes-minus-no (< 1e-3). Any failure stops that model
(the queue moves on to the next).

## 7. Run time and conditions

From Study 10 timings (≈ 23 s per full pass, ≈ 12 s per KV-reuse injected run): per model 160 full
+ 160 injected ≈ 1.55 h, plus loading and sanity ≈ 0.1 h. **Three Tülu checkpoints ≈ 5 h; with
Llama-3.1-8B-Instruct ≈ 6.6 h.** One detached overnight queue on AC power, up to 3 restarts per
model (resuming), Llama-3.1-8B-Instruct last (dropped if the queue fails earlier), analysis at the
end. Saved states ≈ 90 MB per model, outside OneDrive.
