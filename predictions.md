# Predictions (pre-registration)

Predictions are recorded **verbatim** from the researcher, committed, and only then is the
corresponding experiment run. Nothing in this file is edited after the run it refers to.

## Gate 1 — positive control (allenai/Llama-3.1-Tulu-3-8B, sentence localization)

_Status: waiting for the researcher's predictions. No experiment has been run._

Questions:

1. **Calibration sweep:** best cell accuracy (any layer × strength), in %?
2. **Held-out test** at the pre-selected cell (300 trials, chance 10%): accuracy in %? Will it
   pass the ≥ 30% rule (yes / no)?
3. **Random-vector control** (same size, random direction): accuracy in %?
4. **Text-only reader control** (a clean model sees only the text the injected model
   "perceived"): accuracy in %? Will it come within 10 points of the injected model?
5. **Layer pattern:** will localization only work at early layers (≤ 4 of 32)?
6. **Transcription:** in what fraction of trials will the injected sentence be copied back
   *exactly*?
7. **Overall:** if Gate 1 passes, will it look like "noticing" or like "reading changed text"?

---
