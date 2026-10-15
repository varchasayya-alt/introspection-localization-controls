# Predictions (pre-registration)

Predictions are recorded **verbatim** from the researcher, committed, and only then is the
corresponding experiment run. Nothing in this file is edited after the run it refers to.

## Gate 1 — positive control (allenai/Llama-3.1-Tulu-3-8B, sentence localization)

Recorded 2026-10-15, **before** any Gate 1 data was collected (no model weights downloaded yet).

### Original questions (from the Gate 0 report)

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

### Researcher's predictions — verbatim

The researcher answered a simplified multiple-choice version (their teacher rephrased the 7
questions as a "whisper game"). Letter answers and mapping, exactly as given:

> Q1 (fresh-test accuracy): "c" = 3 or more in 10, so it PASSES the 30% bar. (Best practice setting: not predicted separately; implied >=30%.)
> Q2 (random-vector accuracy): "a" = random vectors work just as well as real concepts.
> Q3 (clean text reader): "b" = the reader does NOT match the injected model, i.e. not within 10 points; real noticing.
> Q4 (layers): "c" = works best when injected LATE in the network, not only early (<=4).
> Q5 (injected sentence copied back exactly): "b" = about half the time.
> Q6/Q7 (noticing vs reading changed text, if it passes): "b" = a little; mostly noticing, but weak.

Mapping to the original questions: whisper-game Q1 → original Q2 (and Q1 implied ≥ 30%);
Q2 → original Q3; Q3 → original Q4; Q4 → original Q5; Q5 → original Q6; Q6/Q7 → original Q7.

### How each prediction will be scored (set by Claude before the run, not by the researcher)

These rules are written down now so the predictions can't be re-judged after seeing results.

| Prediction | Counts as correct if… |
|---|---|
| Q1 "c" | Phase B accuracy ≥ 30% (≥ 90 / 300). Implied part: best Phase A cell ≥ 30% (≥ 9 / 30). |
| Q2 "a" | Random-vector accuracy (C3) is within 10 points of concept accuracy at the same cell (same threshold as the C3 labelling rule). |
| Q3 "b" | Perceived-text reader (C1) accuracy is **more** than 10 points below the injected model's accuracy on the same 150 trials. |
| Q4 "c" | The best Phase A cell is at a layer > 4 (i.e. layer 6 or 8). **Limitation:** the approved sweep only covers layers 1–8 of 32 (the first quarter of the network), so truly late layers are *not tested* in Gate 1; this prediction can only be checked within that range. |
| Q5 "b" | The injected (target) sentence is copied back exactly (every token is the model's top prediction) in 35–65% of the C1 trials. |
| Q6/Q7 "b" | Gate 1 passes **and** Gate 2 passes ("mostly noticing") **and** Phase B accuracy is below 50% ("weak"). Not scored if Gate 1 fails. |

---
