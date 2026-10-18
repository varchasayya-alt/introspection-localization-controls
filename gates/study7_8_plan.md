# Studies 7 and 8 plan — Tülu-3-8B SFT and DPO checkpoints with the Study 6 protocol

Written 2026-10-18, before any Study 7 or 8 data. Design set by the researcher. Scoring rules for
the predictions are in `predictions.md` (Studies 7–8 section).

## 1. Question

Study 6 found that the final Tülu-3-8B localizes the pushed sentence 93% of the time with intact
text (main test PASS), with random vectors working equally well. Do the earlier training stages of
the same model already do this?
- **Study 7:** `allenai/Llama-3.1-Tulu-3-8B-SFT` @ `f2a0b46b0cfda21003c6141b1ff837b7e165524d`
  (after supervised fine-tuning).
- **Study 8:** `allenai/Llama-3.1-Tulu-3-8B-DPO` @ `a7beb67e33ffd01cc87ac3b46cadc1000985b8db`
  (after preference tuning; the final model adds RL on top).

## 2. Protocol: identical to Study 6

Everything follows `gates/study6_plan.md` and `scripts/study6.py`, unchanged:
- the authors' prompt wording in Tülu's chat format (BOS prepended, trailing `<|end_of_text|>`
  removed);
- layer 2, α = 5, injection on the sentence words only;
- copy-back with the D3 fix + clean reader (arm A); random unit vectors of the same size (arm C, no
  reader);
- concept vectors recomputed **from that checkpoint** with the same recipe (Tülu chat format);
- 20 fresh draws × 10 positions = 200 trials per arm, from the authors' pool of 100 sentences;
- sanity checks S1–S8 as in Study 6, with the same stop rules (any failure, including S7 < 19/20
  clean copies, stops that study before trials);
- the same main-test rule: among arm-A exact-copy trials (≥ 40 needed), injected ≥ 50% and
  reader ≤ 15%; the conclusion labels are not used (they referred to the Study 6 question).

Fresh seeds: Study 7 draws `random.Random(20261024)`, random vectors seed 20261025; Study 8 draws
`random.Random(20261026)`, random vectors seed 20261027.

**Only differences from Study 6 (procedure, not protocol):**
1. The Llama size step is not repeated (it was a one-off comparison); each checkpoint's own
   activation size is still reported.
2. The concept vectors are recomputed as the first stage of each queued run, before its sanity
   checks and trials, instead of before the frozen-setup commit. Their SHA-256 hashes are written
   to the stimuli file and saved with the results; the recipe code is frozen in the setup commit.

## 3. Reported (no extra pass/fail)

Overall arm-A accuracy and reader accuracy, arm C accuracy, exact-copy count, per concept and per
position, comparison with final Tülu (186/200), and each checkpoint's activation size at layer 2.

## 4. Queue and conditions

One chained queue: wait for AC power → Study 7 (vectors → sanity → trials) → Study 8 (same). Study 8
starts automatically when Study 7 ends, whatever Study 7's outcome. Each study takes about 4.5–5
hours, so about 10 hours in total. Up to 3 automatic restarts per study, which resume where they
stopped. No monitoring during the runs; both are analysed together afterwards.
