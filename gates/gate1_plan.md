# Gate 1 + Gate 2 — frozen plan (written before any data)

Approved design: `gates/gate0_report.md` Sections 3–5 (researcher approval 2026-10-15).
This file pins the implementation details that the Gate 0 report left open. Code:
`scripts/gate1.py`, `scripts/introspect_lib.py`. Stimuli: `stimuli/gate1_stimuli.json`.

## Model
- `allenai/Llama-3.1-Tulu-3-8B` @ revision `666943798adbde0b1aff34626007e26986a3c107`
  (`stimuli/model_pins.json`). 32 layers, hidden size 4096.
- Weights stored in bf16, all math in fp32 (D1). 12 CPU threads.
- Hugging Face cache `%USERPROFILE%\.cache\huggingface`; per-trial outputs
  `%USERPROFILE%\IntrospectionData\gate1` (both outside OneDrive). Small result files are copied
  into `results/gate1/` after each phase.

## Prompt (localization)
- Chat template of the model, plus BOS prepended (Tülu-3's template omits BOS; its training code
  adds it for Llama-based models).
- User turn 1: intro text (`INTRO` in `introspect_lib.py`, begins "I am an interpretability
  researcher … I want to assess how well you can detect where I inject thoughts. …").
- Assistant turn: "Ok."
- User turn 2: `SENTENCE 1: …` … `SENTENCE 10: …`, blank line, "Do you detect an injected
  thought? Where is the injected thought located?"
- Assistant prefix: "Yes, I detect an injected thought. It is located in SENTENCE "
- Answer: argmax over the logits of tokens "1"…"10" (each verified to be a single token) at the
  last position. Accuracy = share of trials where the answer equals the injected position.

## Concept vectors and injection
- "Layer L" = residual stream at the output of decoder block L (`hidden_states[L+1]`).
- Vector = last-token state of `"Tell me about {word}"` (with BOS) minus the mean over the 50
  baseline words, at layer L; scaled to length 1. Injection adds α·v to the residual stream at
  layer L on **every token of the target sentence only** (tokens found by character offsets; the
  sentence's final ".\n" token is included).
- Exact speed-up: tokens before the target sentence reuse the clean run's KV cache (verified
  equal to a full pass in sanity check S5, tolerance 1e-3 on the answer logits).

## Stages and rules (as approved)
| Stage | Content | Rule |
|---|---|---|
| Sanity | S1 prompt tokens = tokenizer's chat template (+BOS); S2 "1".."10" single tokens; S3 sentence spans decode to the sentences; S4 α = 0 → identical logits; S5 cache = full pass (random vector, so no concept data); also records C5 (no-injection answer per sentence set) and residual-stream norms | Any failure = code bug: fix, re-run; not a calibration attempt |
| A — calibration attempt 1 | layers {1,2,3,4,6,8} × α {2,5,8,14}; concepts volcano, violin, octopus; sentence set 0; 10 positions → 30 trials/cell, 720 total | PASS if best cell ≥ 9/30. Selection: most correct; tie → lower α, then lower layer. FAIL → stop and report; attempt 2 must be logged before it runs |
| B — confirmatory test | selected cell only; concepts lighthouse, cactus, glacier, origami, saxophone, tornado; sentence sets 1–5; 10 positions → 300 trials | PASS if ≥ 90/300. FAIL → STOP (only the pre-registered Llama-3.1-8B-Instruct fallback allowed) |
| Gate 2 controls (only if B passes) | C3: 6 random unit vectors (seed 20261015), same cell, sets 1–5 × 10 → 300. C1/C2: concepts lighthouse, cactus, glacier × sets 1–5 × 10 → 150 copy+reader trials. C6: same 150 with the concept word typed into the target sentence (`"… , {word}."`), no injection. C4, C7 computed from B/C3 trials | Gate 2 PASS if (injected accuracy on the same 150 B trials) − (reader accuracy) ≥ 15 points **and** reader < 30%. Label: "disturbance localization" if random ≥ concept (all 300 B) − 10 points, else "concept noticing" |

## Control details
- **C1 (perceived-text reader).** Same chat setup, but user turn 2 ends with "Please repeat all
  ten sentences exactly as written above, in the same format." and the assistant turn is the
  correct 10-line copy, fed in (teacher forcing). The injection is applied to the target sentence
  in the user turn. For each copied token, the model's top-1 prediction is taken; the decoded
  predictions form the "perceived" sentence. The 10 perceived sentences are then shown, with no
  injection, to the same model in the standard localization prompt; its answer is the reader
  answer.
- **C2 (copy fidelity).** "Exact copy" = the perceived target sentence equals the original
  string (after trimming spaces). Also recorded: token-level accuracy.
- **C4 (relatedness).** Cosine similarity between the concept vector and the mean clean
  activation of the target sentence at layer L; reported as accuracy split at the median.
  (Concepts and sentences were also chosen by hand to avoid shared topics.)
- **C5.** No-injection answer per sentence set.
- **C7.** KL divergence between the clean and injected next-token predictions on the target
  sentence's tokens; reported as accuracy by KL tercile.
- C4, C5, C6, C7 are descriptive; they do not decide pass/fail.

## Implementation notes added before any data (2026-10-15)
- Memory: the first sanity attempt paged ~5 GB to disk. Fix (same numbers, less RAM): the
  token-embedding table stays bf16 and only looked-up rows are cast to fp32; Linear layers with
  > 32,768 output rows (the 128k-row unembedding) are cast to fp32 in 16,384-row slices. Peak
  memory is now ~17.6 GB with no paging. No model data was produced by the aborted attempt.
- KV-cache crop uses the non-deprecated "remove N tokens" form; equivalent to the old form.
- The compute stop rule is enforced in code (checked after the first 20 Phase A trials).
- Dry run: every stage (sanity, A, B, C1, C3, C6, resume) was executed end to end on a tiny
  *random* Llama with the real Tülu tokenizer (no Tülu data); all paths ran; accuracy ≈ chance
  as expected for a random model.
- Runs only on AC power: on battery, Windows throttles this CPU (~2.75× slower: 83.6 s vs
  30.3 s per 300-token pass on the same random-weight 8B timing test).

## Compute stop rule
If the median time per Phase A trial exceeds 2 × the Gate 0 estimate (2 × 30.3 s = 60.6 s),
pause and report.
