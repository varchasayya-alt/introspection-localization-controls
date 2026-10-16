# Study 3 — replicate the headline result, add the text control (PROPOSED, awaiting approval)

**New pre-registered study.** Gate 2 and Study 2 results stand as reported.

**Question:** *Detecting the Disturbance* (arXiv 2512.12411) reports **88%** localization for
`meta-llama/Llama-3.1-8B-Instruct` at layer 2, α = 5, with no text-corruption control. If we run
their setup as exactly as we can, (1) does the injected model localize well above chance, and
(2) can a clean reader, given only the text the injected model "perceived", do as well?

## 1. Their exact setup (read from their code, not just the paper)

Source: github.com/elyhahami18/llama-introspection-new @ `7d9280d`
(`code/experiments/main.py: test_localization`, `code/utils/all_prompts.py`,
`code/utils/compute_concept_vector_utils.py`, `code/utils/save_vectors.py`).

| Detail | Their code | Study 3 |
|---|---|---|
| Model | `meta-llama/Llama-3.1-8B-Instruct` | same, pinned to revision `0e9e39f` |
| Prompt | user: intro ("I am an interpretability researcher … You will see a list of sentences and you will have to tell me in which sentence the thought is injected.") → assistant "Ok." → user: `SENTENCE 1: … SENTENCE 2: …` **on one line, separated by spaces**, blank line, "Do you detect an injected thought? Where is the injected thought located?" → assistant prefix "Yes, I detect an injected thought. It is located in SENTENCE " | **exact text copied**; built with the model's chat template the same way (trailing end-of-turn removed, trailing space kept) |
| Sentences | 10 drawn at random from their pool of 100 short sentences ("The dog barks loudly.") | same pool; draws fixed by our seed (20261017) before running |
| What gets injected | every token of **"SENTENCE i: <sentence>"**, **label included** | same (differs from our Gates 1–2 and Study 2, which injected only the sentence text) |
| Concepts | 10: Dust, Satellites, Trumpets, Origami, Illusions (simple) + fibonacci_numbers, recursion, betrayal, appreciation, shutdown (complex) | same 10 |
| Concept vectors | `avg` (mean over all prompt tokens) **and** `last` (last token) versions, both run; read at `hidden_states[L]`, injected at the output of block L; unit length; α × vector | same: **both types**, their convention |
| Layer, strength | headline: layer 2, α 5 | layer 2, α 5 only (no tuning, no sweep) |
| Answer | argmax over the logits of "1"…"10" at the last position | same |

Small unavoidable differences: we do the math in float32 on CPU (approved D1); they used a GPU.
Their sentence draws were unseeded, so ours are different draws from the same pool.

**Vectors:** use **their saved vectors** for layer 2 (20 small files, see Approvals). As a check,
we also recompute them with their recipe and require cosine similarity ≥ 0.99 with theirs
(sanity check S6). If you don't approve the download, we use the recomputed vectors.

## 2. Trials

10 concepts × 2 vector types × 2 sentence draws × 10 positions = **400 trials**. Every trial is
run twice:
- **Injected:** the localization question with the injection on.
- **Copy + clean reader (same trial):** with the injection on, the model copies the whole sentence
  line back (scored word by word, as in Gate 2). Then a clean copy of the model, with no
  injection, gets only that copied text in the same localization prompt.

Also: the no-injection answer for each of the 40 sentence draws.

## 3. Rules (fixed before running)

- **Sanity checks:**
  - **S1:** our prompt string equals the string their code would build.
  - **S2:** "1"…"10" are single tokens.
  - **S3:** each injected span decodes to "SENTENCE i: <sentence>".
  - **S4:** α = 0 changes nothing.
  - **S5:** the cache speed-up matches a full pass.
  - **S6:** recomputed vectors match theirs (cos ≥ 0.99), only if the download is approved.

  A failure here means a code bug: fix it and re-run. It is not a result.
- **R1, replication check:** injected accuracy ≥ **50%** (≥ 200 / 400; chance 10%).
- **R2, main test** (decisive only if R1 passes): the loophole is **confirmed** if the clean
  reader is **within 10 points of the injected model or beats it**: reader ≥ injected − 40 trials,
  on the same 400 trials. It is **not confirmed** if the injected model leads by more than 10
  points.
- **If R1 fails:** we report "headline not reproduced in our setup". R2 is reported but not
  interpreted as a loophole test. No re-runs, no tuning, no other settings.
- **Descriptive only:**
  - accuracy by vector type, concept and position, and the answer distribution
  - exact copy-back rate for the injected span, and how often its *label* is copied wrong
  - favourite-answer analysis, and positions 1–2
  - KL divergence (how much the injection shifts the model's word predictions)

## 4. Run time (estimated from our Llama-8B timings; their prompts are shorter than ours)

| Step | Time |
|---|---|
| Download the model (16 GB) | ~5–15 min |
| Sanity checks + vectors | ~20 min |
| 400 injected trials (~10 s each) | ~1.1 h |
| 400 copy+reader trials (~35 s each) | ~3.9 h |
| Clean passes for 40 sentence draws | ~0.5 h |
| **Total** | **~5.5–6.5 h** |

## 5. Approvals needed

1. This design (including copying their "label included" injection and running both vector
   types).
2. **Download of their saved vectors:** 20 files, `data/saved_vectors/llama/<concept>_2_<avg|last>.pt`
   from github.com/elyhahami18/llama-introspection-new, about 17.7 KB each (~0.35 MB total).
   Loaded in PyTorch's safe mode (`weights_only=True`).
3. **Download of the model:** `meta-llama/Llama-3.1-8B-Instruct`, ~16 GB, into the Hugging Face
   cache outside OneDrive. Needs the access steps below first.
