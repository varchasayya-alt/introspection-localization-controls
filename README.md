# Where, not what: what the "sentence localization" introspection test measures in small LLMs

This repository is the complete record of a short, pre-registered research program, run from
2026-10-14 to 2026-10-25 on one laptop CPU. It tests the **sentence localization** task, a
proposed confound-free test of "introspective awareness" in language models:
1. inject a concept vector into one of ten sentences;
2. ask the model which sentence contained the injected thought;
3. compare with chance (10%).

There are sixteen studies, on Meta's Llama-3.2-1B/3B-Instruct, Llama-3.1-8B (base) and
Llama-3.1-8B-Instruct, and AllenAI's
Tülu-3-8B ladder (SFT, DPO, final).

**Start with the report: [`REPORT.md`](REPORT.md).**

## About this public release

- **Studies 6–10 were added after the first publication** (pre-registered and committed in the same
  way before their runs).
- **Full history.** The complete commit history is published, because it is the pre-registration
  evidence: in each study, the scoring rules and the researcher's predictions were committed before
  the run, and the mid-run Study 4 hypothesis has its own timestamp.
- **Timestamps are self-reported.** The predictions and rules were committed locally before each run
  and only pushed publicly later, so the commit timestamps are self-reported.
- **One history rewrite (D5).** Before publication, the history was rewritten once to replace this
  laptop's home-folder path with `%USERPROFILE%` / `$HOME` in every commit. Authors, dates, messages
  and order are unchanged; no other content changed. Current documents cite the new commit hashes.
  Older versions inside the history cite the pre-rewrite hashes, so see
  [`logs/commit_hash_map.md`](logs/commit_hash_map.md).
- **Redactions in the current files (D4).** Local paths were replaced by portable defaults; see
  [`logs/public_release_redactions.md`](logs/public_release_redactions.md). No study was re-run and
  no result changed.

## Status and claims

| Level | Meaning |
|---|---|
| 1 | Computational result (what the models do under these interventions) |
| 2 | Interpretation as a form of self-monitoring / introspection, stated with its limits |
| 3 | Phenomenal consciousness: **never asserted** |

Nothing here bears on whether any system is conscious.

| Study | Model | Question | Outcome | Report |
|---|---|---|---|---|
| 0 | — | Feasibility: CPU timing, checkpoint ladders, controls | bf16 storage with fp32 math; Tülu-3-8B chosen as the positive control | [`gate0_report.md`](gates/gate0_report.md) |
| 1 | Tülu-3-8B | Positive control + cheap-trick controls (our protocol) | Localizes 53.7% (Gate 1 PASS), but a clean reader of the copied text scores **74.7%** vs injected 45.3% (Gate 2 FAIL): **text corruption under our protocol** | [`gate1_report.md`](gates/gate1_report.md) |
| 2 | Tülu-3-8B | Is there a text-intact setting that still localizes? (our protocol) | Practice PASS, **fresh test FAIL** (18% vs reader 10%) | [`study2_report.md`](gates/study2_report.md) |
| 3 | Llama-3.1-8B-Instruct | Exact replication of the 88% headline + text control | **89.0%** (replicates); reader 42.0% (loophole not confirmed); exact-copy trials: **94% vs reader 5%** | [`study3_report.md`](gates/study3_report.md) |
| 4 | Llama-3.1-8B-Instruct | Label digit or words? Concept or random? | Words only **97.5%** (exact-copy 97% vs reader 10%: PASS); label only 64% (invisible to the copy test); random vectors **91%** | [`study4_report.md`](gates/study4_report.md) |
| 5 | Llama-3.1-8B-Instruct | Question-specific or spotlight? | "Longest?" pulled 41 points (vs 82.5 on the injection question); "like most?" 6 points; overall **mixed** | [`study5_report.md`](gates/study5_report.md) |
| 6 | Tülu-3-8B | Was Tülu's failure the model or our protocol? (authors' protocol, Study 4 arm A) | **93%**; exact-copy trials 93% vs reader 10% (**PASS**); random vectors **93%**: the earlier failure was **protocol-specific** | [`study6_report.md`](gates/study6_report.md) |
| 7–8 | Tülu-3-8B SFT, DPO | Same protocol on earlier training stages | SFT **93%**, DPO **92.5%**; both main tests **PASS**; random 93.5% / 91.5%: already present after SFT | [`study7_8_report.md`](gates/study7_8_report.md) |
| 9 | Llama-3.1-8B (base) | Is localization there before any tuning? (few-shot format) | Format check 50/50; **94%**; exact-copy trials 95% vs reader 9% (**PASS**); random **94%** | [`study9_10_report.md`](gates/study9_10_report.md) |
| 10 | Tülu-3-8B SFT, DPO, final | Yes/no detection ("Do you detect an injected thought?") | Hits − false alarms **≤ 2 points** at every stage; SFT's score 10.7 is mostly a yes-habit (16% false alarms); pre-registered stage split **not supported** | [`study9_10_report.md`](gates/study9_10_report.md) |
| 11 | Tülu-3-8B SFT, DPO, final; Llama-3.1-8B-Instruct | Is the yes/no report suppressed or absent? (logit lens, probes) | Tülu **ABSENT** per the 1-nat bar, but a real sub-threshold detection-specific yes-lean (0.4–0.75 nats) in late layers that shrinks at the output; Llama **SUPPRESSED** narrowly; injection **decodable** everywhere for both questions (AUC 0.93–1.00) | [`study11_report.md`](gates/study11_report.md) |
| 12 | Tülu-3-8B SFT, final; Llama-3.1-8B-Instruct | Can the yes/no report be unlocked? (informative prompt; refusal-direction ablation; both) | **Not unlocked** anywhere (bar: hits − false alarms ≥ 20 points, CI > 0). Ablation valid (refusals ~100% → 0%) but a **yes-bias**, not detection; informative prompt +8 points only in Llama, under the bar | [`study12_report.md`](gates/study12_report.md) |
| 13 | Llama-3.2-1B, 3B (8B reused) | Size curve (matched relative depth and push) | Decodable inside at every size (AUC ~0.99–1.00); intact-text localization **PASS** at 3B (96% vs reader 9%) and 8B (97% vs 10%); 1B officially **not testable** (copy check 15/20; deviation D6, not scored: 48.5% vs reader 15%, FAIL); yes/no report **flat** at 1–2 points | [`study13_report.md`](gates/study13_report.md) |
| 14 | Llama-3.2-3B | "Push the button": does adding the injected direction at the answer position make clean trials say "yes"? Specific to detection? | Steering layer 5 (probe AUC 1.000); **0% "yes" in every condition and dose** (detection, control questions, random directions, and the yes − no positive control); pushes too weak to move any answer → Q1 "nothing", **inconclusive** | [`study14_report.md`](gates/study14_report.md) |
| 15 | Llama-3.2-3B | Same question with a later layer (18) and pushes sized so a "yes"-word push works | Positive control **96% "yes"** (valid). Injected direction: detection **0% "yes"**, unrelated questions **11%** (mostly one question); logit +1.46 (detection) vs +1.89 (control) vs +0.20 (random): a **general yes-lean, not a detection signal**. Q1 nothing, Q2 generic "yes button" (CI crosses the bar), Q3 random about as strong | [`study15_report.md`](gates/study15_report.md) |
| 16 | Llama-3.1-8B-Instruct | Study 15 replicated at 8B (block 21; distinct doses; logit shift pre-registered) | Positive control **94.5%** (valid); injected direction **0% "yes"** on detection at every dose, 1.5% on unrelated questions; logit +2.95 (detection) vs **+4.09** (unrelated) vs −0.08 (random): **general yes-lean, larger on unrelated questions; not detection**. Q1 nothing, Q2 neither moves, Q3 random about as strong | [`study16_report.md`](gates/study16_report.md) |

**Headline (full arc):** the **sensing** is present at every stage including the base model; the
**information is decodable** inside the models; but an **explicit yes/no report is absent** and is
**not unlocked** at 8B by an informative prompt or refusal-direction ablation. Across Llama sizes
(1B/3B/8B, Study 13) this forms three levels: the **information is present inside** at every size (probe AUC ~0.99–1.00); it becomes **usable for localization** between 1B and 3B; an **explicit yes/no report** is absent through 8B. A first causal test (Study 14, 3B), adding the injection's answer-position footprint to clean trials, was **inconclusive**: the pushes were too weak to move any answer, including a positive control. With pushes sized so the positive control works (Study 15), the footprint acts as a **general yes-lean** (as strong on unrelated questions), never a detection "yes". Llama-3.1-8B-Instruct shows the same (Study 16): a strong, non-random yes-lean, larger on unrelated questions than on the detection question, and no detection "yes".

In detail: intact-text, content-agnostic localization (~93–95%) is present at **every** stage of
the Tülu-3 ladder (base Llama-3.1-8B in a few-shot format, SFT, DPO, final), while yes/no detection
is essentially **absent** at every stage tested (hits − false alarms ≤ 2 points). So localization
measures a sensing signal already present after pretraining, which these 8B models do not turn into
an explicit self-report. The pre-registered prediction that reporting emerges at DPO was **not**
supported; the sensing/reporting dissociation across tasks is a descriptive finding. Macar et al.
(2026; main analyses on Gemma3-27B) found that DPO enabled yes/no detection in their training-stage
comparison of OLMo-3.1-32B checkpoints (base, SFT, DPO, instruct); scale is a possible explanation,
untested here. Inside the models the injection is linearly decodable everywhere; in Tülu only a
small, sub-threshold "yes" lean forms in late layers (Study 11). Neither an informative prompt nor
refusal-direction ablation unlocks the report at 8B (Study 12); in Macar et al.'s Gemma3-27B
experiments, refusal ablation raised detection from 10.8% to 63.8% (false positives from 0% to 7.3%). Scale and model family remain
untested explanations.

Under the authors' protocol, sentence localization detects **where** processing was disturbed, not
**what**:
- it works with random vectors;
- it works with the text left intact;
- it partly leaks into unrelated questions (tested in Llama);
- injection-themed questions draw it out most strongly (tested in Llama).

Under our original protocol (different prompt, injection span and strength), Tülu-3-8B's
localization was explained by text corruption (Studies 1–2); under the authors' protocol it
localizes 93% with the text intact (Study 6), so that failure was protocol-specific, not a model
property. The report includes a
**recommended-controls checklist** for anyone using the test.

## Repository layout

```
README.md                    this file
REPORT.md                    the write-up (start here)
LICENSE                      MIT (code); see "License" for documentation (CC BY 4.0)
THIRD_PARTY_NOTICES.md       material reused from the "Detecting the Disturbance" code (MIT)
predictions.md               researcher's predictions (verbatim) + scoring rules, per study
gates/                       design proposals/plans (frozen before runs) and reports, per study
logs/                        deviations (D0–D7), calibration log, public-release redactions
stimuli/                     frozen sentence sets, concepts, seeds, model pins, vector hashes
scripts/                     experiment code (one script per study), analysis scripts, runners
results/                     per-trial JSONL results, sanity checks, status and summaries per study
```

## How to reproduce

### Environment

The runs used Windows 11, an Intel Core Ultra 7 255U (AVX2, CPU only), 32 GB RAM and Python 3.12.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install transformers==5.19.0 accelerate==1.15.0 huggingface_hub==2.2.0 numpy==2.5.4 safetensors==0.8.0 tokenizers==0.23.3
```

On Linux or macOS, use `.venv/bin/python`.

**Environment variables:**
- `INTROSPECTION_DATA`: per-trial outputs and logs (default `./data`, git-ignored);
- `HF_HOME`: the Hugging Face cache;
- `PYTHON`: the interpreter for the `run_*_all.sh` runners.

All models run with weights stored in bf16 and math in fp32 (about 17.5 GB RAM for an 8B model).
An 8B run takes 5–10 hours on this CPU. Run on AC power: on battery the CPU was about 2.75× slower.

### Model weights (not committed)

- **Studies 1–2, 6 and 10:** `allenai/Llama-3.1-Tulu-3-8B` at revision `666943798adbde0b1aff34626007e26986a3c107`.
- **Studies 3–5:** `meta-llama/Llama-3.1-8B-Instruct` at revision `0e9e39f249a16976918f6564b8830bc894c89659`.
- **Studies 7, 8 and 10:** `allenai/Llama-3.1-Tulu-3-8B-SFT` @ `f2a0b46b0cfda21003c6141b1ff837b7e165524d` and `-DPO` @ `a7beb67e33ffd01cc87ac3b46cadc1000985b8db`.
- **Study 9:** `meta-llama/Llama-3.1-8B` (base, gated like the Instruct model) @ `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`.
  - This model is **gated**. Accept Meta's licence on its Hugging Face page, create a Read token,
    and log in with `hf auth login`.
- **Pins** are in [`stimuli/model_pins.json`](stimuli/model_pins.json). Weights download
  automatically on first use.
- **Studies 3–5 (and Study 6's vector comparison) need the authors' saved concept vectors.** Download and verify them with:

```bash
python scripts/download_their_vectors.py
```

### Re-running and re-analysing

Each study runs end to end and resumes after a crash: finished trials are skipped.

```bash
python scripts/gate1.py --stage all
```
```bash
python scripts/study2.py --stage all
```
```bash
python scripts/study3.py --stage all
```
```bash
python scripts/study4.py --stage all
```
```bash
python scripts/study5.py --stage all
```
```bash
bash scripts/run_study6_all.sh
```
```bash
bash scripts/run_study7_8_queue.sh
```
```bash
bash scripts/run_study9_10_queue.sh
```
```bash
bash scripts/run_study11_queue.sh
```
```bash
bash scripts/run_study12_queue.sh
```

Study 12 also downloads AdvBench (`walledai/AdvBench`, gated: accept its terms on Hugging Face) and
Alpaca (`tatsu-lab/alpaca`) at pinned revisions, and needs `pyarrow`. The refusal ablation is applied
only at runtime (forward hooks); no ablated weights are saved and nothing is generated.

```bash
bash scripts/run_study13_queue.sh
```
```bash
bash scripts/run_study13_d6_after.sh
```

Study 13 needs the gated `meta-llama/Llama-3.2-1B-Instruct` and `-3B-Instruct` (pins in
`stimuli/model_pins.json`); the second command is deviation D6 (1B localization).

```bash
python scripts/study14.py --stage calibrate
```
```bash
bash scripts/run_study14_test.sh
```

Study 14 (Llama-3.2-3B) reuses Study 13's 3B concept vectors (run Study 13 first); calibration saves
the steering direction outside the repository and checks it by hash before the test.

```bash
python scripts/study15.py --stage direction
```
```bash
python scripts/study15.py --stage calibrate
```
```bash
bash scripts/run_study15_test.sh
```

Study 15 builds its direction from Study 14's saved calibration states (run Study 14's calibration
first).

```bash
bash scripts/run_study16_all.sh
```

Study 16 (Llama-3.1-8B-Instruct, about 8 hours) needs the authors' vectors (`download_their_vectors.py`)
and about 18 GB of free RAM; the runner takes stages as arguments (`dircal`, `dosecal`, `test`).

The analysis scripts rebuild `results/<study>/summary.md` and score the predictions with the
pre-committed rules:

```bash
python scripts/study4_analyze.py
```

The other studies use `gate1_analyze.py`, `study2_analyze.py`, `study3_analyze.py` and
`study5_analyze.py`, `study6_analyze.py`, `study7_8_analyze.py`, `study9_10_analyze.py`, `study11_analyze.py`, `study12_analyze.py`, `study13_analyze.py`, `study13_d6_analyze.py`, `study14_analyze.py`, `study15_analyze.py` and `study16_analyze.py` the same way.

### Integrity conventions

- **Pre-run commits.** Every study has commits, in this order:
  1. the design proposal;
  2. the prediction scoring rules;
  3. the researcher's predictions, verbatim;
  4. the frozen materials, code and analysis script, before any model saw the material.

  In Study 1 the scoring rules went into the same commit as the predictions.
- **Sanity checks before data.** Each run checks:
  - that the prompt tokens match the chat template;
  - that "1"…"10" are single tokens;
  - that α = 0 changes nothing;
  - that the KV-cache speed-up matches a full pass;
  - vector hashes, and study-specific checks.
- **Dry runs.** Every pipeline up to Study 13 was first run end to end on a tiny random model,
  where every condition came out at chance. Study 14 instead had a 5-trial timing dry run on the real
  model (placeholder direction, outputs discarded) and its analysis script was tested on synthetic
  data before any real data.
- **Deviations** are logged before the runs they affect ([`logs/deviations.md`](logs/deviations.md)).
  Calibration attempts are capped and logged ([`logs/calibration_log.md`](logs/calibration_log.md)).

## Commit map

| Study | Proposal | Scoring rules | Predictions | Frozen setup | Results |
|---|---|---|---|---|---|
| 0 | `83447df` | — | — | — | `83447df` |
| 1 | `83447df` | `e268a7e`\* | `e268a7e` | `aaa7ba8`, `73e85a1`, `2036c8a` | `8aa3e9b`, `d27559a`, `48f22e3` |
| 2 | `da4892a` | `229a20d` | `c5197b5` | `a31372a` | `34385fa`, `fc47e50` |
| 3 | `97b5003` | `b1b0336` | `a07886f` | `4e17e38` | `423b45f` |
| 4 | `d84560b` | `ac653cb` | `15abe21` | `da36436` | `0c68fde` |
| 5 | `2bef5d6` | `ab0e61c` | `791e0e5` | `2134b4e` | `1930855` |
| 6 | `a8cf4f0` | `a8cf4f0` | `9c602d9` | `4502395` | `de3930c` |
| 7–8 | `6bae069` | `6bae069` | `b70f297` | `9331c85` | `fa76b69` |
| 9–10 | `e5841f5` | `e5841f5` | `89e3f4f` | `f13d15b` | `9ca0f2c` |
| 11 | `c7eb51d` | `c7eb51d` | `cdecf60` | `7b68386` | `b61bea3` |
| 12 | `2e49802` | `2e49802` | `8e5efb9` | `41bf9ba` | `d7fcf86` |
| 13 | `d99992c` | `d99992c` | `e5ef057` | `8a6f67a` (D6: `ff5d91d`, `bea42b6`) | `050bf7a` |
| 14 | `1800c98` | `1800c98`\*\* | `1800c98`\*\* | `001e20a` (D7), calibration + direction hash `12e430b` | `bc4b1f0` |
| 15 | `9a7fa84` | `9a7fa84`\*\* | `9a7fa84`\*\* | `eb0dc7f`, doses `fff0b87` | `2e4a904` |
| 16 | `a7ce49e` | `a7ce49e`\*\* | `a7ce49e`\*\* | `d51f17f`, dry run `61e6487`, direction `389d2e4`, doses `97f2f2c` | `aeec33a` |

\* In Study 1 the scoring rules were committed together with the predictions.
\*\* In Studies 14–16 the design, bars and predictions were given together by the researcher and committed together, before any data.

Also:
- `824e556`: the Study 4 hypothesis, logged during the Study 3 run (D2).
- Study 1's `73e85a1` (a memory fix) and `2036c8a` (the analysis script) came before any Study 1
  result.

## Contributions

- **Principal investigator** (Varchas Yogesh Hebbale): set the research direction and constraints,
  approved or amended every design, gave every prediction, and made every go/no-go decision.
- **Claude Opus 5.5** (Anthropic): did the literature review, wrote the code, ran the experiments
  and analyses, and drafted the plans and reports.

## License

| Content | License |
|---|---|
| **Code**: Python and shell scripts | MIT; see [`LICENSE`](LICENSE) |
| **Report and documentation**: `REPORT.md`, READMEs, plans, reports, predictions, logs, and the project's own results files and materials | [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) |
| **Third-party material**: the sentence pool, concept datasets and prompt wording from the *Detecting the Disturbance* code | MIT, © 2026 Ely Hahami; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) |
| **Model weights** (not included) | Llama 3.1 Community License (Meta) |

## Related prior work

- varchasayya-alt, *llm-consciousness-indicators-negative-results*
  (github.com/varchasayya-alt/llm-consciousness-indicators-negative-results).

## Related work found after publication

These came to our attention after Studies 1–13 were published; none of them shaped those studies.

- **Ferrara (2026)**, *Open-Weight Masked Introspection: Measuring What Language Models Can Report
  About Their Own Computation*, arXiv 2608.20569. Eight open-weight models, including
  Llama-3.1-8B-Instruct. Linear probes recover the interventions while the models' verbal reports
  stay at chance. A text-only observer (an unintervened copy of the same model that sees only the
  visible output) is used as a baseline, similar to our clean reader. It does not test
  forced-choice localization, and it does not causally test the path from internal state to report.
- **Macar et al. (2026)**, *Mechanisms of Introspective Awareness*, arXiv 2603.21396: in Gemma3-27B,
  patching "gate" feature activations from steered examples onto unsteered prompts gives only
  partial detection (max 25.1%).
- **Hasik (17 Oct 2026)**, *Control questions matter: introspection, refusal ablation, and gates in
  open Gemma models*, LessWrong: the detection gains from refusal-direction ablation coincide with
  model degradation, so control questions are needed to interpret them.
- **Macar et al. (2026), code** (github.com/safety-research/introspection-mechanisms): real
  injected-thought detection in Gemma-3-27B-it (injection at block 37, strength 4, judged by an LLM),
  with "gate" features in blocks 45–61; patching gate features at the last token gives partial
  detection (max 25.1%).
- **Hasik (2026)**, also: in Gemma 3 (27B), a detection-specific signal relative to a control question
  ("Do you believe that 1+1=3?") for most concepts, measured as a yes − no logit shift; detection
  rates depend heavily on the prompt (28.5% with the main prompt, 79.5% with a short one, 1% with no
  introduction); gate layer 45 is causally important.

**What overlaps.** Our "probe-decodable but not reported" result (Studies 11–13) overlaps with
Ferrara (2026). Our localization, text-corruption (copy-back + clean reader), random-vector and
training-stage results do not.

**What has not been done (as of October 2026, to our knowledge).** No one has compared the
injection's answer-position fingerprint on the detection question vs unrelated questions, at a dose
matched to a yes-direction positive control, across models that do and don't report injections.
Studies 15–16 did this for two non-reporting models (Llama 3B and 8B); Study 17 (pre-registration in
preparation) extends it to Gemma-3-27B-it, which Macar et al. report detects injections (our
Study 17 re-tests this first), and Gemma-3-4B-it.

