# Where, not what: what the "sentence localization" introspection test measures in small LLMs

This repository is the complete record of a short, pre-registered research program, run from
2026-10-14 to 2026-10-20 on one laptop CPU. It tests the **sentence localization** task, a
proposed confound-free test of "introspective awareness" in language models:
1. inject a concept vector into one of ten sentences;
2. ask the model which sentence contained the injected thought;
3. compare with chance (10%).

There are twelve studies, on Meta's Llama-3.1-8B (base) and Llama-3.1-8B-Instruct, and AllenAI's
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

**Headline (full arc):** the **sensing** is present at every stage including the base model; the
**information is decodable** inside the models; but an **explicit yes/no report is absent** and is
**not unlocked** at 8B by an informative prompt or refusal-direction ablation.

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
logs/                        deviations (D0–D5), calibration log, public-release redactions
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

The analysis scripts rebuild `results/<study>/summary.md` and score the predictions with the
pre-committed rules:

```bash
python scripts/study4_analyze.py
```

The other studies use `gate1_analyze.py`, `study2_analyze.py`, `study3_analyze.py` and
`study5_analyze.py`, `study6_analyze.py`, `study7_8_analyze.py`, `study9_10_analyze.py`, `study11_analyze.py` and `study12_analyze.py` the same way.

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
- **Dry runs.** Every pipeline was first run end to end on a tiny random model, where every
  condition came out at chance.
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

\* In Study 1 the scoring rules were committed together with the predictions.

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
