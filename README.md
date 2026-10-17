# Where, not what: what the "sentence localization" introspection test measures in small LLMs

This repository is the complete record of a short, pre-registered research program, run from
2026-10-14 to 2026-10-17 on one laptop CPU. It tests the **sentence localization** task, a
proposed confound-free test of "introspective awareness" in language models:
1. inject a concept vector into one of ten sentences;
2. ask the model which sentence contained the injected thought;
3. compare with chance (10%).

There are five studies, on AllenAI's Tülu-3-8B and Meta's Llama-3.1-8B-Instruct.

**Start with the report: [`REPORT.md`](REPORT.md).**

## About this public release

- **All studies are closed.** No further experiments are planned in this repository.
- **Single-commit snapshot.** This is a snapshot of a private development repository at its final
  state (development commit `DEVCOMMIT`; the only later change is this paragraph). The development
  history isn't published. Commit hashes cited in the reports, `predictions.md` and the commit map
  below refer to that history.
- **Redactions.** Local paths were replaced by portable defaults and placeholders before release
  (deviation D4). See [`logs/public_release_redactions.md`](logs/public_release_redactions.md).
  No study was re-run and no result changed.

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
| 1 | Tülu-3-8B | Positive control + cheap-trick controls | Localizes 53.7% (Gate 1 PASS), but a clean reader of the copied text scores **74.7%** vs injected 45.3% (Gate 2 FAIL): **text corruption** | [`gate1_report.md`](gates/gate1_report.md) |
| 2 | Tülu-3-8B | Is there a text-intact setting that still localizes? | Practice PASS, **fresh test FAIL** (18% vs reader 10%) | [`study2_report.md`](gates/study2_report.md) |
| 3 | Llama-3.1-8B-Instruct | Exact replication of the 88% headline + text control | **89.0%** (replicates); reader 42.0% (loophole not confirmed); exact-copy trials: **94% vs reader 5%** | [`study3_report.md`](gates/study3_report.md) |
| 4 | Llama-3.1-8B-Instruct | Label digit or words? Concept or random? | Words only **97.5%** (exact-copy 97% vs reader 10%: PASS); label only 64% (invisible to the copy test); random vectors **91%** | [`study4_report.md`](gates/study4_report.md) |
| 5 | Llama-3.1-8B-Instruct | Question-specific or spotlight? | "Longest?" pulled 41 points (vs 82.5 on the injection question); "like most?" 6 points; overall **mixed** | [`study5_report.md`](gates/study5_report.md) |

**One-sentence result:** in Llama-3.1-8B-Instruct, sentence localization detects **where**
processing was disturbed, not **what**:
- it works with random vectors;
- it works with the text left intact;
- it partly leaks into unrelated questions;
- injection-themed questions draw it out most strongly.

In Tülu-3-8B the same test is fully explained by text corruption. The report includes a
**recommended-controls checklist** for anyone using the test.

## Repository layout

```
README.md                    this file
REPORT.md                    the write-up (start here)
LICENSE                      MIT (code); see "License" for documentation (CC BY 4.0)
THIRD_PARTY_NOTICES.md       material reused from the "Detecting the Disturbance" code (MIT)
predictions.md               researcher's predictions (verbatim) + scoring rules, per study
gates/                       design proposals/plans (frozen before runs) and reports, per study
logs/                        deviations (D0–D4), calibration log, public-release redactions
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

- **Studies 1–2:** `allenai/Llama-3.1-Tulu-3-8B` at revision `666943798adbde0b1aff34626007e26986a3c107`.
- **Studies 3–5:** `meta-llama/Llama-3.1-8B-Instruct` at revision `0e9e39f249a16976918f6564b8830bc894c89659`.
  - This model is **gated**. Accept Meta's licence on its Hugging Face page, create a Read token,
    and log in with `hf auth login`.
- **Pins** are in [`stimuli/model_pins.json`](stimuli/model_pins.json). Weights download
  automatically on first use.
- **Studies 3–5 also need the authors' saved concept vectors.** Download and verify them with:

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

The analysis scripts rebuild `results/<study>/summary.md` and score the predictions with the
pre-committed rules:

```bash
python scripts/study4_analyze.py
```

The other studies use `gate1_analyze.py`, `study2_analyze.py`, `study3_analyze.py` and
`study5_analyze.py` the same way.

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

## Commit map (development history)

| Study | Proposal | Scoring rules | Predictions | Frozen setup | Results |
|---|---|---|---|---|---|
| 0 | `0e71871` | — | — | — | `0e71871` |
| 1 | `0e71871` | `acc7335`\* | `acc7335` | `0d82a56`, `8ef360d`, `89965b9` | `6b92130`, `89f30e3`, `ca120a2` |
| 2 | `a97bf82` | `f6eb6c3` | `3bdd3f2` | `22e2cbd` | `e5caf91`, `faa4e18` |
| 3 | `674f137` | `c23d17a` | `d925dc9` | `db5edbf` | `385dfba` |
| 4 | `db69988` | `ef2431c` | `c7a3dc6` | `8d30992` | `8d7d9ee` |
| 5 | `1135cc3` | `960ec22` | `b948ceb` | `5a5b121` | `ca4cb22` |

\* In Study 1 the scoring rules were committed together with the predictions.

Also:
- `de698e5`: the Study 4 hypothesis, logged during the Study 3 run (D2).
- Study 1's `8ef360d` (a memory fix) and `89965b9` (the analysis script) came before any Study 1
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
