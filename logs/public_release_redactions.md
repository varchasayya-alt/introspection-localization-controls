# Public-release redactions

Made on 2026-10-17, after all five studies were finished (deviation **D4**). No study was re-run;
no computation, rule, material or result changed.

## What changed

| File(s) | Before | After |
|---|---|---|
| `scripts/introspect_lib.py` | `HF_HOME` forced to a local cache path; `DATA_ROOT` defaulted to a local folder | `HF_HOME` is no longer set (normal Hugging Face behaviour); `DATA_ROOT` = `$INTROSPECTION_DATA`, default `./data` (git-ignored) |
| `scripts/run_gate1_all.sh`, `run_study2_all.sh` … `run_study5_all.sh` | absolute path to the local Python and log folder | `PY="${PYTHON:-python}"`, `LOGDIR="${INTROSPECTION_DATA:-$REPO/data}/logs"` |
| `scripts/gate1.py`, `study2.py`, `study3.py` (docstrings only) | local output path | `$INTROSPECTION_DATA/<study>` |
| `stimuli/study4_stimuli.json`, `stimuli/study5_stimuli.json` | `vector_dir` = absolute local path | `vector_dir` = `<INTROSPECTION_DATA>/study3/their_vectors`, resolved by `study4.py`/`study5.py`. The `vector_sha256` hashes (the actual integrity check) are unchanged |
| `gates/gate1_plan.md`, `logs/deviations.md` (D0) | local paths in text | `<home>\…` placeholders |
| `requirements.txt` | header comment with a local path | neutral header; package versions unchanged |
| `README.md` | development README | rewritten for the public release |

## Added for the release

- `REPORT.md`, `LICENSE` (MIT), `THIRD_PARTY_NOTICES.md`;
- `scripts/download_their_vectors.py`, which fetches the authors' vectors and checks their SHA-256;
- this file.

## How to verify

The pre-registration order (proposal → scoring rules → predictions → frozen setup → run →
results) is documented in the commit map in `README.md`, using development-history hashes. The
frozen materials' integrity can be checked:
- the vector files by SHA-256 (`scripts/download_their_vectors.py`);
- the sentence draws by re-running the seeded generation described in each stimuli file's `notes`.
