# RUNBOOK — running Study 17's Gemma-3-27B arm on a rented GPU

For a first-time user. Budget: **$40 maximum**; expected cost **about $4–10**. Everything runs with one
command. Read the whole page once before renting anything.

## 0. Before you rent (on your laptop)

1. **The 4B arm must be finished and pushed first.** The GPU run also judges the 4B model's Gate 1
   answers, which it reads from this repository. Check that
   `results/study17/4b/gate_responses.jsonl` exists on GitHub.
2. **Gemma access.** On huggingface.co, while logged in, open the `google/gemma-3-27b-it` page and
   make sure it says you have access. (Your account already has it.)
3. **Make a Hugging Face token** for this run only: huggingface.co → Settings → Access Tokens →
   *Create new token* → type **Read** → name it `runpod-study17` → Create. Copy it somewhere private
   (e.g. a password manager). **Never paste it into a file, a chat, or the repository.**

## 1. Rent the GPU (RunPod)

1. Sign up at runpod.io. Under **Billing**, add **$20** of credit. That's enough. Don't turn on
   auto-pay, so you can never spend more than you put in.
2. **Pods → Deploy.** Choose:
   - **GPU:** *A100 80GB* (SXM or PCIe). If none are free, *H100 80GB*. It must have **80 GB**.
   - **Type:** *On-Demand* (not Spot: Spot machines can be taken away mid-run).
   - **Template:** *RunPod PyTorch* (any 2.x version with CUDA 12).
   - **Container disk:** 50 GB. **Volume disk:** 150 GB (the model alone is about 55 GB).
3. Click **Deploy**. The hourly price is shown, about $1.2–2/h for an A100 or $2.5–3.5/h for an
   H100. **The clock starts now**, so continue straight away.
4. When the pod is *Running*: **Connect → Start Jupyter Lab** (or *Web Terminal*). In Jupyter Lab,
   open **File → New → Terminal**.

## 2. Set up (copy each line into the terminal, press Enter, wait for it to finish)

```bash
cd /workspace && git clone https://github.com/varchasayya-alt/introspection-localization-controls.git
```
```bash
cd /workspace/introspection-localization-controls
```
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
The last command must print `True` and a GPU name with "A100" or "H100". Then:
```bash
pip install "transformers==5.19.0" "accelerate==1.15.0" "huggingface_hub==2.2.0" "safetensors==0.8.0" "tokenizers==0.23.3" numpy
```
```bash
export HF_HOME=/workspace/hf INTROSPECTION_DATA=/workspace/data
```
Now the token. The next command waits for you to **paste the token and press Enter**. Nothing
appears on screen while you paste; that is on purpose, so the token isn't shown or saved in the
command history.
```bash
read -s HF_TOKEN && export HF_TOKEN && echo "token set"
```

## 3. Quick test (2–4 minutes; uses the small 1B model, results thrown away)

```bash
python scripts/study17.py --model 1b --dry
```
It must end with a line containing `DONE` and, earlier, `SANITY PASSED`. If it shows an error, go
to §7.

## 4. The real run (about 1–2 hours)

```bash
nohup python scripts/study17.py --model 27b > /workspace/run27b.log 2>&1 &
```
To watch progress (press **Ctrl+C** to stop *watching*; the run keeps going):
```bash
tail -f /workspace/run27b.log
```
What you should see, in order: `FETCH` → `loaded google/gemma-3-27b-it` (after a 5–15 minute
download) → `VECTORS` → `GATE` → many `judged …` lines (the slowest part) → `GATE VERDICT (27b)` and
`GATE VERDICT (4b_judged_on_gpu)`. Then:
- if the 27B verdict is **REPORTS**: `SANITY PASSED` → `FINGERPRINT` → `DOSES` (twice) → test
  progress → `TEST DONE` → `DONE`;
- otherwise: `STOP: 27B Gate 1 = …` and `DONE`. **This is a valid, pre-registered outcome**, not a
  failure. Continue with §5.

If the browser tab closes, the run continues. Reopen a terminal and run the `tail -f` command again.

## 5. Package and download the results

When the log ends with `DONE`:
```bash
cd /workspace/introspection-localization-controls && python scripts/study17_analyze.py
```
```bash
tar czf /workspace/study17_gpu_results.tar.gz results/study17 results/study17_summary.md results/study17_summary.json -C /workspace/data logs
```
In Jupyter Lab's file panel (left), go to `/workspace`, right-click
**`study17_gpu_results.tar.gz` → Download**. Also download `run27b.log`. Check that the downloaded
file is larger than a few kilobytes. Put both files in the project folder on your laptop.

## 6. SHUT DOWN (do not skip)

1. In the terminal, clear the token: `unset HF_TOKEN`.
2. On runpod.io: **Pods → your pod → ⋮ (or the trash icon) → Terminate / Delete**. *Stop* alone
   still charges for the 150 GB disk; you want **Terminate**.
3. Refresh the Pods page: it must show **no pods**. Check **Billing**: the spend should be well under
   $40.
4. On huggingface.co → Settings → Access Tokens, **delete** the `runpod-study17` token.

Then tell Claude where the two downloaded files are. Claude extracts them, checks them, runs the
analysis, and commits the results.

## 7. If something goes wrong

| what you see | what to do |
|---|---|
| `401` / `403` / "gated repo" when loading Gemma | The token is wrong or Gemma access is missing: redo §0.2–0.3 and the `read -s` line, then rerun the same command. |
| `CUDA out of memory` | Rerun with smaller batches; finished work is kept: `export STUDY17_JUDGE_BATCH=8 STUDY17_GEN_BATCH=25 STUDY17_STEER_BATCH=32`, then the §4 command again. |
| The run stopped (crash, disconnect, pod restarted) | Run the §4 command again: every stage resumes where it stopped (finished trials are skipped). If the pod was terminated, start again from §1. |
| `AssertionError` mentioning `S4`/`S5` (sanity) | Don't retry. Download `run27b.log` (§5), terminate the pod (§6), and send the log to Claude. |
| Anything else, or the cost approaches $20 | Download the log, terminate the pod, and ask Claude. |

## Time and cost estimate

| step | A100 80GB (~$1.6/h) | H100 80GB (~$3/h) |
|---|---|---|
| setup + quick test | 15 min | 15 min |
| model download (55 GB) | 5–15 min | 5–15 min |
| vectors + Gate 1 generation (400 answers) | ~10 min | ~6 min |
| judging (800 answers, up to 500 words each) | ~30–45 min | ~20–30 min |
| fingerprint + doses + test (if REPORTS) | ~15–25 min | ~10–15 min |
| package, download, shut down | 10 min | 10 min |
| **total** | **~1.5–2 h ≈ $3–4** | **~1–1.5 h ≈ $3–5** |

These are estimates; even three times longer stays within $20. **Stop and ask if the pod has run
for more than 4 hours.**

## What is (and is not) on the rented machine

- This public repository and the downloaded model weights, deleted with the pod.
- Your Hugging Face token, **only** in the terminal's memory (`read -s`), never in a file or a log;
  you delete it afterwards.
- No other accounts or passwords. Nothing is uploaded from the machine anywhere; you download the
  results yourself.
