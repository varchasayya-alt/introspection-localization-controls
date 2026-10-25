"""Study 15 — push the button with pushes sized by the positive control. Plan: gates/PREREG_study15.md.

Llama-3.2-3B-Instruct; Study 14's prompts, token sets and batched last-token steering; steering at
block 18 (hidden_states[19]); injection direction and typical norm from Study 14's saved calibration
states; doses = push sizes at which the (yes - no) direction gives >= 25 / 50 / 90% "yes" on
dose-calibration trials.

Stages:
  stimuli    write stimuli/study15_stimuli.json (no model)
  direction  unit injection direction + typical norm N from Study 14's calibration states (no model)
  dryrun     timing on 5 calibration-half draws, placeholder doses (outputs discarded)
  calibrate  positive-control grid on 100 dose-calibration draws -> doses
  sanity     checks only;  test: sanity + 400 clean test draws;  summary: copy results

Usage:
  python scripts/study15.py --stage stimuli|direction|dryrun|calibrate|sanity|test|summary
"""
import argparse
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import introspect_lib as L  # noqa: E402
import study10 as T  # noqa: E402
import study14 as S14  # noqa: E402

REPO_ID = S14.REPO_ID
LAYER = 18  # block output; hidden_states[LAYER + 1]
CFG = {"n_cal": 100, "n_test": 400, "n_random": 5, "targets": [25, 50, 90], "grid_n": 40, "grid_lo": 0.02,
       "grid_hi": 16.0, "chunk": 20, "tol_logit": 1e-3,
       "seeds": {"cal": 20261051, "test": 20261052, "random": 20261053, "bootstrap": 20261054}}
GRID = [CFG["grid_lo"] * (CFG["grid_hi"] / CFG["grid_lo"]) ** (i / (CFG["grid_n"] - 1)) for i in range(CFG["grid_n"])]
STIM = L.REPO_ROOT / "stimuli" / "study15_stimuli.json"
OUT = L.DATA_ROOT / "study15"
LOG = L.DATA_ROOT / "logs" / "study15.log"
REPO_RESULTS = L.REPO_ROOT / "results" / "study15"
S14_STATES = L.DATA_ROOT / "study14" / "calib_states"
jread, sha = S14.jread, S14.sha


def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    L.log(msg, LOG)


# ----------------------------------------------------------------------------- no-model stages

def stage_stimuli():
    s14 = jread(L.REPO_ROOT / "stimuli" / "study14_stimuli.json")
    rc, rt = random.Random(CFG["seeds"]["cal"]), random.Random(CFG["seeds"]["test"])
    cal = [{"draw": i, "sentences": rc.sample(s14["calibration_pool"], 10)} for i in range(CFG["n_cal"])]
    test = [{"draw": i, "control_q": i % 10, "sentences": rt.sample(s14["test_pool"], 10)} for i in range(CFG["n_test"])]
    out = {"version": "study15-v1",
           "notes": "Frozen before any Study 15 data. Dose-calibration draws: rng 20261051 over Study 14's calibration "
                    "half; test draws: rng 20261052 over Study 14's test half, control question cycling 0-9.",
           "date_string": S14.DATE, "control_questions": s14["control_questions"],
           "calibration_pool": s14["calibration_pool"], "test_pool": s14["test_pool"],
           "cal_draws": cal, "test_draws": test, "grid_fractions": GRID,
           "random_direction_seed": CFG["seeds"]["random"], "bootstrap_seed": CFG["seeds"]["bootstrap"]}
    STIM.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {STIM.name}: {len(cal)} dose-calibration draws, {len(test)} test draws")


def stage_direction():
    OUT.mkdir(parents=True, exist_ok=True)
    s14 = jread(L.REPO_ROOT / "stimuli" / "study14_stimuli.json")
    hs = LAYER + 1
    ids = [d["draw"] for d in s14["cal_draws"]]
    st = {(i, x): torch.load(S14_STATES / f"d{i}_{x}.pt", weights_only=True)[hs].float() for i in ids for x in ("none", "inj")}
    clean = torch.stack([st[(i, "none")] for i in ids])
    v = torch.stack([st[(i, "inj")] for i in ids]).mean(0) - clean.mean(0)
    u = v / v.norm()
    N = float(clean.norm(dim=-1).mean())
    torch.save(u.clone(), OUT / "direction_unit.pt")
    res = {"layer": LAYER, "hidden_states_index": hs, "n_draws": len(ids), "raw_direction_norm": float(v.norm()),
           "typical_norm_N": N, "raw_norm_over_N": float(v.norm()) / N,
           "direction_file_sha256": sha(OUT / "direction_unit.pt"),
           "direction_float32_sha256": hashlib.sha256(u.contiguous().numpy().astype("float32").tobytes()).hexdigest()}
    (OUT / "direction.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    log(f"DIRECTION: block {LAYER}, |raw v| {res['raw_direction_norm']:.4f}, N {N:.3f} (ratio {res['raw_norm_over_N']:.4f})")


# ----------------------------------------------------------------------------- model context

class Ctx15:
    def __init__(self):
        OUT.mkdir(parents=True, exist_ok=True)
        rev = jread(L.REPO_ROOT / "stimuli" / "model_pins.json")[REPO_ID]
        t0 = time.time()
        self.model, self.tok = L.load_model(REPO_ID, rev)
        log(f"loaded {REPO_ID}@{rev} in {time.time() - t0:.0f}s")
        self.stim = jread(STIM)
        self.y, self.n = T.yes_no_ids(self.tok)
        self.yes_set, self.no_set = set(self.y), set(self.n)
        self.rand = L.random_unit_vectors(self.model.config.hidden_size, CFG["n_random"], CFG["seeds"]["random"])
        with torch.inference_mode():
            W = self.model.lm_head.weight
            p = W[self.y].float().mean(0) - W[self.n].float().mean(0)
        self.posdir = p / p.norm()

    def load_direction(self):
        meta = jread(L.REPO_ROOT / "results" / "study15" / "direction.json", jread(OUT / "direction.json"))
        assert sha(OUT / "direction_unit.pt") == meta["direction_file_sha256"]
        self.u = torch.load(OUT / "direction_unit.pt", weights_only=True).float()
        self.N = meta["typical_norm_N"]

    def prompt(self, sentences, question):
        return S14.P14(self.tok, L_marker(sentences), question)

    def read(self, logits):
        top = int(torch.argmax(logits))
        cls = "yes" if top in self.yes_set else ("no" if top in self.no_set else "neither")
        yn = float(torch.logsumexp(logits[self.y], -1) - torch.logsumexp(logits[self.n], -1))
        return {"cls": cls, "top": top, "yn": round(yn, 5)}


def L_marker(sentences):
    return [f"SENTENCE {i + 1}: {s}" for i, s in enumerate(sentences)]


def batched(ctx, P, cache, vecs):
    """steer_batch in chunks (memory), same computation."""
    return torch.cat([S14.steer_batch(ctx.model, P, cache, LAYER, vecs[i:i + CFG["chunk"]])
                      for i in range(0, vecs.shape[0], CFG["chunk"])])


# ----------------------------------------------------------------------------- dose calibration

def grid_trial(ctx, P, N):
    cl = L.clean_pass(ctx.model, P)
    lg = batched(ctx, P, cl["cache"], torch.stack([ctx.posdir * f * N for f in GRID]))
    return {"k0": ctx.read(cl["logits"]), "grid": [ctx.read(r) for r in lg]}


def stage_calibrate(ctx):
    ctx.load_direction()
    st = L.ResultStore(OUT / "dose_calibration.jsonl")
    t0 = time.time()
    for d in ctx.stim["cal_draws"]:
        key = f"cal|d{d['draw']}"
        if st.has(key):
            continue
        st.add({"key": key, "draw": d["draw"], **grid_trial(ctx, ctx.prompt(d["sentences"], T.Q_DETECT), ctx.N)})
        if (d["draw"] + 1) % 10 == 0:
            log(f"dose calibration draw {d['draw'] + 1}/{CFG['n_cal']}; {(time.time() - t0) / 60:.1f} min")
    R = list(st.records.values())
    yes = [100 * sum(r["grid"][i]["cls"] == "yes" for r in R) / len(R) for i in range(len(GRID))]
    nei = [100 * sum(r["grid"][i]["cls"] == "neither" for r in R) / len(R) for i in range(len(GRID))]
    doses = {}
    for t in CFG["targets"]:
        hit = [i for i in range(len(GRID)) if yes[i] >= t]
        doses[str(t)] = GRID[hit[0]] if hit else None
    res = {"grid_fractions": GRID, "yes_rate": yes, "neither_rate": nei, "k0_yes_rate": 100 * sum(r["k0"]["cls"] == "yes" for r in R) / len(R),
           "doses_fraction_of_N": doses, "typical_norm_N": ctx.N, "stop": doses["90"] is None, "n": len(R)}
    (OUT / "doses.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    log(f"DOSES: {json.dumps(doses)} (fractions of N = {ctx.N:.3f}); stop={res['stop']}; {(time.time() - t0) / 60:.1f} min")
    return res


def load_doses():
    meta = jread(L.REPO_ROOT / "results" / "study15" / "doses.json", jread(OUT / "doses.json"))
    assert not meta["stop"]
    return [meta["doses_fraction_of_N"][str(t)] for t in CFG["targets"]]


# ----------------------------------------------------------------------------- test

def conditions(ctx, question):
    if question == "ctl":
        return [("inj", ctx.u)]
    return [("inj", ctx.u)] + [(f"rand{j}", r) for j, r in enumerate(ctx.rand)] + [("pos", ctx.posdir)]


def steer_trial(ctx, P, fracs, question):
    cl = L.clean_pass(ctx.model, P)
    out = {"k0": ctx.read(cl["logits"])}
    names = [(f"{nm}|d{i + 1}", unit * f * ctx.N) for nm, unit in conditions(ctx, question) for i, f in enumerate(fracs)]
    lg = batched(ctx, P, cl["cache"], torch.stack([x for _, x in names]))
    for (k, _), row in zip(names, lg):
        out[k] = ctx.read(row)
    return out


def stage_sanity(ctx, fracs):
    m, s, res = ctx.model, ctx.stim, {}
    assert not set(s["calibration_pool"]) & set(s["test_pool"])
    assert all(set(d["sentences"]) <= set(s["test_pool"]) for d in s["test_draws"])
    assert all(set(d["sentences"]) <= set(s["calibration_pool"]) for d in s["cal_draws"])
    P = ctx.prompt(s["cal_draws"][0]["sentences"], T.Q_DETECT)
    n = len(P.ids)
    assert f"Today Date: {S14.DATE}" in P.text
    x0 = L.run_full(m, P)
    x1 = L.run_full(m, P, layer=LAYER, vec=torch.zeros_like(ctx.u), inj_abs=[n - 1])
    res["S3_zero_vector_max_abs_diff"] = float((x0 - x1).abs().max())
    assert res["S3_zero_vector_max_abs_diff"] == 0.0
    cl = L.clean_pass(m, P)
    res["S4_k0_clean_vs_full"] = float((cl["logits"] - x0).abs().max())
    top = fracs[-1] * ctx.N
    for nm, unit in (("inj", ctx.u), ("pos", ctx.posdir)):
        fi = L.run_full(m, P, layer=LAYER, vec=unit * top, inj_abs=[n - 1])
        si = L.run_suffix(m, P, n - 1, cl["cache"], LAYER, unit * top, [n - 1])["logits"]
        res[f"S5_suffix_vs_full_{nm}_top"] = float((fi - si).abs().max())
    names = [unit * f * ctx.N for _, unit in conditions(ctx, "det") for f in fracs]
    bl = batched(ctx, P, cl["cache"], torch.stack(names))
    singles = [L.run_suffix(m, P, n - 1, cl["cache"], LAYER, x, [n - 1])["logits"] for x in names]
    res["S7_batched_vs_single_max"] = max(float((bl[i] - singles[i]).abs().max()) for i in range(len(names)))
    res["S7_batched_same_top_token"] = all(int(bl[i].argmax()) == int(singles[i].argmax()) for i in range(len(names)))
    assert res["S7_batched_same_top_token"]
    assert all(v < CFG["tol_logit"] for k, v in res.items() if k.startswith(("S4", "S5", "S7")) and not isinstance(v, bool))
    (OUT / "sanity.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    log(f"SANITY PASSED: {json.dumps(res)}")


def stage_test(ctx, fracs):
    st = L.ResultStore(OUT / "test.jsonl")
    t0, done = time.time(), 0
    for d in ctx.stim["test_draws"]:
        for q, text in (("det", T.Q_DETECT), ("ctl", ctx.stim["control_questions"][d["control_q"]] + " Answer yes or no.")):
            key = f"{q}|d{d['draw']}"
            if st.has(key):
                continue
            st.add({"key": key, "draw": d["draw"], "question": q, "control_q": d["control_q"],
                    "out": steer_trial(ctx, ctx.prompt(d["sentences"], text), fracs, q)})
        done += 1
        if done % 25 == 0:
            log(f"test draw {d['draw'] + 1}/{CFG['n_test']}; {(time.time() - t0) / 60:.1f} min")
    log(f"TEST DONE: {len(st.records)}/{2 * CFG['n_test']} records")


def stage_dryrun(ctx):
    """Timing only: 5 calibration-half draws, full grid + a test-style trial with placeholder doses."""
    ctx.u = L.random_unit_vectors(ctx.model.config.hidden_size, 1, seed=999)[0]
    ctx.N = 1.0
    tc, tt = [], []
    for d in ctx.stim["cal_draws"][:5]:
        P = ctx.prompt(d["sentences"], T.Q_DETECT)
        t1 = time.time()
        grid_trial(ctx, P, ctx.N)
        tc.append(time.time() - t1)
        t1 = time.time()
        steer_trial(ctx, P, [0.5, 1.0, 2.0], "det")
        steer_trial(ctx, ctx.prompt(d["sentences"], "Is London in Japan? Answer yes or no."), [0.5, 1.0, 2.0], "ctl")
        tt.append(time.time() - t1)
    est = {"calib_sec_per_draw": sum(tc) / 5, "test_sec_per_draw": sum(tt) / 5,
           "est_calib_h": sum(tc) / 5 * CFG["n_cal"] / 3600, "est_test_h": sum(tt) / 5 * CFG["n_test"] / 3600}
    est["est_total_h"] = est["est_calib_h"] + est["est_test_h"]
    (OUT / "dryrun_timing.json").write_text(json.dumps(est, indent=2), encoding="utf-8")
    log(f"DRY RUN TIMING: {json.dumps({k: round(x, 3) for k, x in est.items()})}")


def stage_summary():
    REPO_RESULTS.mkdir(parents=True, exist_ok=True)
    for f in sorted(OUT.glob("*.json")) + sorted(OUT.glob("*.jsonl")):
        shutil.copy2(f, REPO_RESULTS / f.name)
    log("copied results to results/study15")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["stimuli", "direction", "dryrun", "calibrate", "sanity", "test", "summary"])
    args = ap.parse_args()
    if args.stage == "stimuli":
        return stage_stimuli()
    if args.stage == "direction":
        stage_direction()
        return stage_summary()
    if args.stage == "summary":
        return stage_summary()
    ctx = Ctx15()
    if args.stage == "dryrun":
        stage_dryrun(ctx)
    elif args.stage == "calibrate":
        stage_calibrate(ctx)
    else:
        ctx.load_direction()
        fracs = load_doses()
        stage_sanity(ctx, fracs)
        if args.stage == "test":
            stage_test(ctx, fracs)
    stage_summary()
    log("DONE")


if __name__ == "__main__":
    main()
