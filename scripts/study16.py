"""Study 16 — Study 15 replicated on Llama-3.1-8B-Instruct. Plan: gates/PREREG_study16.md.

Steering at block 21 (hidden_states[22], Study 11's Llama lens peak). Study 14/15 prompts (date
fixed), token sets and batched last-token steering. Fresh direction calibration with the authors'
vectors (layer 2, alpha 5); two-stage dose search on the (yes - no) positive control.

Stages:
  stimuli   write stimuli/study16_stimuli.json (no model)
  dryrun    timing on 5 calibration-half draws, placeholder direction and doses (discarded)
  dircal    100 draws clean + injected -> unit direction at block 21 and typical norm N
  dosecal   stage 1 (20 draws, coarse geometric grid) -> bracket; stage 2 (100 draws, fine linear grid) -> doses
  sanity / test / summary

Usage:
  python scripts/study16.py --stage stimuli|dryrun|dircal|dosecal|sanity|test|summary
"""
import argparse
import copy
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
import study11 as E  # noqa: E402
import study14 as S14  # noqa: E402

REPO_ID = "meta-llama/Llama-3.1-8B-Instruct"
LAYER = 21  # block output; hidden_states[22] (Study 11 lens index 22)
INJ = {"layer": 2, "alpha": 5.0, "vec_study": 4}
CFG = {"n_dir": 100, "n_stage1": 20, "n_stage2": 100, "n_test": 400, "n_random": 5, "targets": [25, 50, 90],
       "coarse_n": 40, "coarse_lo": 0.02, "coarse_hi": 16.0, "fine_n": 40, "chunk": 24, "tol_logit": 1e-3,
       "seeds": {"dir": 20261061, "dose": 20261062, "test": 20261063, "random": 20261064, "bootstrap": 20261065}}
COARSE = [CFG["coarse_lo"] * (CFG["coarse_hi"] / CFG["coarse_lo"]) ** (i / (CFG["coarse_n"] - 1)) for i in range(CFG["coarse_n"])]
STIM = L.REPO_ROOT / "stimuli" / "study16_stimuli.json"
OUT = L.DATA_ROOT / "study16"
LOG = L.DATA_ROOT / "logs" / "study16.log"
REPO_RESULTS = L.REPO_ROOT / "results" / "study16"
jread, sha = S14.jread, S14.sha


def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    L.log(msg, LOG)


def frozen(name):
    """Prefer the committed copy in results/ (the frozen value) over the working copy."""
    return jread(REPO_RESULTS / name, jread(OUT / name))


# ----------------------------------------------------------------------------- stimuli

def stage_stimuli():
    s14 = jread(L.REPO_ROOT / "stimuli" / "study14_stimuli.json")
    s3 = jread(L.REPO_ROOT / "stimuli" / "study3_stimuli.json")
    pairs = [(c, t) for c in s3["concepts"] for t in ("avg", "last")]
    rd, rq, rt = (random.Random(CFG["seeds"][k]) for k in ("dir", "dose", "test"))
    cal, test = s14["calibration_pool"], s14["test_pool"]
    dirs = [{"draw": i, "concept": pairs[i % 20][0], "vec_type": pairs[i % 20][1], "position": i % 10 + 1,
             "sentences": rd.sample(cal, 10)} for i in range(CFG["n_dir"])]
    dose = [{"draw": i, "stage": 1 if i < CFG["n_stage1"] else 2, "sentences": rq.sample(cal, 10)}
            for i in range(CFG["n_stage1"] + CFG["n_stage2"])]
    tests = [{"draw": i, "control_q": i % 10, "sentences": rt.sample(test, 10)} for i in range(CFG["n_test"])]
    out = {"version": "study16-v1",
           "notes": "Frozen before any Study 16 data. Direction draws rng 20261061, dose draws rng 20261062 (0-19 stage 1, "
                    "20-119 stage 2), both over Study 14's calibration half; test draws rng 20261063 over Study 14's test "
                    "half, control question cycling 0-9.",
           "date_string": S14.DATE, "concepts": s3["concepts"], "control_questions": s14["control_questions"],
           "calibration_pool": cal, "test_pool": test, "dir_draws": dirs, "dose_draws": dose, "test_draws": tests,
           "coarse_grid": COARSE, "random_direction_seed": CFG["seeds"]["random"], "bootstrap_seed": CFG["seeds"]["bootstrap"]}
    STIM.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {STIM.name}: {len(dirs)} direction, {len(dose)} dose, {len(tests)} test draws")


# ----------------------------------------------------------------------------- context

class Ctx16:
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

    def load_injection(self):
        self.vecs = E.load_vectors("llama", INJ["vec_study"])  # checks the SHA-256 hashes

    def load_direction(self):
        meta = frozen("direction.json")
        assert sha(OUT / "direction_unit.pt") == meta["direction_file_sha256"]
        self.u = torch.load(OUT / "direction_unit.pt", weights_only=True).float()
        self.N = meta["typical_norm_N"]

    def prompt(self, sentences, question):
        return S14.P14(self.tok, [f"SENTENCE {i + 1}: {s}" for i, s in enumerate(sentences)], question)

    def read(self, logits):
        top = int(torch.argmax(logits))
        cls = "yes" if top in self.yes_set else ("no" if top in self.no_set else "neither")
        yn = float(torch.logsumexp(logits[self.y], -1) - torch.logsumexp(logits[self.n], -1))
        return {"cls": cls, "top": top, "yn": round(yn, 5)}


def batched(ctx, P, cache, vecs):
    return torch.cat([S14.steer_batch(ctx.model, P, cache, LAYER, vecs[i:i + CFG["chunk"]])
                      for i in range(0, vecs.shape[0], CFG["chunk"])])


@torch.inference_mode()
def shared_prefix_pass(model, Pd, Pc):
    """Clean logits + caches for two prompts that share a token prefix: Pd in full, then Pc's own
    tail on a copy of Pd's cache cropped to the shared prefix (exact: the model can't look ahead)."""
    j = next(i for i, (a, b) in enumerate(zip(Pd.ids, Pc.ids)) if a != b)
    out = model(input_ids=Pd.tensor(), use_cache=True, logits_to_keep=1)
    cd = out.past_key_values
    cc = copy.deepcopy(cd)
    cc.crop(-(cc.get_seq_length() - j))
    oc = model(input_ids=Pc.tensor()[:, j:], past_key_values=cc, use_cache=True, logits_to_keep=1)
    return (out.logits[0, -1].float(), cd), (oc.logits[0, -1].float(), oc.past_key_values), j


# ----------------------------------------------------------------------------- direction calibration

def dir_states(ctx, d):
    E.CFG["layer"] = INJ["layer"]
    P = ctx.prompt(d["sentences"], T.Q_DETECT)
    lg0, st0, cache = E.full_states(ctx.model, P)
    idx = P.arm[d["position"] - 1]["sentence"]
    lg1, st1 = E.suffix_states(ctx.model, P, idx[0], cache, INJ["alpha"] * ctx.vecs[(d["concept"], d["vec_type"])], idx)
    return st0.float(), st1.float(), lg0, lg1


def stage_dircal(ctx):
    ctx.load_injection()
    sd = OUT / "dir_states"
    sd.mkdir(parents=True, exist_ok=True)
    st = L.ResultStore(OUT / "direction_runs.jsonl")
    t0 = time.time()
    for d in ctx.stim["dir_draws"]:
        key = f"dir|d{d['draw']}"
        if st.has(key):
            continue
        s0, s1, lg0, lg1 = dir_states(ctx, d)
        torch.save(s0.clone(), sd / f"d{d['draw']}_none.pt")
        torch.save(s1.clone(), sd / f"d{d['draw']}_inj.pt")
        st.add({"key": key, "draw": d["draw"], "concept": d["concept"], "vec_type": d["vec_type"],
                "position": d["position"], "clean": ctx.read(lg0), "injected": ctx.read(lg1)})
        if (d["draw"] + 1) % 10 == 0:
            log(f"direction draw {d['draw'] + 1}/{CFG['n_dir']}; {(time.time() - t0) / 60:.1f} min")
    hs = LAYER + 1
    ids = [d["draw"] for d in ctx.stim["dir_draws"]]
    clean = torch.stack([torch.load(sd / f"d{i}_none.pt", weights_only=True)[hs] for i in ids])
    inj = torch.stack([torch.load(sd / f"d{i}_inj.pt", weights_only=True)[hs] for i in ids])
    v = inj.mean(0) - clean.mean(0)
    u = v / v.norm()
    N = float(clean.norm(dim=-1).mean())
    torch.save(u.clone(), OUT / "direction_unit.pt")
    res = {"layer": LAYER, "hidden_states_index": hs, "n_draws": len(ids), "raw_direction_norm": float(v.norm()),
           "typical_norm_N": N, "raw_norm_over_N": float(v.norm()) / N,
           "direction_file_sha256": sha(OUT / "direction_unit.pt"),
           "direction_float32_sha256": hashlib.sha256(u.contiguous().numpy().astype("float32").tobytes()).hexdigest(),
           "cos_with_yes_minus_no": float(u @ ctx.posdir)}
    (OUT / "direction.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    log(f"DIRECTION: block {LAYER}, |raw v| {res['raw_direction_norm']:.4f}, N {N:.3f}; {(time.time() - t0) / 60:.1f} min")


# ----------------------------------------------------------------------------- dose calibration

def grid_trial(ctx, P, fracs):
    cl = L.clean_pass(ctx.model, P)
    lg = batched(ctx, P, cl["cache"], torch.stack([ctx.posdir * f * ctx.N for f in fracs]))
    return {"k0": ctx.read(cl["logits"]), "grid": [ctx.read(r) for r in lg]}


def run_grid(ctx, store, draws, fracs, tag):
    t0 = time.time()
    for d in draws:
        key = f"{tag}|d{d['draw']}"
        if not store.has(key):
            store.add({"key": key, "draw": d["draw"], "fracs": fracs, **grid_trial(ctx, ctx.prompt(d["sentences"], T.Q_DETECT), fracs)})
        if (d["draw"] + 1) % 10 == 0:
            log(f"{tag} draw {d['draw'] + 1}; {(time.time() - t0) / 60:.1f} min")
    R = [store.records[f"{tag}|d{d['draw']}"] for d in draws]
    yes = [100 * sum(r["grid"][i]["cls"] == "yes" for r in R) / len(R) for i in range(len(fracs))]
    nei = [100 * sum(r["grid"][i]["cls"] == "neither" for r in R) / len(R) for i in range(len(fracs))]
    return yes, nei


def stage_dosecal(ctx):
    ctx.load_direction()
    st = L.ResultStore(OUT / "dose_calibration.jsonl")
    d1 = [d for d in ctx.stim["dose_draws"] if d["stage"] == 1]
    d2 = [d for d in ctx.stim["dose_draws"] if d["stage"] == 2]
    y1, n1 = run_grid(ctx, st, d1, COARSE, "s1")
    hi = [i for i in range(len(COARSE)) if y1[i] >= 90]
    res = {"typical_norm_N": ctx.N, "stage1": {"fracs": COARSE, "yes_rate": y1, "neither_rate": n1}}
    if not hi:
        res.update(stop=True, reason="positive control never reaches 90% on stage 1")
        (OUT / "doses.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        log("DOSES: STOP (90% never reached on stage 1)")
        return res
    f_hi = COARSE[hi[0]]
    lo = [i for i in range(hi[0]) if y1[i] < 25]
    f_lo = COARSE[lo[-1]] if lo else 0.0
    a, b = (0.8 * f_lo if f_lo > 0 else f_hi / 4), 1.25 * f_hi
    fine = [a + (b - a) * i / (CFG["fine_n"] - 1) for i in range(CFG["fine_n"])]
    y2, n2 = run_grid(ctx, st, d2, fine, "s2")
    doses, notes = {}, []
    for t in CFG["targets"]:
        h = [i for i in range(len(fine)) if y2[i] >= t]
        if h:
            doses[str(t)] = fine[h[0]]
        else:
            doses[str(t)] = fine[-1]
            notes.append(f"{t}% not reached on stage 2; dose set to the top of the fine grid")
    vals = [doses[str(t)] for t in CFG["targets"]]
    res.update(stop=False, f_lo=f_lo, f_hi=f_hi, stage2={"fracs": fine, "yes_rate": y2, "neither_rate": n2},
               doses_fraction_of_N=doses, collapsed=len(set(vals)) < 3, notes=notes,
               k0_yes_rate_stage2=100 * sum(st.records[f"s2|d{d['draw']}"]["k0"]["cls"] == "yes" for d in d2) / len(d2))
    (OUT / "doses.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    log(f"DOSES: bracket [{f_lo:.4f}, {f_hi:.4f}] -> {json.dumps(doses)}; collapsed={res['collapsed']}; {notes}")
    return res


def load_doses():
    meta = frozen("doses.json")
    assert not meta["stop"]
    return [meta["doses_fraction_of_N"][str(t)] for t in CFG["targets"]]


# ----------------------------------------------------------------------------- test

def conditions(ctx, question):
    if question == "ctl":
        return [("inj", ctx.u)]
    return [("inj", ctx.u)] + [(f"rand{j}", r) for j, r in enumerate(ctx.rand)] + [("pos", ctx.posdir)]


def steer(ctx, P, logits, cache, fracs, question):
    """k0 plus every condition at every dose; identical dose sizes are run once and shared."""
    uniq = sorted(set(fracs))
    out = {"k0": ctx.read(logits)}
    names = [(nm, f, unit * f * ctx.N) for nm, unit in conditions(ctx, question) for f in uniq]
    lg = batched(ctx, P, cache, torch.stack([x for _, _, x in names]))
    got = {(nm, f): ctx.read(row) for (nm, f, _), row in zip(names, lg)}
    for nm, _ in conditions(ctx, question):
        for i, f in enumerate(fracs):
            out[f"{nm}|d{i + 1}"] = got[(nm, f)]
    return out


def test_draw(ctx, d, fracs):
    Pd = ctx.prompt(d["sentences"], T.Q_DETECT)
    Pc = ctx.prompt(d["sentences"], ctx.stim["control_questions"][d["control_q"]] + " Answer yes or no.")
    (ld, cd), (lc, cc), _ = shared_prefix_pass(ctx.model, Pd, Pc)
    return steer(ctx, Pd, ld, cd, fracs, "det"), steer(ctx, Pc, lc, cc, fracs, "ctl")


def stage_sanity(ctx, fracs):
    m, s, res = ctx.model, ctx.stim, {}
    assert not set(s["calibration_pool"]) & set(s["test_pool"])
    assert all(set(d["sentences"]) <= set(s["test_pool"]) for d in s["test_draws"])
    assert all(set(d["sentences"]) <= set(s["calibration_pool"]) for d in s["dir_draws"] + s["dose_draws"])
    ctx.load_injection()  # re-checks the authors' vector hashes
    d = s["test_draws"][0]
    Pd = ctx.prompt(d["sentences"], T.Q_DETECT)
    Pc = ctx.prompt(d["sentences"], s["control_questions"][d["control_q"]] + " Answer yes or no.")
    assert f"Today Date: {S14.DATE}" in Pd.text and Pd.text.count(S14.S6.BOS) == 1
    n = len(Pd.ids)
    x0 = L.run_full(m, Pd)
    x1 = L.run_full(m, Pd, layer=LAYER, vec=torch.zeros_like(ctx.u), inj_abs=[n - 1])
    res["S3_zero_vector_max_abs_diff"] = float((x0 - x1).abs().max())
    assert res["S3_zero_vector_max_abs_diff"] == 0.0
    (ld, cd), (lc, cc), j = shared_prefix_pass(m, Pd, Pc)
    res["S4_shared_prefix_tokens"] = j
    res["S4_det_vs_full"] = float((ld - x0).abs().max())
    res["S4_ctl_vs_full"] = float((lc - L.run_full(m, Pc)).abs().max())
    top = max(fracs) * ctx.N
    for nm, unit, P, c in (("inj_det", ctx.u, Pd, cd), ("pos_det", ctx.posdir, Pd, cd), ("inj_ctl", ctx.u, Pc, cc)):
        k = len(P.ids)
        fi = L.run_full(m, P, layer=LAYER, vec=unit * top, inj_abs=[k - 1])
        si = S14.steer_batch(m, P, c, LAYER, (unit * top)[None])[0]
        res[f"S5_last_token_vs_full_{nm}_top"] = float((fi - si).abs().max())
    vecs = [unit * f * ctx.N for _, unit in conditions(ctx, "det") for f in sorted(set(fracs))]
    bl = batched(ctx, Pd, cd, torch.stack(vecs))
    singles = [L.run_suffix(m, Pd, n - 1, cd, LAYER, x, [n - 1])["logits"] for x in vecs]
    res["S7_batched_vs_single_max"] = max(float((bl[i] - singles[i]).abs().max()) for i in range(len(vecs)))
    res["S7_batched_same_top_token"] = all(int(bl[i].argmax()) == int(singles[i].argmax()) for i in range(len(vecs)))
    assert res["S7_batched_same_top_token"]
    assert all(v < CFG["tol_logit"] for k, v in res.items() if k.startswith(("S4_det", "S4_ctl", "S5", "S7")) and not isinstance(v, bool))
    (OUT / "sanity.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    log(f"SANITY PASSED: {json.dumps(res)}")


def stage_test(ctx, fracs):
    st = L.ResultStore(OUT / "test.jsonl")
    t0, done = time.time(), 0
    for d in ctx.stim["test_draws"]:
        if st.has(f"det|d{d['draw']}") and st.has(f"ctl|d{d['draw']}"):
            continue
        od, oc = test_draw(ctx, d, fracs)
        for q, o in (("det", od), ("ctl", oc)):
            st.add({"key": f"{q}|d{d['draw']}", "draw": d["draw"], "question": q, "control_q": d["control_q"], "out": o})
        done += 1
        if done % 25 == 0:
            log(f"test draw {d['draw'] + 1}/{CFG['n_test']}; {(time.time() - t0) / 60:.1f} min")
    log(f"TEST DONE: {len(st.records)}/{2 * CFG['n_test']} records")


# ----------------------------------------------------------------------------- dry run

def stage_dryrun(ctx):
    """Timing only, on calibration-half draws; placeholder direction (random) and doses; discarded."""
    ctx.load_injection()
    ctx.u = L.random_unit_vectors(ctx.model.config.hidden_size, 1, seed=999)[0]
    ctx.N = 1.0
    t = {"dir": [], "grid": [], "test": []}
    for d in ctx.stim["dir_draws"][:5]:
        t1 = time.time()
        dir_states(ctx, d)
        t["dir"].append(time.time() - t1)
        P = ctx.prompt(d["sentences"], T.Q_DETECT)
        t1 = time.time()
        grid_trial(ctx, P, COARSE)
        t["grid"].append(time.time() - t1)
        t1 = time.time()
        test_draw(ctx, {"sentences": d["sentences"], "control_q": d["draw"] % 10}, [0.5, 1.0, 2.0])
        t["test"].append(time.time() - t1)
    m = {k: sum(v) / len(v) for k, v in t.items()}
    est = {"dir_sec_per_draw": m["dir"], "grid_sec_per_draw": m["grid"], "test_sec_per_draw": m["test"],
           "est_dircal_h": m["dir"] * CFG["n_dir"] / 3600,
           "est_dosecal_h": m["grid"] * (CFG["n_stage1"] + CFG["n_stage2"]) / 3600,
           "est_test_h": m["test"] * CFG["n_test"] / 3600}
    est["est_total_h"] = est["est_dircal_h"] + est["est_dosecal_h"] + est["est_test_h"]
    (OUT / "dryrun_timing.json").write_text(json.dumps(est, indent=2), encoding="utf-8")
    log(f"DRY RUN TIMING: {json.dumps({k: round(x, 3) for k, x in est.items()})}")


def stage_summary():
    REPO_RESULTS.mkdir(parents=True, exist_ok=True)
    for f in sorted(OUT.glob("*.json")) + sorted(OUT.glob("*.jsonl")):
        shutil.copy2(f, REPO_RESULTS / f.name)
    log("copied results to results/study16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["stimuli", "dryrun", "dircal", "dosecal", "sanity", "test", "summary"])
    args = ap.parse_args()
    if args.stage == "stimuli":
        return stage_stimuli()
    if args.stage == "summary":
        return stage_summary()
    ctx = Ctx16()
    if args.stage == "dryrun":
        stage_dryrun(ctx)
    elif args.stage == "dircal":
        stage_dircal(ctx)
    elif args.stage == "dosecal":
        stage_dosecal(ctx)
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
