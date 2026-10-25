"""Study 14 — "push the button": steer clean trials with the injected direction. Plan: gates/PREREG_study14.md.

Llama-3.2-3B-Instruct, Study 10's yes/no prompts (date line fixed), Study 13's 3B injection (layer 2,
alpha 7.937, 3B concept vectors, checked by hash) used only to compute the direction.

Stages:
  stimuli    write stimuli/study14_stimuli.json from the plan's seeds (no model)
  dryrun     timing on 5 calibration-half draws with a placeholder direction (outputs discarded)
  calibrate  answer-position states, clean vs injected (detection question) -> probe AUC per layer
             (fit half -> held-out half) -> steering layer -> direction v = mean(inj) - mean(clean)
  sanity     last-token-only steering equals a full pass; alpha 0 changes nothing; disjoint splits
  test       400 clean draws x {detection, control} x doses x conditions (a)-(d)
  summary    copy result files into results/study14

Usage:
  python scripts/study14.py --stage stimuli|dryrun|calibrate|sanity|test|summary
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
import study3 as S3  # noqa: E402
import study4 as S4  # noqa: E402
import study6 as S6  # noqa: E402
import study10 as T  # noqa: E402
import study11 as E  # noqa: E402
from study11_analyze import auc, fit_predict  # noqa: E402

REPO_ID = "meta-llama/Llama-3.2-3B-Instruct"
DATE = "24 Oct 2026"
CFG = {"inj_layer": 2, "n_cal": 100, "n_fit": 50, "n_test": 400, "doses": [0, 1, 2, 4, 8], "n_random": 5,
       "candidate_layers": list(range(0, 27)), "auc_min": 0.75, "tol_logit": 1e-3,
       "seeds": {"split": 20261041, "cal": 20261042, "test": 20261043, "random": 20261044, "bootstrap": 20261045}}
STIM = L.REPO_ROOT / "stimuli" / "study14_stimuli.json"
OUT = L.DATA_ROOT / "study14"
LOG = L.DATA_ROOT / "logs" / "study14.log"
REPO_RESULTS = L.REPO_ROOT / "results" / "study14"
S13 = L.DATA_ROOT / "study13" / "3b_l2"
S13_RESULTS = L.REPO_ROOT / "results" / "study13" / "3b_l2"


def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    L.log(msg, LOG)


def jread(p, default=None):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class P14(T.P10):
    """Study 10's two-turn prompt, with the Llama-3.2 template's date line fixed."""

    def __init__(self, tok, spans, question):
        self.spans = list(spans)
        block = " ".join(self.spans)
        self.msgs = [{"role": "user", "content": T.INTRO10}, {"role": "assistant", "content": "Ok."},
                     {"role": "user", "content": f"{block}\n\n{question}"}]
        s = tok.apply_chat_template(self.msgs, tokenize=False, add_generation_prompt=True, date_string=DATE)
        self.text = s if s.startswith(S6.BOS) else S6.BOS + s
        enc = tok(self.text, add_special_tokens=False, return_offsets_mapping=True)
        self.ids, self.offsets = enc["input_ids"], enc["offset_mapping"]
        b0 = self.text.find(block)
        assert b0 >= 0
        self.user_tok = self._span_tokens(b0)
        self.arm = S4.arm_spans(self, self.spans, b0)


# ----------------------------------------------------------------------------- stimuli

def stage_stimuli():
    s3 = jread(L.REPO_ROOT / "stimuli" / "study3_stimuli.json")
    s10 = jread(L.REPO_ROOT / "stimuli" / "study10_stimuli.json")
    pool = list(s3["sentence_pool"])
    random.Random(CFG["seeds"]["split"]).shuffle(pool)
    cal_pool, test_pool = pool[:50], pool[50:]
    pairs = [(c, t) for c in s3["concepts"] for t in ("avg", "last")]
    rc, rt = random.Random(CFG["seeds"]["cal"]), random.Random(CFG["seeds"]["test"])
    cal = [{"draw": i, "concept": pairs[i % 20][0], "vec_type": pairs[i % 20][1], "position": i % 10 + 1,
            "half": "fit" if i < CFG["n_fit"] else "heldout", "sentences": rc.sample(cal_pool, 10)}
           for i in range(CFG["n_cal"])]
    test = [{"draw": i, "control_q": i % 10, "sentences": rt.sample(test_pool, 10)} for i in range(CFG["n_test"])]
    out = {"version": "study14-v1",
           "notes": "Frozen before any Study 14 data. Pool = study3 sentence_pool shuffled with seed 20261041: first 50 "
                    "calibration, last 50 test. Calibration draws: rng 20261042, 20 (concept, vec_type) pairs cycled, "
                    "position cycling 1-10, draws 0-49 fit / 50-99 held-out. Test draws: rng 20261043, control "
                    "question cycling 0-9. Control questions = Study 10's (true answer 'no').",
           "date_string": DATE, "control_questions": s10["control_questions"], "concepts": s3["concepts"],
           "calibration_pool": cal_pool, "test_pool": test_pool, "cal_draws": cal, "test_draws": test,
           "random_direction_seed": CFG["seeds"]["random"], "bootstrap_seed": CFG["seeds"]["bootstrap"]}
    assert not set(cal_pool) & set(test_pool)
    STIM.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {STIM.name}: {len(cal)} calibration draws, {len(test)} test draws")


# ----------------------------------------------------------------------------- context

class Ctx14:
    def __init__(self):
        OUT.mkdir(parents=True, exist_ok=True)
        rev = jread(L.REPO_ROOT / "stimuli" / "model_pins.json")[REPO_ID]
        t0 = time.time()
        self.model, self.tok = L.load_model(REPO_ID, rev)
        log(f"loaded {REPO_ID}@{rev} in {time.time() - t0:.0f}s")
        self.stim = jread(STIM)
        self.y, self.n = T.yes_no_ids(self.tok)
        self.yes_set, self.no_set = set(self.y), set(self.n)
        d = self.model.config.hidden_size
        self.rand = L.random_unit_vectors(d, CFG["n_random"], CFG["seeds"]["random"])
        with torch.inference_mode():
            W = self.model.lm_head.weight
            u = W[self.y].float().mean(0) - W[self.n].float().mean(0)
        self.posdir = u / u.norm()

    def load_injection(self):
        """Study 13's 3B vectors (hash-checked) and alpha."""
        meta = jread(S13_RESULTS / "vectors.json")
        assert meta["layer"] == CFG["inj_layer"]
        self.vecs = {}
        for f, h in meta["sha256"].items():
            assert sha(S13 / "vectors" / f) == h, f
        for c in self.stim["concepts"]:
            for t in ("avg", "last"):
                v = torch.load(S13 / "vectors" / f"{c}_{CFG['inj_layer']}_{t}.pt", weights_only=True).float()
                self.vecs[(c, t)] = v / v.norm()
        self.alpha = jread(S13_RESULTS / "calibration.json")["alpha"]

    def prompt(self, sentences, question):
        return P14(self.tok, S3.marker_spans(sentences), question)

    def ctrl_q(self, d):
        return self.stim["control_questions"][d["control_q"]] + " Answer yes or no."

    def read(self, logits):
        top = int(torch.argmax(logits))
        cls = "yes" if top in self.yes_set else ("no" if top in self.no_set else "neither")
        yn = float(torch.logsumexp(logits[self.y], -1) - torch.logsumexp(logits[self.n], -1))
        return {"cls": cls, "top": top, "yn": round(yn, 5)}


# ----------------------------------------------------------------------------- calibration

def cal_states(ctx, d):
    E.CFG["layer"] = CFG["inj_layer"]
    P = ctx.prompt(d["sentences"], T.Q_DETECT)
    lg0, st0, cache = E.full_states(ctx.model, P)
    idx = P.arm[d["position"] - 1]["sentence"]
    lg1, st1 = E.suffix_states(ctx.model, P, idx[0], cache, ctx.alpha * ctx.vecs[(d["concept"], d["vec_type"])], idx)
    return st0.float(), st1.float(), lg0, lg1


def stage_calibrate(ctx):
    ctx.load_injection()
    sd = OUT / "calib_states"
    sd.mkdir(parents=True, exist_ok=True)
    st = L.ResultStore(OUT / "calibration_runs.jsonl")
    t0 = time.time()
    for d in ctx.stim["cal_draws"]:
        key = f"cal|d{d['draw']}"
        if st.has(key):
            continue
        s0, s1, lg0, lg1 = cal_states(ctx, d)
        torch.save(s0.clone(), sd / f"d{d['draw']}_none.pt")
        torch.save(s1.clone(), sd / f"d{d['draw']}_inj.pt")
        st.add({"key": key, "draw": d["draw"], "half": d["half"], "concept": d["concept"], "vec_type": d["vec_type"],
                "position": d["position"], "clean": ctx.read(lg0), "injected": ctx.read(lg1)})
        if (d["draw"] + 1) % 10 == 0:
            log(f"calibration draw {d['draw'] + 1}/{CFG['n_cal']}; {(time.time() - t0) / 60:.1f} min")
    draws = ctx.stim["cal_draws"]
    S = {(d["draw"], x): torch.load(sd / f"d{d['draw']}_{x}.pt", weights_only=True) for d in draws for x in ("none", "inj")}
    fit = [d["draw"] for d in draws if d["half"] == "fit"]
    held = [d["draw"] for d in draws if d["half"] == "heldout"]
    rows = lambda ds, hs: (torch.stack([S[(i, x)][hs] for i in ds for x in ("none", "inj")]),
                           [x == "inj" for i in ds for x in ("none", "inj")])
    torch.manual_seed(0)
    aucs = {}
    for lay in CFG["candidate_layers"]:
        Xtr, ytr = rows(fit, lay + 1)
        Xte, yte = rows(held, lay + 1)
        aucs[lay] = round(auc(fit_predict(Xtr, ytr, Xte), yte), 3)
    best = max(aucs.values())
    layer = min(l for l in aucs if aucs[l] == best)  # tie -> earlier layer
    hs = layer + 1
    alld = [d["draw"] for d in draws]
    v = torch.stack([S[(i, "inj")][hs] for i in alld]).mean(0) - torch.stack([S[(i, "none")][hs] for i in alld]).mean(0)
    resid = float(torch.stack([S[(i, "none")][hs] for i in alld]).norm(dim=-1).mean())
    torch.save(v.clone(), OUT / "direction.pt")
    res = {"auc_heldout_by_layer": aucs, "steering_layer": layer, "auc_at_layer": best,
           "stop": best < CFG["auc_min"], "direction_norm": float(v.norm()),
           "mean_clean_resid_norm_answer_pos": resid, "direction_norm_ratio": float(v.norm()) / resid,
           "direction_file_sha256": sha(OUT / "direction.pt"),
           "direction_float32_sha256": hashlib.sha256(v.contiguous().numpy().astype("float32").tobytes()).hexdigest(),
           "alpha_injection": ctx.alpha, "n_draws": len(alld)}
    (OUT / "calibration.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    log(f"CALIBRATION: layer {layer} (held-out AUC {best}); |v| {res['direction_norm']:.3f} "
        f"({res['direction_norm_ratio']:.3f} x resid); stop={res['stop']}; {(time.time() - t0) / 60:.1f} min")
    return res


def load_direction():
    cal = jread(OUT / "calibration.json")
    frozen = jread(L.REPO_ROOT / "results" / "study14" / "calibration.json", cal)
    assert sha(OUT / "direction.pt") == frozen["direction_file_sha256"]
    v = torch.load(OUT / "direction.pt", weights_only=True).float()
    return cal["steering_layer"], v


# ----------------------------------------------------------------------------- steering

def conditions(ctx, v, question):
    """(name, unit-or-raw direction) list; every vector is scaled so its norm is k * |v|."""
    if question == "ctl":
        return [("inj", v)]
    nv = v.norm()
    return [("inj", v)] + [(f"rand{j}", r * nv) for j, r in enumerate(ctx.rand)] + [("pos", ctx.posdir * nv)]


@torch.inference_mode()
def steer_batch(model, P, cache, layer, vecs):
    """All steered variants of one prompt in one batch: the clean cache for tokens [0, n-1) is
    repeated B times and only the last token is run, with row b's vector added at the output of
    `layer`. Same computation as one L.run_suffix per vector (checked in sanity S7)."""
    n, B = len(P.ids), vecs.shape[0]
    c = copy.deepcopy(cache)
    c.crop(-(c.get_seq_length() - (n - 1)))
    c.batch_repeat_interleave(B)
    with L.inject(model, layer, vecs[:, None, :], [0]):
        out = model(input_ids=P.tensor()[:, n - 1:].repeat(B, 1), past_key_values=c, use_cache=True, logits_to_keep=1)
    return out.logits[:, -1].float()


def steer_trial(ctx, P, layer, v, question):
    m = ctx.model
    cl = L.clean_pass(m, P)
    out = {"k0": ctx.read(cl["logits"])}
    names = [(f"{name}|k{k}", k * base) for name, base in conditions(ctx, v, question) for k in CFG["doses"][1:]]
    lg = steer_batch(m, P, cl["cache"], layer, torch.stack([x for _, x in names]))
    for (key, _), row in zip(names, lg):
        out[key] = ctx.read(row)
    return out


def stage_sanity(ctx, layer, v):
    m, res = ctx.model, {}
    s = ctx.stim
    assert not set(s["calibration_pool"]) & set(s["test_pool"])
    assert all(set(d["sentences"]) <= set(s["test_pool"]) for d in s["test_draws"])
    assert all(set(d["sentences"]) <= set(s["calibration_pool"]) for d in s["cal_draws"])
    d = s["cal_draws"][0]
    P = ctx.prompt(d["sentences"], T.Q_DETECT)
    assert P.text.count(S6.BOS) == 1 and f"Today Date: {DATE}" in P.text and P.text.endswith("<|end_header_id|>\n\n")
    res["S1_n_tokens"] = len(P.ids)
    res["S2_yes_ids"], res["S2_no_ids"] = ctx.y, ctx.n
    n = len(P.ids)
    x0 = L.run_full(m, P)
    x1 = L.run_full(m, P, layer=layer, vec=torch.zeros_like(v), inj_abs=[n - 1])
    res["S3_zero_vector_max_abs_diff"] = float((x0 - x1).abs().max())
    assert res["S3_zero_vector_max_abs_diff"] == 0.0
    cl = L.clean_pass(m, P)
    res["S4_k0_clean_vs_full"] = float((cl["logits"] - x0).abs().max())
    for k in (1, 8):
        for name, base in (("inj", v), ("pos", ctx.posdir * v.norm())):
            fi = L.run_full(m, P, layer=layer, vec=k * base, inj_abs=[n - 1])
            si = L.run_suffix(m, P, n - 1, cl["cache"], layer, k * base, [n - 1])["logits"]
            res[f"S5_suffix_vs_full_{name}_k{k}"] = float((fi - si).abs().max())
    # S7: the batched last-token path equals one run_suffix per vector, for every condition and dose
    names = [(f"{nm}|k{k}", k * base) for nm, base in conditions(ctx, v, "det") for k in CFG["doses"][1:]]
    bl = steer_batch(m, P, cl["cache"], layer, torch.stack([x for _, x in names]))
    res["S7_batched_vs_single_max"] = max(
        float((bl[i] - L.run_suffix(m, P, n - 1, cl["cache"], layer, x, [n - 1])["logits"]).abs().max())
        for i, (_, x) in enumerate(names))
    res["S7_batched_same_top_token"] = all(
        int(bl[i].argmax()) == int(L.run_suffix(m, P, n - 1, cl["cache"], layer, x, [n - 1])["logits"].argmax())
        for i, (_, x) in enumerate(names))
    assert res["S7_batched_same_top_token"]
    assert all(val < CFG["tol_logit"] for kk, val in res.items() if kk.startswith(("S4", "S5", "S7")) and not isinstance(val, bool))
    res["S6_splits_disjoint"] = True
    (OUT / "sanity.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    log(f"SANITY PASSED: {json.dumps({k: res[k] for k in res if k.startswith(('S3', 'S4', 'S5', 'S7'))})}")


def stage_test(ctx, layer, v):
    st = L.ResultStore(OUT / "test.jsonl")
    t0, done = time.time(), 0
    for d in ctx.stim["test_draws"]:
        for q, text in (("det", T.Q_DETECT), ("ctl", ctx.ctrl_q(d))):
            key = f"{q}|d{d['draw']}"
            if st.has(key):
                continue
            P = ctx.prompt(d["sentences"], text)
            st.add({"key": key, "draw": d["draw"], "question": q, "control_q": d["control_q"],
                    "out": steer_trial(ctx, P, layer, v, q)})
        done += 1
        if done % 25 == 0:
            log(f"test draw {d['draw'] + 1}/{CFG['n_test']}; {(time.time() - t0) / 60:.1f} min")
    log(f"TEST DONE: {len(st.records)}/{2 * CFG['n_test']} records")


# ----------------------------------------------------------------------------- dry run

def stage_dryrun(ctx):
    """Timing only, on calibration-half draws 0-4 with a placeholder (random) direction; discarded."""
    ctx.load_injection()
    tc = []
    for d in ctx.stim["cal_draws"][:5]:
        t1 = time.time()
        cal_states(ctx, d)
        tc.append(time.time() - t1)
    v = L.random_unit_vectors(ctx.model.config.hidden_size, 1, seed=999)[0] * 3.0
    P = ctx.prompt(ctx.stim["cal_draws"][0]["sentences"], T.Q_DETECT)
    n, cl = len(P.ids), L.clean_pass(ctx.model, P)
    names = [k * base for _, base in conditions(ctx, v, "det") for k in CFG["doses"][1:]]
    bl = steer_batch(ctx.model, P, cl["cache"], 14, torch.stack(names))
    eq = max(float((bl[i] - L.run_suffix(ctx.model, P, n - 1, cl["cache"], 14, x, [n - 1])["logits"]).abs().max())
             for i, x in enumerate(names))
    log(f"DRY RUN: batched vs single last-token runs, max logit diff {eq:.2e}")
    tt = []
    for d in ctx.stim["cal_draws"][:5]:
        t1 = time.time()
        for q in ("det", "ctl"):
            steer_trial(ctx, ctx.prompt(d["sentences"], T.Q_DETECT if q == "det" else "Is London in Japan? Answer yes or no."), 14, v, q)
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
    log("copied results to results/study14")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["stimuli", "dryrun", "calibrate", "sanity", "test", "summary"])
    args = ap.parse_args()
    if args.stage == "stimuli":
        return stage_stimuli()
    if args.stage == "summary":
        return stage_summary()
    ctx = Ctx14()
    if args.stage == "dryrun":
        stage_dryrun(ctx)
    elif args.stage == "calibrate":
        stage_calibrate(ctx)
    else:
        layer, v = load_direction()
        stage_sanity(ctx, layer, v)
        if args.stage == "test":
            stage_test(ctx, layer, v)
    stage_summary()
    log("DONE")


if __name__ == "__main__":
    main()
