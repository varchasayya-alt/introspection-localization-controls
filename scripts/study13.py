"""Study 13 — size curve in the Llama family (1B, 3B; 8B reused). Plan: gates/study13_plan.md.

Per model setting (1b_l1 primary, 3b_l2 primary, 1b_l2 secondary):
  vectors    recompute the concept vectors with the authors' recipe at hidden_states[layer]
  calibrate  alpha = 2.186 x mean residual norm at the injection point (first 5 draws, clean prompts)
  all        sanity -> localization (Study 4 arm A + copy/reader, arm C) -> yes/no + saved states
             (Study 11 design)

Usage:
  python scripts/study13.py --setting 1b_l1|3b_l2|1b_l2 --stage vectors|calibrate|all|summary
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import introspect_lib as L  # noqa: E402
import study3 as S3  # noqa: E402
import study4 as S4  # noqa: E402
import study10 as T  # noqa: E402
import study11 as E  # noqa: E402

SETTINGS = {"1b_l1": ("meta-llama/Llama-3.2-1B-Instruct", 1), "3b_l2": ("meta-llama/Llama-3.2-3B-Instruct", 2),
            "1b_l2": ("meta-llama/Llama-3.2-1B-Instruct", 2)}
CFG = {"ratio": 2.186, "calib_draws": 5, "s7_min_draws": 19, "tol_cache_logit": 1e-3, "main_min_exact": 40,
       "main_inj_min": 0.50, "main_reader_max": 0.15}
STIM = L.REPO_ROOT / "stimuli" / "study13_stimuli.json"
OUT = LOG = REPO_RESULTS = None


def configure(setting):
    global OUT, LOG, REPO_RESULTS
    OUT = L.DATA_ROOT / "study13" / setting
    LOG = L.DATA_ROOT / "logs" / f"study13_{setting}.log"
    REPO_RESULTS = L.REPO_ROOT / "results" / "study13" / setting


def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    L.log(msg, LOG)


def jread(p, default=None):
    return json.loads(p.read_text()) if p.exists() else default


def write_status(**kw):
    st = jread(OUT / "status.json", {})
    st.update(kw)
    st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (OUT / "status.json").write_text(json.dumps(st, indent=2))
    return st


# ----------------------------------------------------------------------------- vectors and strength

@torch.inference_mode()
def recompute_vectors(model, tok, src, hs):
    """S3.recompute_their_vectors (authors' recipe: hand-built Llama chat string, no BOS; 'avg' = mean
    over prompt tokens, 'last' = last token; mean differences) at hidden_states[hs]."""
    def states(user_msg):
        p = f"<|start_header_id|>user<|end_header_id|>{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        enc = tok(p, return_tensors="pt", add_special_tokens=False)
        h = model(**enc, output_hidden_states=True, logits_to_keep=1).hidden_states[hs][0].float()
        return h.mean(0), h[-1]

    out = {}
    base = [states(f"Tell me about {w}.") for w in src["baseline_words_first50"]]
    bavg, blast = torch.stack([b[0] for b in base]).mean(0), torch.stack([b[1] for b in base]).mean(0)
    for c in src["concepts"][:5]:
        a, l_ = states(f"Tell me about {c}.")
        out[(c, "avg")], out[(c, "last")] = a - bavg, l_ - blast
    for c in src["concepts"][5:]:
        pos, neg = src["complex_data"][c]
        P, N = [states(s) for s in pos], [states(s) for s in neg]
        out[(c, "avg")] = torch.stack([x[0] for x in P]).mean(0) - torch.stack([x[0] for x in N]).mean(0)
        out[(c, "last")] = torch.stack([x[1] for x in P]).mean(0) - torch.stack([x[1] for x in N]).mean(0)
    return out


class Ctx13:
    def __init__(self, setting):
        OUT.mkdir(parents=True, exist_ok=True)
        self.setting = setting
        self.repo, self.layer = SETTINGS[setting]
        rev = json.loads((L.REPO_ROOT / "stimuli" / "model_pins.json").read_text())[self.repo]
        t0 = time.time()
        self.model, self.tok = L.load_model(self.repo, rev)
        log(f"loaded {self.repo}@{rev} in {time.time() - t0:.0f}s; layer {self.layer}")
        self.stim = json.loads(STIM.read_text(encoding="utf-8"))
        p0 = S3.P3(self.tok, S3.marker_spans(self.stim["loc_draws"][0]["sentences"]), "locate")
        self.digits = L.digit_token_ids(self.tok, p0)
        self.y, self.n = T.yes_no_ids(self.tok)

    def answer(self, logits):
        cand = logits[self.digits]
        return int(torch.argmax(cand)) + 1, [round(float(x), 4) for x in cand]

    def loc_prompts(self, sentences):
        spans = S3.marker_spans(sentences)
        p, cp = S3.P3(self.tok, spans, "locate"), S3.P3(self.tok, spans, "copy")
        return spans, p, cp, S4.arm_spans(p, spans, p.text.find(" ".join(spans))), \
            S4.arm_spans(cp, spans, cp.text.find(" ".join(spans)))

    def load_vectors_and_alpha(self):
        meta = jread(OUT / "vectors.json")
        self.vecs = {}
        for f, h in meta["sha256"].items():
            b = (OUT / "vectors" / f).read_bytes()
            assert hashlib.sha256(b).hexdigest() == h, f
        for c in self.stim["concepts"]:
            for t in ("avg", "last"):
                v = torch.load(OUT / "vectors" / f"{c}_{self.layer}_{t}.pt", weights_only=True).float()
                self.vecs[(c, t)] = v / v.norm()
        self.alpha = jread(OUT / "calibration.json")["alpha"]
        g = torch.Generator().manual_seed(self.stim["random_vector_seed"])
        self.rand = []
        for _ in self.stim["loc_draws"]:
            v = torch.randn(self.model.config.hidden_size, generator=g)
            self.rand.append(v / v.norm())


def stage_vectors(ctx):
    if (OUT / "vectors.json").exists():
        return
    src = json.loads((L.REPO_ROOT / "stimuli" / "study3_stimuli.json").read_text(encoding="utf-8"))
    vecs = recompute_vectors(ctx.model, ctx.tok, src, ctx.layer)
    (OUT / "vectors").mkdir(exist_ok=True)
    hashes = {}
    for (c, t), v in vecs.items():
        assert torch.isfinite(v).all() and float(v.norm()) > 0, (c, t)
        f = OUT / "vectors" / f"{c}_{ctx.layer}_{t}.pt"
        torch.save(v.clone(), f)
        hashes[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    (OUT / "vectors.json").write_text(json.dumps({"layer": ctx.layer, "sha256": dict(sorted(hashes.items()))}, indent=2))
    log(f"VECTORS: {len(vecs)} recomputed at hidden_states[{ctx.layer}]")


@torch.inference_mode()
def stage_calibrate(ctx):
    if (OUT / "calibration.json").exists():
        return
    norms = []
    for d in ctx.stim["loc_draws"][: CFG["calib_draws"]]:
        _, p, _, pa, _ = ctx.loc_prompts(d["sentences"])
        h = ctx.model(input_ids=p.tensor(), output_hidden_states=True, logits_to_keep=1).hidden_states[ctx.layer + 1][0]
        idx = [i for a in pa for i in a["sentence"]]
        norms.append(float(h[idx].float().norm(dim=-1).mean()))
    mean = sum(norms) / len(norms)
    res = {"layer": ctx.layer, "norm_per_draw": norms, "mean_norm": mean, "ratio": CFG["ratio"],
           "alpha": CFG["ratio"] * mean}
    (OUT / "calibration.json").write_text(json.dumps(res, indent=2))
    log(f"CALIBRATION: layer {ctx.layer}, mean norm {mean:.3f}, alpha {res['alpha']:.3f}")


# ----------------------------------------------------------------------------- sanity

def stage_sanity(ctx):
    m, tok, res, lay, a = ctx.model, ctx.tok, {}, ctx.layer, ctx.alpha
    spans, p, cp, pa, ca = ctx.loc_prompts(ctx.stim["loc_draws"][0]["sentences"])
    assert p.text.startswith("<|begin_of_text|>") and p.text.endswith(S3.PREFIX3) and S3.INTRO3 in p.text
    assert ctx.digits == [tok.encode(str(i), add_special_tokens=False)[-1] for i in range(1, 11)]
    for P, A in ((p, pa), (cp, ca)):
        for k in range(10):
            lab, sen = set(A[k]["label"]), set(A[k]["sentence"])
            assert lab and sen and not (lab & sen) and sorted(lab | sen) == P.user_tok[k], k
    idx = pa[3]["sentence"]
    x0 = L.run_full(m, p)
    x1 = L.run_full(m, p, layer=lay, vec=torch.zeros(m.config.hidden_size), inj_abs=idx)
    res["S4_alpha0_max_abs_diff"] = float((x0 - x1).abs().max())
    assert res["S4_alpha0_max_abs_diff"] == 0.0
    cl = L.clean_pass(m, p)
    rv = L.random_unit_vectors(m.config.hidden_size, 1, seed=999)[0] * a
    fi = L.run_full(m, p, layer=lay, vec=rv, inj_abs=idx)
    ci = L.run_suffix(m, p, idx[0], cl["cache"], lay, rv, idx)["logits"]
    res["S5_maxdiff"] = float((ci[ctx.digits] - fi[ctx.digits]).abs().max())
    assert res["S5_maxdiff"] < CFG["tol_cache_logit"]
    # yes/no path: cache reuse equals a full pass
    E.CFG["layer"] = lay
    pd = T.P10(tok, S3.marker_spans(ctx.stim["yn_draws"][0]["sentences"]), T.Q_DETECT)
    jdx = pd.arm[3]["sentence"]
    _, _, cache = E.full_states(m, pd)
    lf, _ = E.full_states_injected(m, pd, rv, jdx)
    ls, _ = E.suffix_states(m, pd, jdx[0], cache, rv, jdx)
    res["S5_yesno_maxdiff"] = float((lf[ctx.y + ctx.n] - ls[ctx.y + ctx.n]).abs().max())
    assert res["S5_yesno_maxdiff"] < CFG["tol_cache_logit"]
    ok, bad = 0, []
    for di, d in enumerate(ctx.stim["loc_draws"]):
        spans_d, _, cp_d, _, _ = ctx.loc_prompts(d["sentences"])
        cl_d = L.clean_pass(m, cp_d, want_final=True)
        perc, _ = S4.perceived_with_fix(m, tok, cp_d, {"final": cl_d["final"]}, 0)
        if all(perc[j] == spans_d[j] for j in range(10)):
            ok += 1
        else:
            bad.append({"draw": di, "diffs": [(j + 1, perc[j]) for j in range(10) if perc[j] != spans_d[j]]})
    res["S7_clean_copy_all_exact_draws"], res["S7_failures"] = ok, bad
    res["localization_ok"] = ok >= CFG["s7_min_draws"]
    (OUT / "sanity.json").write_text(json.dumps(res, indent=2))
    log(f"SANITY: S4 0.0, S5 {res['S5_maxdiff']:.1e}/{res['S5_yesno_maxdiff']:.1e}, S7 {ok}/20 "
        f"-> localization {'runs' if res['localization_ok'] else 'STOPPED'}")
    write_status(sanity="PASS" if res["localization_ok"] else "PASS (yes/no only; S7 failed)")
    return res["localization_ok"]


# ----------------------------------------------------------------------------- localization

def stage_localization(ctx):
    m, tok, lay, a = ctx.model, ctx.tok, ctx.layer, ctx.alpha
    st = {arm: L.ResultStore(OUT / f"arm{arm}_injected.jsonl") for arm in "AC"}
    rdA = L.ResultStore(OUT / "armA_reader.jsonl")
    base = L.ResultStore(OUT / "no_injection.jsonl")
    t0 = time.time()
    for di, d in enumerate(ctx.stim["loc_draws"]):
        c, t = d["concept"], d["vec_type"]
        keys = {pos: {arm: f"{arm}|{c}|{t}|pos{pos}" for arm in "AC"} for pos in range(1, 11)}
        if all(st[x].has(keys[q][x]) for q in keys for x in "AC") and all(rdA.has("R" + keys[q]["A"]) for q in keys):
            continue
        spans, p, cp, pa, ca = ctx.loc_prompts(d["sentences"])
        meta = {"concept": c, "vec_type": t, "draw_index": di}
        cl = L.clean_pass(m, p, want_final=True)
        if not base.has(f"B|d{di}"):
            pred, cand = ctx.answer(cl["logits"])
            base.add(dict(meta, key=f"B|d{di}", pred=pred, cand_logits=cand))
        ccl = L.clean_pass(m, cp)
        for k in range(10):
            pos = k + 1
            for arm, vec in (("A", a * ctx.vecs[(c, t)]), ("C", a * ctx.rand[di])):
                key = keys[pos][arm]
                if not st[arm].has(key):
                    idx = pa[k]["sentence"]
                    r = L.run_suffix(m, p, idx[0], cl["cache"], lay, vec, idx)
                    pred, cand = ctx.answer(r["logits"])
                    st[arm].add(dict(meta, key=key, arm=arm, pos=pos, pred=pred, correct=pred == pos, cand_logits=cand))
                if arm == "A" and not rdA.has("R" + key):
                    idx = ca[k]["sentence"]
                    r = L.run_suffix(m, cp, idx[0], ccl["cache"], lay, vec, idx, want_final=True)
                    perc, tacc = S4.perceived_with_fix(m, tok, cp, r, idx[0])
                    exact = [perc[j] == spans[j] for j in range(10)]
                    rpred, rcand = ctx.answer(L.run_full(m, S3.P3(tok, perc, "reader")))
                    rdA.add(dict(meta, key="R" + key, arm=arm, pos=pos, perceived=perc, exact_target=exact[k],
                                 n_exact=sum(exact), tok_acc_target=round(tacc[k], 4), reader_pred=rpred,
                                 reader_correct=rpred == pos, reader_cand_logits=rcand))
        log(f"loc draw {di + 1}/20 done; {(time.time() - t0) / 60:.1f} min")
    I = {x: list(st[x].records.values()) for x in "AC"}
    R = list(rdA.records.values())
    imap = {(r["concept"], r["vec_type"], r["pos"]): r for r in I["A"]}
    ex = [r for r in R if r["exact_target"]]
    inj_ex = sum(imap[(r["concept"], r["vec_type"], r["pos"])]["correct"] for r in ex)
    rdr_ex = sum(r["reader_correct"] for r in ex)
    n_ex = len(ex)
    if n_ex < CFG["main_min_exact"]:
        main, why = "NOT TESTABLE", None
    elif inj_ex >= CFG["main_inj_min"] * n_ex and rdr_ex <= CFG["main_reader_max"] * n_ex:
        main, why = "PASS", None
    else:
        main, why = "FAIL", "injected < 50%" if inj_ex < CFG["main_inj_min"] * n_ex else "reader > 15%"
    v = {"A": sum(r["correct"] for r in I["A"]), "C": sum(r["correct"] for r in I["C"]),
         "reader": sum(r["reader_correct"] for r in R), "exact_n": n_ex, "exact_inj": inj_ex, "exact_reader": rdr_ex,
         "main_test": main, "main_fail_reason": why,
         "intact_score_pct": round(100 * inj_ex / n_ex, 1) if n_ex >= CFG["main_min_exact"] else None,
         "complete": len(I["A"]) == len(I["C"]) == len(R) == 200}
    write_status(localization=v)
    log(f"LOCALIZATION: A {v['A']}/200, reader {v['reader']}, exact {n_ex}: inj {inj_ex} rdr {rdr_ex} -> {main}; C {v['C']}/200")


# ----------------------------------------------------------------------------- yes/no + states (Study 11 design)

def stage_yesno(ctx):
    m, a = ctx.model, ctx.alpha
    E.CFG["layer"] = ctx.layer
    (OUT / "states").mkdir(exist_ok=True)
    st = L.ResultStore(OUT / "yesno.jsonl")
    t0 = time.time()
    for di, d in enumerate(ctx.stim["yn_draws"]):
        keys = [f"{q}|d{di}|{x}" for q in ("det", "ctl") for x in ("none", "inj")]
        if all(st.has(k) for k in keys):
            continue
        c, t, pos = d["concept"], d["vec_type"], d["position"]
        meta = {"draw_index": di, "split": "A" if di < 40 else "B", "concept": c, "vec_type": t,
                "control_q": d["control_q"], "pos": pos}
        vec = a * ctx.vecs[(c, t)]
        spans = S3.marker_spans(d["sentences"])
        ctrl = ctx.stim["control_questions"][d["control_q"]] + " Answer yes or no."
        for q, P in (("det", T.P10(ctx.tok, spans, T.Q_DETECT)), ("ctl", T.P10(ctx.tok, spans, ctrl))):
            lg, sts, cache = E.full_states(m, P)
            for x in ("none", "inj"):
                key = f"{q}|d{di}|{x}"
                if st.has(key):
                    continue
                if x == "inj":
                    idx = P.arm[pos - 1]["sentence"]
                    lg, sts = E.suffix_states(m, P, idx[0], cache, vec, idx)
                torch.save(sts.to(torch.float16).clone(), OUT / "states" / f"{q}_d{di}_{x}.pt")
                py, mass = T.p_yes(lg, ctx.y, ctx.n)
                st.add(dict(meta, key=key, question=q, injected=x == "inj", p_yes=round(py, 6), yes_no_mass=round(mass, 6)))
        if (di + 1) % 10 == 0:
            log(f"yes/no draw {di + 1}/80 done; {(time.time() - t0) / 60:.1f} min")
    write_status(yesno_complete=len(st.records) == 320)
    log(f"YES/NO DONE: {len(st.records)}/320 runs")


def stage_summary():
    REPO_RESULTS.mkdir(parents=True, exist_ok=True)
    for f in sorted(OUT.glob("*.json")) + sorted(OUT.glob("*.jsonl")):
        shutil.copy2(f, REPO_RESULTS / f.name)
    log(f"copied results to results/study13/{REPO_RESULTS.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True, choices=sorted(SETTINGS))
    ap.add_argument("--stage", required=True, choices=["vectors", "calibrate", "all", "summary"])
    args = ap.parse_args()
    configure(args.setting)
    if args.stage == "summary":
        stage_summary()
        return
    ctx = Ctx13(args.setting)
    stage_vectors(ctx)
    stage_calibrate(ctx)
    if args.stage == "all":
        ctx.load_vectors_and_alpha()
        if stage_sanity(ctx):
            stage_localization(ctx)
            stage_summary()
        stage_yesno(ctx)
    stage_summary()
    log("DONE")


if __name__ == "__main__":
    main()
