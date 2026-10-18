"""Study 6 — Tülu-3-8B with the exact Study 4 arm-A protocol. Plan: gates/study6_plan.md.

Reuses scripts/study4.py (span split, D3 copy fix) and scripts/study3.py (the authors' prompt
wording). Changes only what a different model needs: Tülu's own chat format (BOS prepended,
trailing <|end_of_text|> removed) and concept vectors recomputed from Tülu with the authors'
recipe. Two arms on 20 fresh draws:
  A  concept vector on the sentence words only  + copy-back (D3) + clean reader
  C  random unit vector of the same size        (injected only)

Usage:
  python scripts/study6.py --stage vectors       # recompute Tülu vectors, write hashes to stimuli
  python scripts/study6.py --stage llama_norms   # Llama-3.1-8B-Instruct activation size (plan §4)
  python scripts/study6.py --stage all|sanity|trials|summary
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

CFG = {
    "repo": "allenai/Llama-3.1-Tulu-3-8B",
    "llama_repo": "meta-llama/Llama-3.1-8B-Instruct",
    "layer": 2,
    "alpha": 5.0,
    "main_min_exact": 40,
    "main_inj_min": 0.50,
    "main_reader_max": 0.15,
    "c_margin": 20,
    "s7_min_draws": 19,
    "tol_cache_logit": 1e-3,
}
OUT = L.DATA_ROOT / "study6"
LOG = L.DATA_ROOT / "logs" / "study6.log"
REPO_RESULTS = L.REPO_ROOT / "results" / "study6"
STIM = L.REPO_ROOT / "stimuli" / "study6_stimuli.json"
BOS, EOT = "<|begin_of_text|>", "<|end_of_text|>"


def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    L.log(msg, LOG)


def tulu_format(tok, msgs, add_space):
    """The authors' format rule in Tülu's template: no generation prompt, drop the trailing
    end-of-turn token, trailing space before the digit; BOS prepended (Tülu's template omits it)."""
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    if s.endswith(EOT):
        s = s[: -len(EOT)]
    if add_space and not s.endswith(" "):
        s = s + " "
    return s if s.startswith(BOS) else BOS + s


class P6(S3.P3):
    """S3.P3 with Tülu's format; same modes and span rules."""

    def __init__(self, tok, spans, mode):
        orig = S3.their_format
        S3.their_format = tulu_format
        try:
            super().__init__(tok, spans, mode)
        finally:
            S3.their_format = orig


def prompts(tok, sentences, P=P6):
    spans = S3.marker_spans(sentences)
    p, cp = P(tok, spans, "locate"), P(tok, spans, "copy")
    pa = S4.arm_spans(p, spans, p.text.find(" ".join(spans)))
    ca = S4.arm_spans(cp, spans, cp.text.find(" ".join(spans)))
    return spans, p, cp, pa, ca


# ----------------------------------------------------------------------------- vectors

def tulu_user_turn(user_msg):
    return f"<|user|>\n{user_msg}\n<|assistant|>\n"


@torch.inference_mode()
def recompute_tulu_vectors(model, tok, src):
    """The authors' recipe (as in S3.recompute_their_vectors) at hidden_states[2], no BOS; the
    only change is Tülu's chat wrapper instead of the Llama header string."""
    def states(user_msg):
        enc = tok(tulu_user_turn(user_msg), return_tensors="pt", add_special_tokens=False)
        h = model(**enc, output_hidden_states=True, logits_to_keep=1).hidden_states[2][0].float()
        return h.mean(0), h[-1]

    out = {}
    base = [states(f"Tell me about {w}.") for w in src["baseline_words_first50"]]
    bavg = torch.stack([b[0] for b in base]).mean(0)
    blast = torch.stack([b[1] for b in base]).mean(0)
    for c in src["concepts"][:5]:
        a, l_ = states(f"Tell me about {c}.")
        out[(c, "avg")], out[(c, "last")] = a - bavg, l_ - blast
    for c in src["concepts"][5:]:
        pos, neg = src["complex_data"][c]
        P = [states(s) for s in pos]
        N = [states(s) for s in neg]
        out[(c, "avg")] = torch.stack([x[0] for x in P]).mean(0) - torch.stack([x[0] for x in N]).mean(0)
        out[(c, "last")] = torch.stack([x[1] for x in P]).mean(0) - torch.stack([x[1] for x in N]).mean(0)
    return out


def vec_dir(stim):
    return Path(stim["vector_dir"].replace("<INTROSPECTION_DATA>", str(L.DATA_ROOT)))


def stage_vectors():
    stim = json.loads(STIM.read_text(encoding="utf-8"))
    src = json.loads((L.REPO_ROOT / "stimuli" / "study3_stimuli.json").read_text(encoding="utf-8"))
    m, tok = L.load_model(CFG["repo"], CFG["revision"])
    want = tok.apply_chat_template([{"role": "user", "content": "X"}], tokenize=False, add_generation_prompt=True)
    assert want == tulu_user_turn("X"), repr(want)
    vecs = recompute_tulu_vectors(m, tok, src)
    d = vec_dir(stim)
    d.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for (c, t), v in vecs.items():
        assert torch.isfinite(v).all() and float(v.norm()) > 0, (c, t)
        f = d / f"{c}_2_{t}.pt"
        torch.save({"vector": v.clone(), "model_name": CFG["repo"], "concept_name": c, "layer": 2, "vec_type": t}, f)
        hashes[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    stim["vector_sha256"] = dict(sorted(hashes.items()))
    STIM.write_text(json.dumps(stim, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"VECTORS: recomputed {len(vecs)} Tulu vectors -> {d}; hashes written to {STIM.name}")


def load_tulu_vectors(stim):
    d, vecs = vec_dir(stim), {}
    for f, h in stim["vector_sha256"].items():
        assert hashlib.sha256((d / f).read_bytes()).hexdigest() == h, f
    for c in stim["concepts"]:
        for t in ("avg", "last"):
            x = torch.load(d / f"{c}_2_{t}.pt", weights_only=True)
            assert x["layer"] == 2 and x["vec_type"] == t and x["concept_name"] == c
            v = x["vector"].float().reshape(-1)
            vecs[(c, t)] = v / v.norm()
    return vecs


# ----------------------------------------------------------------------------- size report

@torch.inference_mode()
def sentence_norm(model, p, pa):
    """Mean L2 norm of the residual stream at the injection point (output of layers[2] =
    hidden_states[3]) over all sentence-word tokens of a clean localization prompt."""
    h = model(input_ids=p.tensor(), output_hidden_states=True, logits_to_keep=1).hidden_states[CFG["layer"] + 1][0]
    idx = [i for a in pa for i in a["sentence"]]
    return float(h[idx].float().norm(dim=-1).mean())


def stage_llama_norms():
    stim = json.loads(STIM.read_text(encoding="utf-8"))
    pins = json.loads((L.REPO_ROOT / "stimuli" / "model_pins.json").read_text())
    m, tok = L.load_model(CFG["llama_repo"], pins[CFG["llama_repo"]])
    norms = [sentence_norm(m, p, pa) for _, p, _, pa, _ in (prompts(tok, d["sentences"], S3.P3) for d in stim["draws"])]
    res = {"model": CFG["llama_repo"], "per_draw": [round(x, 4) for x in norms], "mean": sum(norms) / len(norms)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "llama_norms.json").write_text(json.dumps(res, indent=2))
    log(f"LLAMA NORMS: mean {res['mean']:.3f} over {len(norms)} draws")


# ----------------------------------------------------------------------------- context

class Ctx6:
    def __init__(self, stim_path=None):
        LOG.parent.mkdir(parents=True, exist_ok=True)
        OUT.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        self.model, self.tok = L.load_model(CFG["repo"], CFG["revision"])
        log(f"loaded {CFG['repo']}@{CFG['revision']} in {time.time() - t0:.0f}s")
        self.stim = json.loads(Path(stim_path or STIM).read_text(encoding="utf-8"))
        self.vecs = load_tulu_vectors(self.stim)
        g = torch.Generator().manual_seed(self.stim["random_vector_seed"])
        self.rand = []
        for _ in self.stim["draws"]:
            v = torch.randn(self.model.config.hidden_size, generator=g)
            self.rand.append(v / v.norm())
        p0 = P6(self.tok, S3.marker_spans(self.stim["draws"][0]["sentences"]), "locate")
        self.digits = L.digit_token_ids(self.tok, p0)

    def answer(self, logits):
        cand = logits[self.digits]
        return int(torch.argmax(cand)) + 1, [round(float(x), 4) for x in cand]

    def prompts(self, sentences):
        return prompts(self.tok, sentences)


def write_status(**kw):
    path = OUT / "status.json"
    st = json.loads(path.read_text()) if path.exists() else {}
    st.update(kw)
    st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(json.dumps(st, indent=2))
    return st


# ----------------------------------------------------------------------------- sanity

def stage_sanity(ctx):
    m, tok, res = ctx.model, ctx.tok, {}
    lay, a = CFG["layer"], CFG["alpha"]
    spans, p, cp, pa, ca = ctx.prompts(ctx.stim["draws"][0]["sentences"])
    # S1: Tülu format with the authors' wording; S2: digits; spans decode correctly
    assert p.text.startswith(BOS + "<|user|>\n") and p.text.endswith(S3.PREFIX3) and S3.INTRO3 in p.text
    assert p.text.count(BOS) == 1 and EOT not in p.text.split("<|assistant|>")[-1]
    assert p.ids[0] == tok.convert_tokens_to_ids(BOS)
    res["S1_n_tokens"] = len(p.ids)
    assert ctx.digits == [tok.encode(str(i), add_special_tokens=False)[-1] for i in range(1, 11)]
    res["S2_digit_ids"] = ctx.digits
    for k, idx in enumerate(p.user_tok):
        assert spans[k] in tok.decode([p.ids[i] for i in idx])
    # S8: label/sentence split
    for P, A in ((p, pa), (cp, ca)):
        for k in range(10):
            lab, sen = set(A[k]["label"]), set(A[k]["sentence"])
            assert lab and sen and not (lab & sen) and sorted(lab | sen) == P.user_tok[k], k
    res["S8_example_sentence"] = tok.decode([p.ids[i] for i in pa[2]["sentence"]])
    # S4: alpha 0 -> identical
    idx = pa[3]["sentence"]
    x0 = L.run_full(m, p)
    x1 = L.run_full(m, p, layer=lay, vec=torch.zeros(m.config.hidden_size), inj_abs=idx)
    res["S4_alpha0_max_abs_diff"] = float((x0 - x1).abs().max())
    assert res["S4_alpha0_max_abs_diff"] == 0.0
    # S5: cache == full pass
    cl = L.clean_pass(m, p)
    rv = L.random_unit_vectors(m.config.hidden_size, 1, seed=999)[0] * a
    fi = L.run_full(m, p, layer=lay, vec=rv, inj_abs=idx)
    ci = L.run_suffix(m, p, idx[0], cl["cache"], lay, rv, idx)["logits"]
    res["S5_maxdiff_sentence"] = float((ci[ctx.digits] - fi[ctx.digits]).abs().max())
    assert res["S5_maxdiff_sentence"] < CFG["tol_cache_logit"] and torch.argmax(ci[ctx.digits]) == torch.argmax(fi[ctx.digits])
    # S6'': vector files match the frozen hashes (checked in load_tulu_vectors), finite, non-zero
    res["S6pp_vectors_ok"] = all(torch.isfinite(v).all() for v in ctx.vecs.values())
    assert res["S6pp_vectors_ok"]
    # Size report (plan §4) and vector similarity to the authors' Llama vectors
    norms = [sentence_norm(m, pp, aa) for _, pp, _, aa, _ in (ctx.prompts(d["sentences"]) for d in ctx.stim["draws"])]
    res["tulu_norms_per_draw"] = [round(x, 4) for x in norms]
    res["tulu_norm_mean"] = sum(norms) / len(norms)
    S3.VEC_DIR = L.DATA_ROOT / "study3" / "their_vectors"
    if (S3.VEC_DIR / "Dust_2_avg.pt").exists():
        theirs = S3.load_their_vectors(ctx.stim["concepts"])
        res["cos_tulu_vs_their_llama"] = {f"{c}|{t}": round(float(ctx.vecs[(c, t)] @ theirs[(c, t)]), 4)
                                          for (c, t) in ctx.vecs}
    # S7: clean copy exact with the D3 fix
    ok, bad = 0, []
    for di, d in enumerate(ctx.stim["draws"]):
        spans_d, _, cp_d, _, _ = ctx.prompts(d["sentences"])
        cl_d = L.clean_pass(m, cp_d, want_final=True)
        perc, _ = S4.perceived_with_fix(m, tok, cp_d, {"final": cl_d["final"]}, 0)
        if all(perc[j] == spans_d[j] for j in range(10)):
            ok += 1
        else:
            bad.append({"draw": di, "diffs": [(j + 1, perc[j]) for j in range(10) if perc[j] != spans_d[j]]})
    res["S7_clean_copy_all_exact_draws"] = ok
    res["S7_failures"] = bad
    (OUT / "sanity.json").write_text(json.dumps(res, indent=2))
    if ok < CFG["s7_min_draws"]:
        write_status(sanity="FAIL (S7)")
        log(f"SANITY STOP: S7 only {ok}/20 clean copies exact (need {CFG['s7_min_draws']}); no trials run")
        return None
    log(f"SANITY PASSED: S4 0.0, S5 {res['S5_maxdiff_sentence']:.1e}, S6'' ok, S7 {ok}/20, S8 ok; "
        f"Tulu norm {res['tulu_norm_mean']:.3f}")
    write_status(sanity="PASS")
    return res


# ----------------------------------------------------------------------------- trials

def stage_trials(ctx):
    m, tok, lay, a = ctx.model, ctx.tok, CFG["layer"], CFG["alpha"]
    st = {arm: L.ResultStore(OUT / f"arm{arm}_injected.jsonl") for arm in "AC"}
    rdA = L.ResultStore(OUT / "armA_reader.jsonl")
    base = L.ResultStore(OUT / "no_injection.jsonl")
    draws = ctx.stim["draws"]
    total, t0, done = len(draws) * 10, time.time(), 0
    for di, d in enumerate(draws):
        c, t = d["concept"], d["vec_type"]
        keys = {pos: {arm: f"{arm}|{c}|{t}|pos{pos}" for arm in "AC"} for pos in range(1, 11)}
        if all(st[arm].has(keys[q][arm]) for q in keys for arm in "AC") and all(rdA.has("R" + keys[q]["A"]) for q in keys):
            continue
        spans, p, cp, pa, ca = ctx.prompts(d["sentences"])
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
                    t1 = time.time()
                    idx = pa[k]["sentence"]
                    r = L.run_suffix(m, p, idx[0], cl["cache"], lay, vec, idx, want_final=True)
                    pred, cand = ctx.answer(r["logits"])
                    with torch.inference_mode():
                        lc = torch.log_softmax(L.logits_at(m, cl["final"][idx]), -1)
                        li = torch.log_softmax(L.logits_at(m, r["final"][[i - idx[0] for i in idx]]), -1)
                        kl = float((lc.exp() * (lc - li)).sum(-1).mean())
                    st[arm].add(dict(meta, key=key, arm=arm, pos=pos, pred=pred, correct=pred == pos,
                                     cand_logits=cand, kl_target=kl, sec=round(time.time() - t1, 2)))
                if arm == "A" and not rdA.has("R" + key):
                    t1 = time.time()
                    idx = ca[k]["sentence"]
                    r = L.run_suffix(m, cp, idx[0], ccl["cache"], lay, vec, idx, want_final=True)
                    perc, tacc = S4.perceived_with_fix(m, tok, cp, r, idx[0])
                    exact = [perc[j] == spans[j] for j in range(10)]
                    rpred, rcand = ctx.answer(L.run_full(m, P6(tok, perc, "reader")))
                    rdA.add(dict(meta, key="R" + key, arm=arm, pos=pos, perceived=perc,
                                 exact_target=exact[k], n_exact=sum(exact),
                                 label_ok_target=perc[k].startswith(f"SENTENCE {pos}:"),
                                 sentence_ok_target=perc[k].endswith(d["sentences"][k]),
                                 tok_acc_target=round(tacc[k], 4), reader_pred=rpred,
                                 reader_correct=rpred == pos, reader_cand_logits=rcand,
                                 sec=round(time.time() - t1, 2)))
            done += 1
            eta = (time.time() - t0) / done * (total - di * 10 - k - 1)
            ia, ic = st["A"].records[keys[pos]["A"]], st["C"].records[keys[pos]["C"]]
            ra = rdA.records["R" + keys[pos]["A"]]
            log(f"pos {di * 10 + pos}/{total} {c}|{t}|pos{pos} A={ia['pred']}{'+' if ia['correct'] else '-'}"
                f"(rdr {ra['reader_pred']}, exact {ra['exact_target']}) C={ic['pred']}{'+' if ic['correct'] else '-'} "
                f"eta={eta / 3600:.2f}h")
    return verdict()


def verdict():
    I = {arm: list(L.ResultStore(OUT / f"arm{arm}_injected.jsonl").records.values()) for arm in "AC"}
    R = list(L.ResultStore(OUT / "armA_reader.jsonl").records.values())
    complete = all(len(I[x]) == 200 for x in "AC") and len(R) == 200
    k = {x: sum(r["correct"] for r in I[x]) for x in "AC"}
    rk = sum(r["reader_correct"] for r in R)
    imap = {(r["concept"], r["vec_type"], r["pos"]): r for r in I["A"]}
    ex = [r for r in R if r["exact_target"]]
    n_ex = len(ex)
    inj_ex = sum(imap[(r["concept"], r["vec_type"], r["pos"])]["correct"] for r in ex)
    rdr_ex = sum(r["reader_correct"] for r in ex)
    if n_ex < CFG["main_min_exact"]:
        main, concl = "NOT TESTABLE", "text corruption again"
    elif inj_ex >= CFG["main_inj_min"] * n_ex and rdr_ex <= CFG["main_reader_max"] * n_ex:
        main, concl = "PASS", "setup"
    elif inj_ex < CFG["main_inj_min"] * n_ex:
        main, concl = "FAIL", "model"
    else:
        main, concl = "FAIL", "unclear"
    v = {"complete": complete, "inj_k": k, "reader_k_A": rk, "A_exact_n": n_ex, "A_exact_inj_k": inj_ex,
         "A_exact_reader_k": rdr_ex, "main_test": main,
         "main_fail_reason": (None if main != "FAIL" else
                              "injected < 50%" if inj_ex < CFG["main_inj_min"] * n_ex else "reader > 15%"),
         "C_vs_A": ("within 10 points" if abs(k["C"] - k["A"]) <= CFG["c_margin"]
                    else "C lower" if k["C"] < k["A"] else "C higher"),
         "conclusion": concl if complete else None}
    write_status(result=v)
    log(f"RESULT: A {k['A']}/200 (reader {rk}), exact-copy {n_ex}: inj {inj_ex} rdr {rdr_ex} -> main {main}; "
        f"C {k['C']}/200; conclusion: {v['conclusion']}")
    return v


def stage_summary():
    REPO_RESULTS.mkdir(parents=True, exist_ok=True)
    for f in sorted(OUT.glob("*.json")) + sorted(OUT.glob("*.jsonl")):
        shutil.copy2(f, REPO_RESULTS / f.name)
    log(f"copied results to results/study6")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["vectors", "llama_norms", "all", "sanity", "trials", "summary"])
    args = ap.parse_args()
    CFG["revision"] = json.loads((L.REPO_ROOT / "stimuli" / "model_pins.json").read_text())[CFG["repo"]]
    if args.stage == "summary":
        stage_summary()
        return
    if args.stage == "vectors":
        stage_vectors()
        return
    if args.stage == "llama_norms":
        stage_llama_norms()
        stage_summary()
        return
    ctx = Ctx6()
    if args.stage in ("all", "sanity"):
        ok = stage_sanity(ctx)
        stage_summary()
        if ok is None:
            log("DONE (stopped at sanity)")
            return
    if args.stage in ("all", "trials"):
        stage_trials(ctx)
        stage_summary()
    log("DONE")


if __name__ == "__main__":
    main()
