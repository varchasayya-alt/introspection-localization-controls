"""Study 4 — noticing vs label-digit highlighting. Plan: gates/study4_plan.md.

Same model, prompt and vectors as Study 3 (reuses scripts/study3.py). Three arms on the same
20 fresh sentence draws:
  A  inject on the sentence words only (not "SENTENCE N:")  + copy-back + clean reader
  B  inject on the "SENTENCE N:" label only                  + copy-back + clean reader
  C  as A, with a random unit vector of the same size        (injected only)
Copy-back uses the D3 fix: the first token of each copied span is taken from the real text.

Usage:
  python scripts/study4.py --stage all|sanity|trials|summary
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

CFG = {
    "repo": "meta-llama/Llama-3.1-8B-Instruct",
    "layer": 2,
    "alpha": 5.0,
    "main_min_exact": 40,      # main test needs >= 40 exact-copy trials in arm A
    "main_inj_min": 0.50,      # injected >= 50% among them
    "main_reader_max": 0.15,   # reader <= 15% among them
    "label_alt_min": 100,      # arm B injected >= 100/200 -> label highlighting is a real alternative
    "loophole_margin": 20,     # secondary: reader >= injected - 20 (10 points of 200)
    "c_margin": 20,            # arm C within 20 trials of arm A -> not concept-specific
    "s7_min_draws": 19,        # clean copy exact for all 10 spans in >= 19/20 draws
    "tol_cache_logit": 1e-3,
}
OUT = L.DATA_ROOT / "study4"
LOG = L.DATA_ROOT / "logs" / "study4.log"
REPO_RESULTS = L.REPO_ROOT / "results" / "study4"
ARMS_WITH_READER = ("A", "B")


def log(msg):
    L.log(msg, LOG)


def arm_spans(p, spans, block_start):
    """For each "SENTENCE N: <sentence>" span: token indices of the label part ("SENTENCE N:")
    and of the sentence part (the words). Same overlap rule as their code."""
    out, pos = [], block_start
    for i, s in enumerate(spans):
        cs, ce = pos, pos + len(s)
        lab_end = cs + len(f"SENTENCE {i + 1}:")
        sent_start = cs + len(f"SENTENCE {i + 1}: ")
        ov = lambda a0, a1: [j for j, (a, b) in enumerate(p.offsets) if b > a and a < a1 and b > a0]
        out.append({"label": ov(cs, lab_end), "sentence": ov(sent_start, ce)})
        pos = ce + 1
    return out


def perceived_with_fix(m, tok, cp, r, start):
    """D3 fix: the first token of each copied span comes from the real text; all other tokens
    are the model's top prediction (teacher forcing)."""
    out, tacc = [], []
    with torch.inference_mode():
        for toks in cp.resp_tok:
            rest = toks[1:]
            rows = r["final"][[q - 1 - start for q in rest]]
            pid = torch.argmax(L.logits_at(m, rows), -1).tolist()
            out.append(tok.decode([cp.ids[toks[0]]] + pid).strip())
            tacc.append(sum(x == cp.ids[q] for x, q in zip(pid, rest)) / max(len(rest), 1))
    return out, tacc


class Ctx4:
    def __init__(self, stim_path=None):
        LOG.parent.mkdir(parents=True, exist_ok=True)
        OUT.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        self.model, self.tok = L.load_model(CFG["repo"], CFG["revision"])
        log(f"loaded {CFG['repo']}@{CFG['revision']} in {time.time() - t0:.0f}s")
        stim_path = stim_path or (L.REPO_ROOT / "stimuli" / "study4_stimuli.json")
        self.stim = json.loads(Path(stim_path).read_text(encoding="utf-8"))
        self.vec_dir = Path(self.stim["vector_dir"].replace("<INTROSPECTION_DATA>", str(L.DATA_ROOT)))
        S3.VEC_DIR = self.vec_dir
        self.vecs = S3.load_their_vectors(self.stim["concepts"])
        g = torch.Generator().manual_seed(self.stim["random_vector_seed"])
        dim = self.model.config.hidden_size
        self.rand = []
        for _ in self.stim["draws"]:
            v = torch.randn(dim, generator=g)
            self.rand.append(v / v.norm())
        p0 = S3.P3(self.tok, S3.marker_spans(self.stim["draws"][0]["sentences"]), "locate")
        self.digits = L.digit_token_ids(self.tok, p0)

    def answer(self, logits):
        cand = logits[self.digits]
        return int(torch.argmax(cand)) + 1, [round(float(x), 4) for x in cand]

    def prompts(self, sentences):
        spans = S3.marker_spans(sentences)
        p = S3.P3(self.tok, spans, "locate")
        cp = S3.P3(self.tok, spans, "copy")
        pa = arm_spans(p, spans, p.text.find(" ".join(spans)))
        ca = arm_spans(cp, spans, cp.text.find(" ".join(spans)))
        return spans, p, cp, pa, ca


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
    # S1-S3 as in Study 3
    assert p.text.startswith("<|begin_of_text|>") and p.text.endswith(S3.PREFIX3) and S3.INTRO3 in p.text
    res["S1_n_tokens"] = len(p.ids)
    assert ctx.digits == [tok.encode(str(i), add_special_tokens=False)[-1] for i in range(1, 11)]
    res["S2_digit_ids"] = ctx.digits
    for k, idx in enumerate(p.user_tok):
        assert spans[k] in tok.decode([p.ids[i] for i in idx])
    # S8: label and sentence parts are disjoint and together give the full Study 3 span
    for P, A in ((p, pa), (cp, ca)):
        for k in range(10):
            lab, sen = set(A[k]["label"]), set(A[k]["sentence"])
            assert lab and sen and not (lab & sen) and sorted(lab | sen) == P.user_tok[k], k
    res["S8_example_label"] = tok.decode([p.ids[i] for i in pa[2]["label"]])
    res["S8_example_sentence"] = tok.decode([p.ids[i] for i in pa[2]["sentence"]])
    # S4: alpha 0 -> identical
    idx = pa[3]["sentence"]
    x0 = L.run_full(m, p)
    x1 = L.run_full(m, p, layer=lay, vec=torch.zeros(m.config.hidden_size), inj_abs=idx)
    res["S4_alpha0_max_abs_diff"] = float((x0 - x1).abs().max())
    assert res["S4_alpha0_max_abs_diff"] == 0.0
    # S5: cache == full pass, for both arm span types (random vector; no concept data)
    cl = L.clean_pass(m, p)
    rv = L.random_unit_vectors(m.config.hidden_size, 1, seed=999)[0] * a
    for part in ("sentence", "label"):
        ix = pa[3][part]
        fi = L.run_full(m, p, layer=lay, vec=rv, inj_abs=ix)
        ci = L.run_suffix(m, p, ix[0], cl["cache"], lay, rv, ix)["logits"]
        d = float((ci[ctx.digits] - fi[ctx.digits]).abs().max())
        res[f"S5_maxdiff_{part}"] = d
        assert d < CFG["tol_cache_logit"] and torch.argmax(ci[ctx.digits]) == torch.argmax(fi[ctx.digits])
    # S6': vector files identical to Study 3's
    for f, h in ctx.stim["vector_sha256"].items():
        assert hashlib.sha256((ctx.vec_dir / f).read_bytes()).hexdigest() == h, f
    res["S6prime_vector_hashes_match"] = True
    # S7: with the D3 fix, a clean copy (no injection) reproduces all 10 spans exactly
    ok, bad = 0, []
    for di, d in enumerate(ctx.stim["draws"]):
        spans_d, _, cp_d, _, _ = ctx.prompts(d["sentences"])
        cl_d = L.clean_pass(m, cp_d, want_final=True)
        perc, _ = perceived_with_fix(m, tok, cp_d, {"final": cl_d["final"]}, 0)
        if all(perc[j] == spans_d[j] for j in range(10)):
            ok += 1
        else:
            bad.append({"draw": di, "diffs": [(j + 1, perc[j]) for j in range(10) if perc[j] != spans_d[j]]})
    res["S7_clean_copy_all_exact_draws"] = ok
    res["S7_failures"] = bad
    (OUT / "sanity.json").write_text(json.dumps(res, indent=2))
    assert ok >= CFG["s7_min_draws"], f"S7: only {ok}/20 clean copies exact"
    log(f"SANITY PASSED: S4 0.0, S5 {res['S5_maxdiff_sentence']:.1e}/{res['S5_maxdiff_label']:.1e}, "
        f"S6' hashes ok, S7 {ok}/20 clean copies exact, S8 spans ok")
    write_status(sanity="PASS")
    return res


# ----------------------------------------------------------------------------- trials

def stage_trials(ctx):
    m, tok, lay, a = ctx.model, ctx.tok, CFG["layer"], CFG["alpha"]
    st = {arm: L.ResultStore(OUT / f"arm{arm}_injected.jsonl") for arm in "ABC"}
    rd = {arm: L.ResultStore(OUT / f"arm{arm}_reader.jsonl") for arm in ARMS_WITH_READER}
    base = L.ResultStore(OUT / "no_injection.jsonl")
    draws = ctx.stim["draws"]
    total, t0, done = len(draws) * 10, time.time(), 0
    for di, d in enumerate(draws):
        c, t = d["concept"], d["vec_type"]
        keys = {pos: {arm: f"{arm}|{c}|{t}|pos{pos}" for arm in "ABC"} for pos in range(1, 11)}
        if all(st[arm].has(keys[p_][arm]) for p_ in keys for arm in "ABC") and \
                all(rd[arm].has("R" + keys[p_][arm]) for p_ in keys for arm in ARMS_WITH_READER):
            continue
        spans, p, cp, pa, ca = ctx.prompts(d["sentences"])
        meta = {"concept": c, "vec_type": t, "draw_index": di}
        cl = L.clean_pass(m, p, want_final=True)
        if not base.has(f"B|d{di}"):
            pred, cand = ctx.answer(cl["logits"])
            base.add(dict(meta, key=f"B|d{di}", pred=pred, cand_logits=cand))
        ccl = L.clean_pass(m, cp)
        cvec = a * ctx.vecs[(c, t)]
        rvec = a * ctx.rand[di]
        for k in range(10):
            pos = k + 1
            for arm, part, vec in (("A", "sentence", cvec), ("B", "label", cvec), ("C", "sentence", rvec)):
                key = keys[pos][arm]
                if not st[arm].has(key):
                    t1 = time.time()
                    idx = pa[k][part]
                    r = L.run_suffix(m, p, idx[0], cl["cache"], lay, vec, idx, want_final=True)
                    pred, cand = ctx.answer(r["logits"])
                    with torch.inference_mode():
                        lc = torch.log_softmax(L.logits_at(m, cl["final"][idx]), -1)
                        li = torch.log_softmax(L.logits_at(m, r["final"][[i - idx[0] for i in idx]]), -1)
                        kl = float((lc.exp() * (lc - li)).sum(-1).mean())
                    st[arm].add(dict(meta, key=key, arm=arm, pos=pos, pred=pred, correct=pred == pos,
                                     cand_logits=cand, kl_target=kl, sec=round(time.time() - t1, 2)))
                if arm in ARMS_WITH_READER and not rd[arm].has("R" + key):
                    t1 = time.time()
                    idx = ca[k][part]
                    r = L.run_suffix(m, cp, idx[0], ccl["cache"], lay, vec, idx, want_final=True)
                    perc, tacc = perceived_with_fix(m, tok, cp, r, idx[0])
                    exact = [perc[j] == spans[j] for j in range(10)]
                    rpred, rcand = ctx.answer(L.run_full(m, S3.P3(tok, perc, "reader")))
                    rd[arm].add(dict(meta, key="R" + key, arm=arm, pos=pos, perceived=perc,
                                     exact_target=exact[k], n_exact=sum(exact),
                                     label_ok_target=perc[k].startswith(f"SENTENCE {pos}:"),
                                     sentence_ok_target=perc[k].endswith(d["sentences"][k]),
                                     tok_acc_target=round(tacc[k], 4), reader_pred=rpred,
                                     reader_correct=rpred == pos, reader_cand_logits=rcand,
                                     sec=round(time.time() - t1, 2)))
            done += 1
            eta = (time.time() - t0) / done * (total - di * 10 - k - 1)
            ia, ib, ic = (st[x].records[keys[pos][x]] for x in "ABC")
            ra, rb = rd["A"].records["R" + keys[pos]["A"]], rd["B"].records["R" + keys[pos]["B"]]
            log(f"pos {di * 10 + pos}/{total} {c}|{t}|pos{pos} A={ia['pred']}{'+' if ia['correct'] else '-'}"
                f"(rdr {ra['reader_pred']}, exact {ra['exact_target']}) B={ib['pred']}{'+' if ib['correct'] else '-'}"
                f"(rdr {rb['reader_pred']}) C={ic['pred']}{'+' if ic['correct'] else '-'} eta={eta / 3600:.2f}h")
    return verdict()


def verdict():
    I = {arm: list(L.ResultStore(OUT / f"arm{arm}_injected.jsonl").records.values()) for arm in "ABC"}
    R = {arm: list(L.ResultStore(OUT / f"arm{arm}_reader.jsonl").records.values()) for arm in ARMS_WITH_READER}
    complete = all(len(I[x]) == 200 for x in "ABC") and all(len(R[x]) == 200 for x in ARMS_WITH_READER)
    k = {x: sum(r["correct"] for r in I[x]) for x in "ABC"}
    rk = {x: sum(r["reader_correct"] for r in R[x]) for x in ARMS_WITH_READER}
    imap = {(r["concept"], r["vec_type"], r["pos"]): r for r in I["A"]}
    ex = [r for r in R["A"] if r["exact_target"]]
    n_ex = len(ex)
    inj_ex = sum(imap[(r["concept"], r["vec_type"], r["pos"])]["correct"] for r in ex)
    rdr_ex = sum(r["reader_correct"] for r in ex)
    if n_ex < CFG["main_min_exact"]:
        main = "NOT TESTABLE"
    elif inj_ex >= CFG["main_inj_min"] * n_ex and rdr_ex <= CFG["main_reader_max"] * n_ex:
        main = "PASS"
    else:
        main = "FAIL"
    label_alt = k["B"] >= CFG["label_alt_min"]
    a_pass = main == "PASS"
    conclusion = ("noticing (A passes, B < 50%)" if a_pass and not label_alt else
                  "both routes work" if a_pass and label_alt else
                  "label highlighting explains Study 3" if label_alt else "inconclusive")
    v = {"complete": complete, "inj_k": k, "reader_k": rk,
         "A_exact_n": n_ex, "A_exact_inj_k": inj_ex, "A_exact_reader_k": rdr_ex, "main_test": main,
         "B_label_alternative": label_alt,
         "A_loophole_secondary": "confirmed" if rk["A"] >= k["A"] - CFG["loophole_margin"] else "not confirmed",
         "C_vs_A": ("within 10 points (not concept-specific)" if abs(k["C"] - k["A"]) <= CFG["c_margin"]
                    else "C lower" if k["C"] < k["A"] else "C higher"),
         "conclusion": conclusion if complete else None}
    write_status(result=v)
    log(f"RESULT: A {k['A']}/200 (reader {rk['A']}), exact-copy {n_ex}: inj {inj_ex} rdr {rdr_ex} -> main {main}; "
        f"B {k['B']}/200 (reader {rk['B']}) -> label alt {label_alt}; C {k['C']}/200; conclusion: {v['conclusion']}")
    return v


def stage_summary():
    REPO_RESULTS.mkdir(parents=True, exist_ok=True)
    for f in sorted(OUT.glob("*.json")) + sorted(OUT.glob("*.jsonl")):
        shutil.copy2(f, REPO_RESULTS / f.name)
    log(f"copied results to {REPO_RESULTS}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["all", "sanity", "trials", "summary"])
    args = ap.parse_args()
    CFG["revision"] = json.loads((L.REPO_ROOT / "stimuli" / "model_pins.json").read_text())[CFG["repo"]]
    if args.stage == "summary":
        stage_summary()
        return
    ctx = Ctx4()
    if args.stage in ("all", "sanity"):
        stage_sanity(ctx)
        stage_summary()
    if args.stage in ("all", "trials"):
        stage_trials(ctx)
        stage_summary()
    log("DONE")


if __name__ == "__main__":
    main()
