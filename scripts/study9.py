"""Study 9 — localization in the base model meta-llama/Llama-3.1-8B. Plan: gates/study9_10_plan.md.

Plain-text few-shot format (no chat template): 3 worked example lists where one sentence is
visibly odd (ALL CAPS) with its answer, then the test list, where the only difference is the
injection. Same injection protocol as Study 6 (layer 2, alpha 5, sentence words only); concept
vectors recomputed from the base model with the authors' recipe as plain text. Arms A (concept +
few-shot copy-back with the D3 fix + clean reader) and C (random). A positive control (visible ALL
CAPS, no injection) must pass before any injection trial.

Usage:
  python scripts/study9.py --stage vectors|all|sanity|trials|summary
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
import study6 as S6  # noqa: E402

CFG = {
    "repo": "meta-llama/Llama-3.1-8B",
    "layer": 2,
    "alpha": 5.0,
    "main_min_exact": 40,
    "main_inj_min": 0.50,
    "main_reader_max": 0.15,
    "c_margin": 20,
    "s7_min_draws": 19,
    "pc_draws": 5,
    "pc_min": 35,  # positive control: >= 35/50 correct
    "tol_cache_logit": 1e-3,
}
OUT = L.DATA_ROOT / "study9"
LOG = L.DATA_ROOT / "logs" / "study9.log"
REPO_RESULTS = L.REPO_ROOT / "results" / "study9"
STIM = L.REPO_ROOT / "stimuli" / "study9_stimuli.json"
BOS = "<|begin_of_text|>"
HEAD_LOC = ("Each list has ten sentences. One sentence is different from the others. "
            "The answer is the number of the different sentence.")
HEAD_COPY = "Copy each list exactly."


def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    L.log(msg, LOG)


def caps_spans(sentences, odd):
    """Marker spans with the words of sentence `odd` (1-based) in ALL CAPS; label unchanged."""
    return [f"SENTENCE {i + 1}: " + (s.upper() if i + 1 == odd else s) for i, s in enumerate(sentences)]


class FS(S3.P3):
    """Few-shot prompt. mode "locate": examples + test list + "Answer: SENTENCE "; "copy": copy
    examples + test list + its copy (teacher forcing); "reader": like locate, spans already formatted."""

    def __init__(self, tok, spans, mode, draw):
        self.spans = list(spans)
        block = " ".join(self.spans)
        if mode == "copy":
            ex = "".join(f"List: {' '.join(S3.marker_spans(e['sentences']))}\n"
                         f"Copy: {' '.join(S3.marker_spans(e['sentences']))}\n\n" for e in draw["examples"][:2])
            self.text = f"{BOS}{HEAD_COPY}\n\n{ex}List: {block}\nCopy: {block}"
        else:
            ex = "".join(f"List: {' '.join(caps_spans(e['sentences'], e['odd']))}\nAnswer: SENTENCE {e['odd']}\n\n"
                         for e in draw["examples"])
            self.text = f"{BOS}{HEAD_LOC}\n\n{ex}List: {block}\nAnswer: SENTENCE "
        enc = tok(self.text, add_special_tokens=False, return_offsets_mapping=True)
        self.ids, self.offsets = enc["input_ids"], enc["offset_mapping"]
        if mode == "reader":  # perceived text: no span indices needed
            return
        self.b0 = self.text.find(f"List: {block}\n") + len("List: ")
        self.user_tok = self._span_tokens(self.b0)
        self.arm = S4.arm_spans(self, self.spans, self.b0)
        if mode == "copy":
            b1 = self.text.rfind(block)
            assert b1 > self.b0 and self.text[b1:] == block
            self.resp_tok = self._span_tokens(b1)


# ----------------------------------------------------------------------------- vectors

def stage_vectors():
    stim = json.loads(STIM.read_text(encoding="utf-8"))
    src = json.loads((L.REPO_ROOT / "stimuli" / "study3_stimuli.json").read_text(encoding="utf-8"))
    m, tok = L.load_model(CFG["repo"], CFG["revision"])
    orig = S6.tulu_user_turn
    S6.tulu_user_turn = lambda user_msg: user_msg  # base model: plain text, no chat wrapper, no BOS
    try:
        vecs = S6.recompute_tulu_vectors(m, tok, src)
    finally:
        S6.tulu_user_turn = orig
    d = S6.vec_dir(stim)
    d.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for (c, t), v in vecs.items():
        assert torch.isfinite(v).all() and float(v.norm()) > 0, (c, t)
        f = d / f"{c}_2_{t}.pt"
        torch.save({"vector": v.clone(), "model_name": CFG["repo"], "concept_name": c, "layer": 2, "vec_type": t}, f)
        hashes[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    stim["vector_sha256"] = dict(sorted(hashes.items()))
    STIM.write_text(json.dumps(stim, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"VECTORS: recomputed {len(vecs)} base-model vectors; hashes written to {STIM.name}")


# ----------------------------------------------------------------------------- context

class Ctx9:
    def __init__(self, stim_path=None):
        OUT.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        self.model, self.tok = L.load_model(CFG["repo"], CFG["revision"])
        log(f"loaded {CFG['repo']}@{CFG['revision']} in {time.time() - t0:.0f}s")
        self.stim = json.loads(Path(stim_path or STIM).read_text(encoding="utf-8"))
        self.vecs = S6.load_tulu_vectors(self.stim)  # checks the SHA-256 hashes
        g = torch.Generator().manual_seed(self.stim["random_vector_seed"])
        self.rand = []
        for _ in self.stim["draws"]:
            v = torch.randn(self.model.config.hidden_size, generator=g)
            self.rand.append(v / v.norm())
        d0 = self.stim["draws"][0]
        self.digits = L.digit_token_ids(self.tok, FS(self.tok, S3.marker_spans(d0["sentences"]), "locate", d0))

    def answer(self, logits):
        cand = logits[self.digits]
        return int(torch.argmax(cand)) + 1, [round(float(x), 4) for x in cand]

    def prompts(self, d):
        spans = S3.marker_spans(d["sentences"])
        return spans, FS(self.tok, spans, "locate", d), FS(self.tok, spans, "copy", d)


def write_status(**kw):
    path = OUT / "status.json"
    st = json.loads(path.read_text()) if path.exists() else {}
    st.update(kw)
    st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(json.dumps(st, indent=2))
    return st


# ----------------------------------------------------------------------------- sanity + positive control

def stage_sanity(ctx):
    m, tok, res, lay, a = ctx.model, ctx.tok, {}, CFG["layer"], CFG["alpha"]
    d0 = ctx.stim["draws"][0]
    spans, p, cp = ctx.prompts(d0)
    assert p.text.startswith(BOS) and p.text.count(BOS) == 1 and p.text.endswith("Answer: SENTENCE ")
    assert p.ids[0] == tok.convert_tokens_to_ids(BOS)
    res["S1_n_tokens"] = [len(p.ids), len(cp.ids)]
    res["S2_digit_ids"] = ctx.digits
    for P in (p, cp):
        for k in range(10):
            lab, sen = set(P.arm[k]["label"]), set(P.arm[k]["sentence"])
            assert lab and sen and not (lab & sen) and sorted(lab | sen) == P.user_tok[k], k
            assert spans[k] in tok.decode([P.ids[i] for i in P.user_tok[k]])
    idx = p.arm[3]["sentence"]
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
    # reader KV reuse: the reader prompt shares the prefix up to the test list
    rp = FS(tok, spans, "reader", d0)
    assert rp.ids[: p.user_tok[0][0]] == p.ids[: p.user_tok[0][0]]
    rs = L.run_suffix(m, rp, p.user_tok[0][0], cl["cache"])["logits"]
    res["S5_reader_reuse_maxdiff"] = float((rs[ctx.digits] - L.run_full(m, rp)[ctx.digits]).abs().max())
    assert res["S5_reader_reuse_maxdiff"] < CFG["tol_cache_logit"]
    res["S6_vector_hashes_ok"] = True
    # Positive control: visible ALL CAPS target, no injection
    pc = L.ResultStore(OUT / "positive_control.jsonl")
    for di, d in enumerate(ctx.stim["draws"][: CFG["pc_draws"]]):
        for pos in range(1, 11):
            key = f"PC|d{di}|pos{pos}"
            if not pc.has(key):
                pred, cand = ctx.answer(L.run_full(m, FS(tok, caps_spans(d["sentences"], pos), "locate", d)))
                pc.add({"key": key, "draw_index": di, "pos": pos, "pred": pred, "correct": pred == pos, "cand_logits": cand})
    k_pc = sum(r["correct"] for r in pc.records.values())
    res["positive_control_correct"] = k_pc
    res["positive_control_pass"] = k_pc >= CFG["pc_min"]
    # S7: clean few-shot copy reproduces all 10 spans exactly
    ok, bad = 0, []
    for di, d in enumerate(ctx.stim["draws"]):
        spans_d, _, cp_d = ctx.prompts(d)
        cl_d = L.clean_pass(m, cp_d, want_final=True)
        perc, _ = S4.perceived_with_fix(m, tok, cp_d, {"final": cl_d["final"]}, 0)
        if all(perc[j] == spans_d[j] for j in range(10)):
            ok += 1
        else:
            bad.append({"draw": di, "diffs": [(j + 1, perc[j]) for j in range(10) if perc[j] != spans_d[j]]})
    res["S7_clean_copy_all_exact_draws"] = ok
    res["S7_failures"] = bad
    (OUT / "sanity.json").write_text(json.dumps(res, indent=2))
    if not res["positive_control_pass"]:
        write_status(sanity="FAIL (positive control)")
        log(f"STOP: positive control {k_pc}/50 (need {CFG['pc_min']}); no injection trials")
        return None
    if ok < CFG["s7_min_draws"]:
        write_status(sanity="FAIL (S7)")
        log(f"STOP: S7 only {ok}/20 clean copies exact; no injection trials")
        return None
    log(f"SANITY PASSED: positive control {k_pc}/50, S4 0.0, S5 {res['S5_maxdiff']:.1e}, S7 {ok}/20")
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
        spans, p, cp = ctx.prompts(d)
        meta = {"concept": c, "vec_type": t, "draw_index": di}
        cl = L.clean_pass(m, p, want_final=True)
        if not base.has(f"B|d{di}"):
            pred, cand = ctx.answer(cl["logits"])
            base.add(dict(meta, key=f"B|d{di}", pred=pred, cand_logits=cand))
        ccl = L.clean_pass(m, cp)
        r0 = p.user_tok[0][0]
        for k in range(10):
            pos = k + 1
            for arm, vec in (("A", a * ctx.vecs[(c, t)]), ("C", a * ctx.rand[di])):
                key = keys[pos][arm]
                if not st[arm].has(key):
                    t1 = time.time()
                    idx = p.arm[k]["sentence"]
                    r = L.run_suffix(m, p, idx[0], cl["cache"], lay, vec, idx)
                    pred, cand = ctx.answer(r["logits"])
                    st[arm].add(dict(meta, key=key, arm=arm, pos=pos, pred=pred, correct=pred == pos,
                                     cand_logits=cand, sec=round(time.time() - t1, 2)))
                if arm == "A" and not rdA.has("R" + key):
                    t1 = time.time()
                    idx = cp.arm[k]["sentence"]
                    r = L.run_suffix(m, cp, idx[0], ccl["cache"], lay, vec, idx, want_final=True)
                    perc, tacc = S4.perceived_with_fix(m, tok, cp, r, idx[0])
                    exact = [perc[j] == spans[j] for j in range(10)]
                    rp = FS(tok, perc, "reader", d)
                    if rp.ids[:r0] == p.ids[:r0]:
                        rlog = L.run_suffix(m, rp, r0, cl["cache"])["logits"]
                    else:
                        rlog = L.run_full(m, rp)
                    rpred, rcand = ctx.answer(rlog)
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
    S6.OUT = OUT  # reuse Study 6's verdict (same rule) on Study 9's files
    S6.LOG = LOG
    return S6.verdict()


def stage_summary():
    REPO_RESULTS.mkdir(parents=True, exist_ok=True)
    for f in sorted(OUT.glob("*.json")) + sorted(OUT.glob("*.jsonl")):
        shutil.copy2(f, REPO_RESULTS / f.name)
    log("copied results to results/study9")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["vectors", "all", "sanity", "trials", "summary"])
    args = ap.parse_args()
    CFG["revision"] = json.loads((L.REPO_ROOT / "stimuli" / "model_pins.json").read_text())[CFG["repo"]]
    if args.stage == "summary":
        stage_summary()
        return
    if args.stage == "vectors":
        stage_vectors()
        return
    ctx = Ctx9()
    if args.stage in ("all", "sanity"):
        ok = stage_sanity(ctx)
        stage_summary()
        if ok is None:
            log("DONE (stopped before injection trials)")
            return
    if args.stage in ("all", "trials"):
        stage_trials(ctx)
        stage_summary()
    log("DONE")


if __name__ == "__main__":
    main()
