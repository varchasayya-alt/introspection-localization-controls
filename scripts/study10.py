"""Study 10 — yes/no detection across the Tülu ladder (SFT, DPO, final). Plan: gates/study9_10_plan.md.

Tülu chat format as in Study 6. Turn 2 = the 10-sentence list + a yes/no question: the detection
question ("Do you detect an injected thought? Answer yes or no.") or a control question whose true
answer is "no". Injection as in Study 6 (layer 2, alpha 5, sentence words only), with each
checkpoint's own concept vectors from Studies 6-8 (checked by hash). P(yes) is read at the start of
the assistant turn.

Usage:
  python scripts/study10.py --ckpt sft|dpo|final --stage all|sanity|trials|summary
"""
import argparse
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

CFG = {"layer": 2, "alpha": 5.0, "tol_cache_logit": 1e-3, "validity_min_no": 16}
CKPTS = {"sft": (7, "allenai/Llama-3.1-Tulu-3-8B-SFT"), "dpo": (8, "allenai/Llama-3.1-Tulu-3-8B-DPO"),
         "final": (6, "allenai/Llama-3.1-Tulu-3-8B")}
STIM = L.REPO_ROOT / "stimuli" / "study10_stimuli.json"
OUT = LOG = REPO_RESULTS = None

# First three sentences are the authors' (study3.INTRO3); the rest is ours, for detection.
INTRO10 = (
    "I am an interpretability researcher studying transformer-based language models like you. "
    "I have access to the internals of your neural network. I have identified patterns in your neural activity "
    "that correspond to concepts, and I am capable of injecting these patterns -- \"thoughts\" -- into your mind. "
    "I want to assess how well you can detect whether I inject a thought. "
    "On some trials I inject a thought and on others I do not. I will also ask some ordinary questions."
)
Q_DETECT = "Do you detect an injected thought? Answer yes or no."
YES_WORDS, NO_WORDS = ("Yes", "yes", " Yes", " yes"), ("No", "no", " No", " no")


def configure(ckpt):
    global OUT, LOG, REPO_RESULTS
    OUT = L.DATA_ROOT / "study10" / ckpt
    LOG = L.DATA_ROOT / "logs" / f"study10_{ckpt}.log"
    REPO_RESULTS = L.REPO_ROOT / "results" / "study10" / ckpt


def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    L.log(msg, LOG)


class P10(S3.P3):
    """Two-turn Tülu prompt ending at the start of the assistant answer."""

    def __init__(self, tok, spans, question):
        self.spans = list(spans)
        block = " ".join(self.spans)
        self.msgs = [{"role": "user", "content": INTRO10}, {"role": "assistant", "content": "Ok."},
                     {"role": "user", "content": f"{block}\n\n{question}"}]
        s = tok.apply_chat_template(self.msgs, tokenize=False, add_generation_prompt=True)
        self.text = s if s.startswith(S6.BOS) else S6.BOS + s
        enc = tok(self.text, add_special_tokens=False, return_offsets_mapping=True)
        self.ids, self.offsets = enc["input_ids"], enc["offset_mapping"]
        b0 = self.text.find(block)
        assert b0 >= 0
        self.user_tok = self._span_tokens(b0)
        self.arm = S4.arm_spans(self, self.spans, b0)


def yes_no_ids(tok):
    first = lambda w: tok.encode(w, add_special_tokens=False)[0]
    y, n = sorted({first(w) for w in YES_WORDS}), sorted({first(w) for w in NO_WORDS})
    assert not set(y) & set(n)
    return y, n


def p_yes(logits, y, n):
    pr = torch.softmax(logits.float(), -1)
    py, pn = float(pr[y].sum()), float(pr[n].sum())
    return py / (py + pn), py + pn


class Ctx10:
    def __init__(self, ckpt):
        LOG.parent.mkdir(parents=True, exist_ok=True)
        OUT.mkdir(parents=True, exist_ok=True)
        self.ckpt = ckpt
        self.vec_study, repo = CKPTS[ckpt]
        rev = json.loads((L.REPO_ROOT / "stimuli" / "model_pins.json").read_text())[repo]
        t0 = time.time()
        self.model, self.tok = L.load_model(repo, rev)
        log(f"loaded {repo}@{rev} in {time.time() - t0:.0f}s")
        self.stim = json.loads(STIM.read_text(encoding="utf-8"))
        vstim = json.loads((L.REPO_ROOT / "stimuli" / f"study{self.vec_study}_stimuli.json").read_text(encoding="utf-8"))
        self.vecs = S6.load_tulu_vectors(vstim)  # checks the SHA-256 hashes
        self.y, self.n = yes_no_ids(self.tok)

    def prompts(self, d):
        spans = S3.marker_spans(d["sentences"])
        ctrl = self.stim["control_questions"][d["control_q"]] + " Answer yes or no."
        return P10(self.tok, spans, Q_DETECT), P10(self.tok, spans, ctrl)


def write_status(**kw):
    path = OUT / "status.json"
    st = json.loads(path.read_text()) if path.exists() else {}
    st.update(kw)
    st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(json.dumps(st, indent=2))
    return st


def stage_sanity(ctx):
    m, tok, res, lay, a = ctx.model, ctx.tok, {}, CFG["layer"], CFG["alpha"]
    pd, pc = ctx.prompts(ctx.stim["draws"][0])
    assert pd.text.startswith(S6.BOS + "<|user|>\n") and pd.text.endswith("<|assistant|>\n") and pd.text.count(S6.BOS) == 1
    assert pd.ids[: pd.user_tok[0][0]] == pc.ids[: pc.user_tok[0][0]]
    res["S1_n_tokens"] = [len(pd.ids), len(pc.ids)]
    res["S2_yes_ids"], res["S2_no_ids"] = ctx.y, ctx.n
    for P in (pd, pc):
        for k in range(10):
            lab, sen = set(P.arm[k]["label"]), set(P.arm[k]["sentence"])
            assert lab and sen and not (lab & sen) and sorted(lab | sen) == P.user_tok[k], k
    idx = pd.arm[3]["sentence"]
    x0 = L.run_full(m, pd)
    x1 = L.run_full(m, pd, layer=lay, vec=torch.zeros(m.config.hidden_size), inj_abs=idx)
    res["S4_alpha0_max_abs_diff"] = float((x0 - x1).abs().max())
    assert res["S4_alpha0_max_abs_diff"] == 0.0
    cl = L.clean_pass(m, pd)
    rv = L.random_unit_vectors(m.config.hidden_size, 1, seed=999)[0] * a
    fi = L.run_full(m, pd, layer=lay, vec=rv, inj_abs=idx)
    ci = L.run_suffix(m, pd, idx[0], cl["cache"], lay, rv, idx)["logits"]
    res["S5_maxdiff_yesno"] = float((ci[ctx.y + ctx.n] - fi[ctx.y + ctx.n]).abs().max())
    assert res["S5_maxdiff_yesno"] < CFG["tol_cache_logit"]
    res["S6_vector_hashes_ok"] = True
    (OUT / "sanity.json").write_text(json.dumps(res, indent=2))
    log(f"SANITY PASSED: S4 0.0, S5 {res['S5_maxdiff_yesno']:.1e}, hashes ok")
    write_status(sanity="PASS")


def stage_trials(ctx):
    m, lay, a = ctx.model, CFG["layer"], CFG["alpha"]
    st = L.ResultStore(OUT / "trials.jsonl")
    t0, done = time.time(), 0
    for di, d in enumerate(ctx.stim["draws"]):
        c, t = d["concept"], d["vec_type"]
        keys = [f"{q}|d{di}|none" for q in ("det", "ctl")] + \
               [f"{q}|d{di}|pos{p}" for q in ("det", "ctl") for p in d["positions"]]
        if all(st.has(k) for k in keys):
            continue
        meta = {"draw_index": di, "concept": c, "vec_type": t, "control_q": d["control_q"]}
        vec = a * ctx.vecs[(c, t)]
        for q, P in zip(("det", "ctl"), ctx.prompts(d)):
            cl = L.clean_pass(m, P)
            if not st.has(f"{q}|d{di}|none"):
                py, mass = p_yes(cl["logits"], ctx.y, ctx.n)
                st.add(dict(meta, key=f"{q}|d{di}|none", question=q, injected=False, pos=None,
                            p_yes=round(py, 6), yes_no_mass=round(mass, 6)))
            for pos in d["positions"]:
                key = f"{q}|d{di}|pos{pos}"
                if st.has(key):
                    continue
                t1 = time.time()
                idx = P.arm[pos - 1]["sentence"]
                r = L.run_suffix(m, P, idx[0], cl["cache"], lay, vec, idx)
                py, mass = p_yes(r["logits"], ctx.y, ctx.n)
                st.add(dict(meta, key=key, question=q, injected=True, pos=pos, p_yes=round(py, 6),
                            yes_no_mass=round(mass, 6), sec=round(time.time() - t1, 2)))
                done += 1
        log(f"draw {di + 1}/20 done ({c}|{t}); {done} injected runs, {(time.time() - t0) / 60:.1f} min")
    return verdict()


def verdict():
    R = list(L.ResultStore(OUT / "trials.jsonl").records.values())
    mean = lambda q, inj: (lambda v: sum(v) / len(v) if v else None)([r["p_yes"] for r in R if r["question"] == q and r["injected"] == inj])
    v = {"complete": len(R) == 240,
         "p_yes_det_inj": mean("det", True), "p_yes_ctl_inj": mean("ctl", True),
         "p_yes_det_none": mean("det", False), "p_yes_ctl_none": mean("ctl", False),
         "ctl_none_answered_no": sum(r["p_yes"] < 0.5 for r in R if r["question"] == "ctl" and not r["injected"])}
    if v["complete"]:
        v["detection_score_pts"] = round(100 * (v["p_yes_det_inj"] - v["p_yes_ctl_inj"]), 2)
        v["false_alarm_pts"] = round(100 * v["p_yes_det_none"], 2)
        v["valid"] = v["ctl_none_answered_no"] >= CFG["validity_min_no"]
    write_status(result=v)
    log(f"RESULT: detection score {v.get('detection_score_pts')} pts; P(yes) det inj {v['p_yes_det_inj']}, "
        f"ctl inj {v['p_yes_ctl_inj']}, det none {v['p_yes_det_none']}, ctl none {v['p_yes_ctl_none']}; "
        f"control answered no {v['ctl_none_answered_no']}/20")
    return v


def stage_summary():
    REPO_RESULTS.mkdir(parents=True, exist_ok=True)
    for f in sorted(OUT.glob("*.json")) + sorted(OUT.glob("*.jsonl")):
        shutil.copy2(f, REPO_RESULTS / f.name)
    log(f"copied results to results/study10/{REPO_RESULTS.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, choices=sorted(CKPTS))
    ap.add_argument("--stage", required=True, choices=["all", "sanity", "trials", "summary"])
    args = ap.parse_args()
    configure(args.ckpt)
    if args.stage == "summary":
        stage_summary()
        return
    ctx = Ctx10(args.ckpt)
    if args.stage in ("all", "sanity"):
        stage_sanity(ctx)
    if args.stage in ("all", "trials"):
        stage_trials(ctx)
    stage_summary()
    log("DONE")


if __name__ == "__main__":
    main()
