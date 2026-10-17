"""Study 5 — question-specificity control (noticing vs salience/spotlight).
Plan: gates/study5_plan.md. Same setup as Study 4 arm A (sentence-words-only push, layer 2,
alpha 5, their vectors). Each injected position is asked three questions in separate trials:
  Q0  their original localization question (their exact prompt)
  Q1  "Which sentence is the longest?"   (neutral intro)
  Q2  "Which sentence do you like most?"  (neutral intro)
with concept vectors and with random vectors (arms Q0/Q1/Q2 x concept/random), plus
no-injection baselines for each question.

Usage:
  python scripts/study5.py --stage all|sanity|trials|summary
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
    "repo": "meta-llama/Llama-3.1-8B-Instruct",
    "layer": 2,
    "alpha": 5.0,
    "validity_min_pull": 50.0,   # Q0-concept pull must be >= 50 points
    "small_pull": 20.0,          # < 20 points -> question-specific
    "tol_cache_logit": 1e-3,
}
OUT = L.DATA_ROOT / "study5"
LOG = L.DATA_ROOT / "logs" / "study5.log"
REPO_RESULTS = L.REPO_ROOT / "results" / "study5"
QUESTIONS = ("Q0", "Q1", "Q2")
ARMS = [(q, v) for q in QUESTIONS for v in ("concept", "random")]


def log(msg):
    L.log(msg, LOG)


class QPrompt:
    """Their format (chat template, trailing end-of-turn removed, trailing space), with a given
    intro, question and answer prefix. Q0 uses their exact intro/question/prefix."""

    def __init__(self, tok, sentences, intro, question, prefix):
        self.spans = S3.marker_spans(sentences)
        block = " ".join(self.spans)
        self.msgs = [{"role": "user", "content": intro}, {"role": "assistant", "content": "Ok."},
                     {"role": "user", "content": f"{block}\n\n{question}"},
                     {"role": "assistant", "content": prefix}]
        self.text = S3.their_format(tok, self.msgs, add_space=True)
        enc = tok(self.text, add_special_tokens=False, return_offsets_mapping=True)
        self.ids, self.offsets = enc["input_ids"], enc["offset_mapping"]
        self.block_start = self.text.find(block)
        assert self.block_start >= 0 and self.text.endswith(prefix)
        self.arm = S4.arm_spans(self, self.spans, self.block_start)

    def tensor(self):
        return torch.tensor([self.ids])


class Ctx5:
    def __init__(self, stim_path=None):
        LOG.parent.mkdir(parents=True, exist_ok=True)
        OUT.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        self.model, self.tok = L.load_model(CFG["repo"], CFG["revision"])
        log(f"loaded {CFG['repo']}@{CFG['revision']} in {time.time() - t0:.0f}s")
        stim_path = stim_path or (L.REPO_ROOT / "stimuli" / "study5_stimuli.json")
        self.stim = json.loads(Path(stim_path).read_text(encoding="utf-8"))
        self.vec_dir = Path(self.stim["vector_dir"].replace("<INTROSPECTION_DATA>", str(L.DATA_ROOT)))
        S3.VEC_DIR = self.vec_dir
        self.vecs = S3.load_their_vectors(self.stim["concepts"])
        g = torch.Generator().manual_seed(self.stim["random_vector_seed"])
        self.rand = []
        for _ in self.stim["draws"]:
            v = torch.randn(self.model.config.hidden_size, generator=g)
            self.rand.append(v / v.norm())
        self.digits = None

    def prompts(self, sentences):
        q = self.stim["questions"]
        ni = self.stim["neutral_intro"]
        return {"Q0": QPrompt(self.tok, sentences, S3.INTRO3, S3.Q_LOC3, S3.PREFIX3),
                "Q1": QPrompt(self.tok, sentences, ni, q["Q1"]["question"], q["Q1"]["prefix"]),
                "Q2": QPrompt(self.tok, sentences, ni, q["Q2"]["question"], q["Q2"]["prefix"])}

    def answer(self, logits):
        cand = logits[self.digits]
        return int(torch.argmax(cand)) + 1, [round(float(x), 4) for x in cand]


def write_status(**kw):
    path = OUT / "status.json"
    st = json.loads(path.read_text()) if path.exists() else {}
    st.update(kw)
    st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(json.dumps(st, indent=2))
    return st


def stage_sanity(ctx):
    m, tok, res = ctx.model, ctx.tok, {}
    d0 = ctx.stim["draws"][0]["sentences"]
    P = ctx.prompts(d0)
    # S1: Q0 is exactly the Study 3/4 prompt; Q1/Q2 end with their prefixes
    assert P["Q0"].text == S3.P3(tok, S3.marker_spans(d0), "locate").text
    res["S1_n_tokens"] = {q: len(P[q].ids) for q in QUESTIONS}
    res["S1_Q1_text_tail"] = P["Q1"].text[-160:]
    # S2: "1".."10" single tokens after every prefix (same ids)
    ids = {q: L.digit_token_ids(tok, P[q]) for q in QUESTIONS}
    assert ids["Q0"] == ids["Q1"] == ids["Q2"]
    ctx.digits = ids["Q0"]
    res["S2_digit_ids"] = ctx.digits
    # S3/S8: spans decode correctly; label and sentence parts disjoint
    for q in QUESTIONS:
        for k in range(10):
            sen = P[q].arm[k]["sentence"]
            lab = P[q].arm[k]["label"]
            assert d0[k] in tok.decode([P[q].ids[i] for i in sen]) and not (set(sen) & set(lab))
    # S4: alpha 0 -> identical; S5: cache == full pass (random vector)
    p = P["Q1"]
    idx = p.arm[3]["sentence"]
    x0 = L.run_full(m, p)
    x1 = L.run_full(m, p, layer=CFG["layer"], vec=torch.zeros(m.config.hidden_size), inj_abs=idx)
    res["S4_alpha0_max_abs_diff"] = float((x0 - x1).abs().max())
    assert res["S4_alpha0_max_abs_diff"] == 0.0
    cl = L.clean_pass(m, p)
    rv = L.random_unit_vectors(m.config.hidden_size, 1, seed=999)[0] * CFG["alpha"]
    fi = L.run_full(m, p, layer=CFG["layer"], vec=rv, inj_abs=idx)
    ci = L.run_suffix(m, p, idx[0], cl["cache"], CFG["layer"], rv, idx)["logits"]
    res["S5_maxdiff"] = float((ci[ctx.digits] - fi[ctx.digits]).abs().max())
    assert res["S5_maxdiff"] < CFG["tol_cache_logit"] and torch.argmax(ci[ctx.digits]) == torch.argmax(fi[ctx.digits])
    # S6': vector files identical to Studies 3-4
    for f, h in ctx.stim["vector_sha256"].items():
        assert hashlib.sha256((ctx.vec_dir / f).read_bytes()).hexdigest() == h, f
    # S9: every draw has an unambiguous longest sentence (words and characters)
    for d in ctx.stim["draws"]:
        s = d["sentences"]
        Lp = d["longest_pos"] - 1
        assert all(len(s[Lp].split()) > len(x.split()) and len(s[Lp]) > len(x) for i, x in enumerate(s) if i != Lp)
    (OUT / "sanity.json").write_text(json.dumps(res, indent=2))
    log(f"SANITY PASSED: S4 0.0, S5 {res['S5_maxdiff']:.1e}, S6' hashes ok, S9 longest unambiguous, tokens {res['S1_n_tokens']}")
    write_status(sanity="PASS")
    return res


def stage_trials(ctx):
    m, lay, a = ctx.model, CFG["layer"], CFG["alpha"]
    if ctx.digits is None:
        ctx.digits = L.digit_token_ids(ctx.tok, ctx.prompts(ctx.stim["draws"][0]["sentences"])["Q0"])
    st = {arm: L.ResultStore(OUT / f"{arm[0]}_{arm[1]}.jsonl") for arm in ARMS}
    base = L.ResultStore(OUT / "no_injection.jsonl")
    draws, t0, done = ctx.stim["draws"], time.time(), 0
    total = len(draws) * 10
    for di, d in enumerate(draws):
        c, t = d["concept"], d["vec_type"]
        keys = {(arm, pos): f"{arm[0]}|{arm[1]}|{c}|{t}|pos{pos}" for arm in ARMS for pos in range(1, 11)}
        if all(st[arm].has(keys[(arm, pos)]) for arm in ARMS for pos in range(1, 11)):
            continue
        P = ctx.prompts(d["sentences"])
        meta = {"concept": c, "vec_type": t, "draw_index": di, "longest_pos": d["longest_pos"]}
        clean = {}
        for q in QUESTIONS:
            clean[q] = L.clean_pass(m, P[q])
            bk = f"B|{q}|d{di}"
            if not base.has(bk):
                pred, cand = ctx.answer(clean[q]["logits"])
                base.add(dict(meta, key=bk, question=q, pred=pred, cand_logits=cand))
        vec = {"concept": a * ctx.vecs[(c, t)], "random": a * ctx.rand[di]}
        for k in range(10):
            pos = k + 1
            line = []
            for arm in ARMS:
                q, vt = arm
                key = keys[(arm, pos)]
                if not st[arm].has(key):
                    t1 = time.time()
                    idx = P[q].arm[k]["sentence"]
                    r = L.run_suffix(m, P[q], idx[0], clean[q]["cache"], lay, vec[vt], idx)
                    pred, cand = ctx.answer(r["logits"])
                    st[arm].add(dict(meta, key=key, question=q, vector=vt, pos=pos, pred=pred,
                                     correct=pred == pos, cand_logits=cand, sec=round(time.time() - t1, 2)))
                rec = st[arm].records[key]
                line.append(f"{q}{vt[0]}={rec['pred']}{'+' if rec['correct'] else '-'}")
            done += 1
            eta = (time.time() - t0) / done * (total - di * 10 - pos)
            log(f"pos {di * 10 + pos}/{total} {c}|{t}|pos{pos} " + " ".join(line) + f" eta={eta / 3600:.2f}h")
    return verdict()


def verdict():
    R = {arm: list(L.ResultStore(OUT / f"{arm[0]}_{arm[1]}.jsonl").records.values()) for arm in ARMS}
    B = {(r["question"], r["draw_index"]): r["pred"] for r in L.ResultStore(OUT / "no_injection.jsonl").records.values()}
    complete = all(len(R[arm]) == 200 for arm in ARMS)
    out = {}
    for arm, recs in R.items():
        n = len(recs)
        hits = sum(r["correct"] for r in recs)
        base_hits = sum(B.get((r["question"], r["draw_index"])) == r["pos"] for r in recs)
        out[f"{arm[0]}_{arm[1]}"] = {"n": n, "hits": hits, "acc_pct": round(100 * hits / n, 1) if n else None,
                                    "baseline_hits": base_hits,
                                    "pull_pts": round(100 * (hits - base_hits) / n, 1) if n else None}
    q0 = out["Q0_concept"]["pull_pts"]
    valid = complete and q0 is not None and q0 >= CFG["validity_min_pull"]

    def label(p):
        if p < CFG["small_pull"]:
            return "question-specific"
        if p >= 0.5 * q0:
            return "spotlight"
        return "partial"

    labels = {q: label(out[f"{q}_concept"]["pull_pts"]) for q in ("Q1", "Q2")} if valid else {}
    overall = None
    if valid:
        if labels["Q1"] == labels["Q2"] == "spotlight":
            overall = "spotlight"
        elif labels["Q1"] == labels["Q2"] == "question-specific":
            overall = "question-specific"
        else:
            overall = "mixed"
    # Q1 correctness: does the model pick the truly longest sentence?
    q1c = R[("Q1", "concept")]
    with_push = 100 * sum(r["pred"] == r["longest_pos"] for r in q1c) / max(len(q1c), 1)
    draws_lp = {r["draw_index"]: r["longest_pos"] for r in q1c}
    no_push = 100 * sum(B.get(("Q1", di)) == lp for di, lp in draws_lp.items()) / max(len(draws_lp), 1)
    v = {"complete": complete, "arms": out, "valid": valid, "labels": labels, "overall": overall,
         "Q1_longest_correct_with_push_pct": round(with_push, 1), "Q1_longest_correct_no_push_pct": round(no_push, 1)}
    write_status(result=v)
    log("RESULT: " + ", ".join(f"{k} acc {x['acc_pct']} pull {x['pull_pts']}" for k, x in out.items()) +
        f"; valid {valid}; labels {labels}; overall {overall}; Q1 longest-correct {with_push:.1f}% vs {no_push:.1f}% no push")
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
    ctx = Ctx5()
    if args.stage in ("all", "sanity"):
        stage_sanity(ctx)
        stage_summary()
    if args.stage in ("all", "trials"):
        stage_trials(ctx)
        stage_summary()
    log("DONE")


if __name__ == "__main__":
    main()
