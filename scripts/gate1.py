"""Gate 1 (positive control) + Gate 2 (cheap-trick controls) for sentence localization.

Usage:
  python scripts/gate1.py --stage sanity|A|B|controls|summary|all

All per-trial results are appended (fsync'd) to JSONL files under
$INTROSPECTION_DATA/gate1 (default ./data/gate1). Re-running any stage resumes:
trials already stored are skipped. Rules applied here are the ones approved in
gates/gate0_report.md Section 5 and frozen in gates/gate1_plan.md.
"""
import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import introspect_lib as L  # noqa: E402

CFG = {
    "repo": "allenai/Llama-3.1-Tulu-3-8B",
    "revision": None,  # filled from gates/gate1_plan.md pin at runtime (see main)
    "layers": [1, 2, 3, 4, 6, 8],
    "alphas": [2, 5, 8, 14],
    "calib_set": 0,
    "test_sets": [1, 2, 3, 4, 5],
    "random_seed": 20261015,
    "pass_A_correct": 9,     # of 30 per cell  (>= 30%)
    "pass_B_correct": 90,    # of 300          (>= 30%)
    "gate2_margin_pts": 15,  # injected - reader >= 15 points
    "gate2_reader_max": 30.0,  # reader < 30%
    "random_label_margin_pts": 10,
    "tol_cache_logit": 1e-3,
}
OUT = L.DATA_ROOT / "gate1"
LOG = L.DATA_ROOT / "logs" / "gate1.log"
REPO_RESULTS = L.REPO_ROOT / "results" / "gate1"


def log(msg):
    L.log(msg, LOG)


class Ctx:
    def __init__(self):
        LOG.parent.mkdir(parents=True, exist_ok=True)
        OUT.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        self.model, self.tok = L.load_model(CFG["repo"], CFG["revision"])
        log(f"loaded {CFG['repo']}@{CFG['revision']} in {time.time() - t0:.0f}s")
        self.stim = json.loads((L.REPO_ROOT / "stimuli" / "gate1_stimuli.json").read_text(encoding="utf-8"))
        self.sets = self.stim["sentence_sets"]
        self.loc = [L.Prompt(self.tok, s, "locate") for s in self.sets]
        self.digits = L.digit_token_ids(self.tok, self.loc[0])
        for p in self.loc[1:]:
            assert L.digit_token_ids(self.tok, p) == self.digits
        self._clean, self._copy_clean, self.sent_mean, self.resid_norm = {}, {}, {}, {}
        self._vecs = None

    # -- clean passes (cached in memory) --
    def clean(self, s):
        if s not in self._clean:
            r = L.clean_pass(self.model, self.loc[s], want_hidden=True, want_final=True)
            for k, idx in enumerate(self.loc[s].user_tok):
                for lay in CFG["layers"]:
                    h = r["hidden"][lay + 1][0, idx]
                    self.sent_mean[(s, k, lay)] = h.mean(0)
            for lay in CFG["layers"]:
                rows = torch.cat([r["hidden"][lay + 1][0, idx] for idx in self.loc[s].user_tok])
                self.resid_norm[(s, lay)] = float(rows.norm(dim=-1).median())
            r["hidden"] = None
            self._clean[s] = r
        return self._clean[s]

    def copy_prompt(self, s):
        if s not in self._copy_clean:
            p = L.Prompt(self.tok, self.sets[s], "copy")
            self._copy_clean[s] = (p, L.clean_pass(self.model, p, want_final=False))
        return self._copy_clean[s]

    def vecs(self):
        if self._vecs is None:
            path = OUT / "concept_vectors.pt"
            words = self.stim["concepts_calibration"] + self.stim["concepts_test"]
            if path.exists():
                self._vecs = torch.load(path)
            else:
                v, norms = L.concept_vectors(self.model, self.tok, words, self.stim["baseline_words"], CFG["layers"])
                self._vecs = {"vecs": {f"{w}|{lay}": t for (w, lay), t in v.items()},
                              "raw_norms": {f"{w}|{lay}": n for (w, lay), n in norms.items()}}
                torch.save(self._vecs, path)
        return self._vecs

    def vec(self, word, layer):
        return self.vecs()["vecs"][f"{word}|{layer}"]

    def answer(self, logits):
        cand = logits[self.digits]
        return int(torch.argmax(cand)) + 1, [round(float(x), 4) for x in cand]

    # -- trial types --
    def locate_trial(self, s, k, layer, unit_vec, alpha, want_kl=False):
        p, cl = self.loc[s], self.clean(s)
        idx = p.user_tok[k]
        start = idx[0]
        t0 = time.time()
        r = L.run_suffix(self.model, p, start, cl["cache"], layer, alpha * unit_vec, idx, want_final=want_kl)
        pred, cand = self.answer(r["logits"])
        rec = {"pred": pred, "target": k + 1, "correct": pred == k + 1, "cand_logits": cand}
        if want_kl:
            with torch.inference_mode():
                lc = torch.log_softmax(L.logits_at(self.model, cl["final"][idx]), -1)
                li = torch.log_softmax(L.logits_at(self.model, r["final"][[i - start for i in idx]]), -1)
                rec["kl_target"] = float((lc.exp() * (lc - li)).sum(-1).mean())
        rec["cos_concept_sentence"] = float(torch.nn.functional.cosine_similarity(
            unit_vec, self.sent_mean[(s, k, layer)], dim=0))
        rec["sec"] = round(time.time() - t0, 2)
        return rec

    def copy_reader_trial(self, s, k, layer, unit_vec, alpha):
        p, cl = self.copy_prompt(s)
        idx = p.user_tok[k]
        start = idx[0]
        t0 = time.time()
        r = L.run_suffix(self.model, p, start, cl["cache"], layer, alpha * unit_vec, idx, want_final=True)
        perceived, tok_acc = [], []
        with torch.inference_mode():
            for j, toks in enumerate(p.resp_tok):
                rows = r["final"][[t - 1 - start for t in toks]]
                pred_ids = torch.argmax(L.logits_at(self.model, rows), -1).tolist()
                true_ids = [p.ids[t] for t in toks]
                perceived.append(self.tok.decode(pred_ids).strip())
                tok_acc.append(sum(a == b for a, b in zip(pred_ids, true_ids)) / len(toks))
        exact = [perceived[j] == self.sets[s][j].strip() for j in range(10)]
        reader_prompt = L.Prompt(self.tok, perceived, "locate", strict=False)
        rp, rcand = self.answer(L.run_full(self.model, reader_prompt))
        return {"target": k + 1, "perceived": perceived, "exact_target": exact[k], "exact_all": all(exact),
                "n_exact": sum(exact), "tok_acc_target": round(tok_acc[k], 4),
                "reader_pred": rp, "reader_correct": rp == k + 1, "reader_cand_logits": rcand,
                "sec": round(time.time() - t0, 2)}

    def literal_trial(self, s, k, word):
        sents = list(self.sets[s])
        sents[k] = sents[k].rstrip(".") + f", {word}."
        p = L.Prompt(self.tok, sents, "locate")
        cl, base = self.clean(s), self.loc[s]
        start = p.user_tok[k][0]
        t0 = time.time()
        if p.ids[:start] == base.ids[:start]:
            logits = L.run_suffix(self.model, p, start, cl["cache"])["logits"]
        else:
            logits = L.run_full(self.model, p)
        pred, cand = self.answer(logits)
        return {"pred": pred, "target": k + 1, "correct": pred == k + 1, "cand_logits": cand,
                "modified": sents[k], "sec": round(time.time() - t0, 2)}


# ----------------------------------------------------------------------------- helpers

class ComputePause(Exception):
    pass


def run_trials(name, jobs, fn, max_median_sec=None):
    """jobs: list of (key, meta, thunk). Resumable, with progress + ETA in the log.
    max_median_sec: compute stop rule, checked after 20 trials."""
    store = L.ResultStore(OUT / f"{name}.jsonl")
    todo = [j for j in jobs if not store.has(j[0])]
    log(f"{name}: {len(jobs)} trials, {len(jobs) - len(todo)} already done, {len(todo)} to run")
    t0, done, secs = time.time(), 0, []
    for key, meta, thunk in todo:
        if max_median_sec and len(secs) == 20:
            med = sorted(secs)[10]
            if med > max_median_sec:
                write_status(compute_pause={"stage": name, "median_sec": med, "limit": max_median_sec})
                raise ComputePause(f"median {med:.1f}s > {max_median_sec}s")
        rec = fn(thunk)
        secs.append(rec.get("sec", 0))
        rec.update(meta)
        rec["key"] = key
        store.add(rec)
        done += 1
        el = time.time() - t0
        eta = el / done * (len(todo) - done)
        ok = rec.get("correct", rec.get("reader_correct"))
        log(f"{name} {done}/{len(todo)} {key} pred={rec.get('pred', rec.get('reader_pred'))} "
            f"ok={ok} {rec.get('sec')}s  eta={eta / 3600:.2f}h")
    return list(store.records.values())


def read(name):
    return list(L.ResultStore(OUT / f"{name}.jsonl").records.values())


def write_status(**kw):
    path = OUT / "status.json"
    st = json.loads(path.read_text()) if path.exists() else {}
    st.update(kw)
    st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(json.dumps(st, indent=2))
    return st


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


def binom_sf(k, n, p=0.1):
    """P(X >= k) for X ~ Binomial(n, p)."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


# ----------------------------------------------------------------------------- stages

def stage_sanity(ctx):
    res = {}
    tok, m = ctx.tok, ctx.model
    # S1: our tokenization of the chat prompt == the tokenizer's own chat-template tokens
    for s, p in enumerate(ctx.loc):
        tmpl = L._template_ids(tok, p.msgs)
        if p.added_bos:
            tmpl = [tok.bos_token_id] + tmpl
        assert p.ids[: len(tmpl)] == tmpl, f"S1 template mismatch set {s}"
    res["S1_template_ids_match"] = True
    res["S1_bos_prepended"] = ctx.loc[0].added_bos
    res["S1_concept_prompt_ids"] = tok(L.CONCEPT_TEMPLATE.format(word="volcano"))["input_ids"]
    # S2: "1".."10" single tokens (checked in Ctx init)
    res["S2_digit_ids"] = ctx.digits
    res["S2_digit_strings"] = [tok.decode([d]) for d in ctx.digits]
    # S3: every sentence maps to tokens that decode back to (a superset of) the sentence
    for s, p in enumerate(ctx.loc):
        for k, idx in enumerate(p.user_tok):
            dec = tok.decode([p.ids[i] for i in idx])
            assert ctx.sets[s][k] in dec, (s, k, dec)
    res["S3_sentence_spans_ok"] = True
    res["n_tokens_locate_prompt"] = [len(p.ids) for p in ctx.loc]
    # S4: alpha = 0 injection gives exactly the same logits as no hook
    p = ctx.loc[0]
    idx = p.user_tok[3]
    dim = m.config.hidden_size
    a = L.run_full(m, p)
    b = L.run_full(m, p, layer=2, vec=torch.zeros(dim), inj_abs=idx)
    res["S4_alpha0_max_abs_diff"] = float((a - b).abs().max())
    assert res["S4_alpha0_max_abs_diff"] == 0.0
    # S5: cache speed-up gives the same answer logits as a full pass (no injection / injection
    # with a RANDOM vector, so no concept data is produced here)
    cl = ctx.clean(0)
    rv = L.random_unit_vectors(dim, 1, seed=999)[0] * 5.0
    t0 = time.time()
    c_none = L.run_suffix(m, p, idx[0], cl["cache"])["logits"]
    t_suffix = time.time() - t0
    f_inj = L.run_full(m, p, layer=2, vec=rv, inj_abs=idx)
    c_inj = L.run_suffix(m, p, idx[0], cl["cache"], 2, rv, idx)["logits"]
    d1 = float((c_none[ctx.digits] - cl["logits"][ctx.digits]).abs().max())
    d2 = float((c_inj[ctx.digits] - f_inj[ctx.digits]).abs().max())
    res["S5_cache_vs_full_maxdiff_noinj"] = d1
    res["S5_cache_vs_full_maxdiff_inj"] = d2
    res["S5_same_argmax"] = bool(torch.argmax(c_inj[ctx.digits]) == torch.argmax(f_inj[ctx.digits]))
    assert d1 < CFG["tol_cache_logit"] and d2 < CFG["tol_cache_logit"] and res["S5_same_argmax"]
    res["sec_suffix_pass_pos4"] = round(t_suffix, 1)
    # Copy prompt builds and spans resolve
    cp, _ = ctx.copy_prompt(1)
    res["n_tokens_copy_prompt"] = len(cp.ids)
    # C5: no-injection answers (position bias) for every sentence set
    c5 = {}
    for s in range(len(ctx.sets)):
        pred, cand = ctx.answer(ctx.clean(s)["logits"])
        c5[s] = {"pred": pred, "cand_logits": cand}
    res["C5_no_injection"] = c5
    res["resid_norm_median_by_layer_set0"] = {lay: round(ctx.resid_norm[(0, lay)], 3) for lay in CFG["layers"]}
    ctx.vecs()
    res["concept_raw_norms"] = {k: round(v, 3) for k, v in ctx.vecs()["raw_norms"].items()}
    (OUT / "sanity.json").write_text(json.dumps(res, indent=2))
    log(f"SANITY PASSED: {json.dumps({k: v for k, v in res.items() if k.startswith('S')})}")
    write_status(sanity="PASS")
    return res


def cellkey(layer, alpha):
    return f"L{layer}|a{alpha}"


def stage_A(ctx):
    s = CFG["calib_set"]
    jobs = []
    for lay in CFG["layers"]:
        for a in CFG["alphas"]:
            for w in ctx.stim["concepts_calibration"]:
                for k in range(10):
                    key = f"A|{cellkey(lay, a)}|{w}|set{s}|pos{k + 1}"
                    meta = {"phase": "A", "layer": lay, "alpha": a, "concept": w, "set": s, "pos": k + 1}
                    jobs.append((key, meta, (s, k, lay, w, a)))
    run_trials("phaseA", jobs, lambda t: ctx.locate_trial(t[0], t[1], t[2], ctx.vec(t[3], t[2]), t[4]),
               max_median_sec=2 * 30.3)
    return verdict_A()


def verdict_A():
    recs = read("phaseA")
    cells = {}
    for r in recs:
        c = cells.setdefault((r["layer"], r["alpha"]), [0, 0])
        c[0] += r["correct"]
        c[1] += 1
    complete = all(v[1] == 30 for v in cells.values()) and len(cells) == len(CFG["layers"]) * len(CFG["alphas"])
    best = sorted(cells.items(), key=lambda kv: (-kv[1][0], kv[0][1], kv[0][0]))[0]
    (lay, a), (k, n) = best
    passed = complete and k >= CFG["pass_A_correct"]
    table = {cellkey(*c): f"{v[0]}/{v[1]}" for c, v in sorted(cells.items())}
    st = write_status(A={"complete": complete, "best_layer": lay, "best_alpha": a, "best_correct": k,
                         "n": n, "pass": passed, "table": table})
    log(f"PHASE A {'PASS' if passed else 'FAIL'}: best L{lay} a{a} = {k}/{n}")
    return st["A"]


def selected_cell():
    st = json.loads((OUT / "status.json").read_text())
    assert st["A"]["pass"], "Phase A did not pass"
    return st["A"]["best_layer"], st["A"]["best_alpha"]


def stage_B(ctx):
    lay, a = selected_cell()
    jobs = []
    for w in ctx.stim["concepts_test"]:
        for s in CFG["test_sets"]:
            for k in range(10):
                key = f"B|{cellkey(lay, a)}|{w}|set{s}|pos{k + 1}"
                jobs.append((key, {"phase": "B", "layer": lay, "alpha": a, "concept": w, "set": s, "pos": k + 1},
                             (s, k, w)))
    run_trials("phaseB", jobs, lambda t: ctx.locate_trial(t[0], t[1], lay, ctx.vec(t[2], lay), a, want_kl=True))
    return verdict_B()


def verdict_B():
    recs = read("phaseB")
    k, n = sum(r["correct"] for r in recs), len(recs)
    passed = n == 300 and k >= CFG["pass_B_correct"]
    st = write_status(B={"correct": k, "n": n, "acc_pct": round(100 * k / n, 1), "wilson95": wilson(k, n),
                         "p_vs_chance": binom_sf(k, n), "pass": passed})
    log(f"PHASE B {'PASS' if passed else 'FAIL'}: {k}/{n}")
    return st["B"]


def stage_controls(ctx):
    lay, a = selected_cell()
    st = json.loads((OUT / "status.json").read_text())
    assert st["B"]["pass"], "Phase B did not pass; controls are not run (stop rule)"
    dim = ctx.model.config.hidden_size
    rvs = L.random_unit_vectors(dim, len(ctx.stim["concepts_test"]), CFG["random_seed"])
    # C3 random vectors
    jobs = []
    for i, rv in enumerate(rvs):
        for s in CFG["test_sets"]:
            for k in range(10):
                key = f"C3|{cellkey(lay, a)}|rand{i}|set{s}|pos{k + 1}"
                jobs.append((key, {"phase": "C3", "layer": lay, "alpha": a, "concept": f"rand{i}", "set": s,
                                   "pos": k + 1}, (s, k, i)))
    run_trials("C3_random", jobs, lambda t: ctx.locate_trial(t[0], t[1], lay, rvs[t[2]], a, want_kl=True))
    # C1/C2 copy + perceived-text reader
    jobs = []
    for w in ctx.stim["concepts_reader_and_literal"]:
        for s in CFG["test_sets"]:
            for k in range(10):
                key = f"C1|{cellkey(lay, a)}|{w}|set{s}|pos{k + 1}"
                jobs.append((key, {"phase": "C1", "layer": lay, "alpha": a, "concept": w, "set": s, "pos": k + 1},
                             (s, k, w)))
    run_trials("C1_reader", jobs, lambda t: ctx.copy_reader_trial(t[0], t[1], lay, ctx.vec(t[2], lay), a))
    # C6 literal word in text
    jobs = []
    for w in ctx.stim["concepts_reader_and_literal"]:
        for s in CFG["test_sets"]:
            for k in range(10):
                key = f"C6|{w}|set{s}|pos{k + 1}"
                jobs.append((key, {"phase": "C6", "concept": w, "set": s, "pos": k + 1}, (s, k, w)))
    run_trials("C6_literal", jobs, lambda t: ctx.literal_trial(t[0], t[1], t[2]))
    return verdict_gate2()


def acc(recs, field="correct"):
    n = len(recs)
    k = sum(r[field] for r in recs)
    return {"correct": k, "n": n, "acc_pct": round(100 * k / n, 1) if n else None, "wilson95": wilson(k, n)}


def verdict_gate2():
    B = read("phaseB")
    reader_concepts = set(json.loads((L.REPO_ROOT / "stimuli" / "gate1_stimuli.json").read_text())
                          ["concepts_reader_and_literal"])
    B150 = [r for r in B if r["concept"] in reader_concepts]
    C1, C3, C6 = read("C1_reader"), read("C3_random"), read("C6_literal")
    inj = acc(B150)
    rdr = acc(C1, "reader_correct")
    rnd = acc(C3)
    allB = acc(B)
    margin = inj["acc_pct"] - rdr["acc_pct"]
    g2_pass = (len(C1) == 150 and margin >= CFG["gate2_margin_pts"] and rdr["acc_pct"] < CFG["gate2_reader_max"])
    label = ("disturbance localization" if rnd["acc_pct"] >= allB["acc_pct"] - CFG["random_label_margin_pts"]
             else "concept noticing")
    exact_rate = round(100 * sum(r["exact_target"] for r in C1) / len(C1), 1) if C1 else None
    st = write_status(gate2={"injected_same150": inj, "reader": rdr, "margin_pts": round(margin, 1),
                             "pass": g2_pass, "random": rnd, "random_label": label,
                             "literal_C6": acc(C6), "exact_copy_target_pct": exact_rate})
    log(f"GATE 2 {'PASS' if g2_pass else 'FAIL'}: injected {inj['acc_pct']}% vs reader {rdr['acc_pct']}% "
        f"(margin {margin:.1f}); random {rnd['acc_pct']}% -> label '{label}'")
    return st["gate2"]


def stage_summary():
    """Copy trial files + status into the repo (small text files) for committing."""
    REPO_RESULTS.mkdir(parents=True, exist_ok=True)
    for f in ["status.json", "sanity.json", "phaseA.jsonl", "phaseB.jsonl", "C1_reader.jsonl",
              "C3_random.jsonl", "C6_literal.jsonl"]:
        if (OUT / f).exists():
            shutil.copy2(OUT / f, REPO_RESULTS / f)
    log(f"copied results to {REPO_RESULTS}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["sanity", "A", "B", "controls", "summary", "all"])
    args = ap.parse_args()
    pin = json.loads((L.REPO_ROOT / "stimuli" / "model_pins.json").read_text())
    CFG["revision"] = pin[CFG["repo"]]
    if args.stage == "summary":
        stage_summary()
        return
    ctx = Ctx()
    if args.stage in ("sanity", "all"):
        stage_sanity(ctx)
    if args.stage in ("A", "all"):
        try:
            A = stage_A(ctx)
        except ComputePause as e:
            log(f"PAUSE (compute stop rule): {e}. Report before continuing.")
            return
        stage_summary()
        if not A["pass"]:
            log("STOP: Phase A failed (calibration attempt 1). Next attempt must be logged before running.")
            return
    if args.stage in ("B", "all"):
        B = stage_B(ctx)
        stage_summary()
        if not B["pass"]:
            log("STOP: Phase B failed. Stop rule: no re-tuning; only the pre-registered fallback is allowed.")
            return
    if args.stage in ("controls", "all"):
        stage_controls(ctx)
        stage_summary()
    log("DONE")


if __name__ == "__main__":
    main()
