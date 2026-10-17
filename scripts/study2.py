"""Study 2 — text-intact search. Plan: gates/text_intact_plan.md.

Reuses the Gate 1 trial code unchanged (scripts/gate1.py: locate_trial, copy_reader_trial,
sanity checks, resumable storage); only the settings, material and rules differ.

Usage:
  python scripts/study2.py --stage all|sanity|practice|test|summary

Per-trial results: $INTROSPECTION_DATA/study2 (default ./data/study2), copied to
results/study2/ after each step. Re-running resumes; finished trials are skipped.
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
import gate1 as G  # noqa: E402

CFG = {
    "layers": [2, 4, 8, 12, 16],
    "attempts": {1: [0.25, 0.5, 1.0], 2: [0.35, 0.7, 1.4]},
    "practice_set": 0,           # index into ctx.sets (= Gate 1 sentence set 0)
    "test_sets": [1, 2, 3, 4],   # index into ctx.sets (= the 4 new Study 2 sets)
    "reader_stop_at": 5,         # practice reader trials stop at this many correct
    "intact_max": 4,             # practice text-intact: reader <= 4/30
    "practice_pass_inj": 8,      # injected >= 8/30
    "practice_margin": 3,        # injected - reader >= 3/30 (10 points)
    "test_pass_inj": 50,         # injected >= 50/200
    "test_reader_max": 30,       # reader <= 30/200
    "test_margin": 20,           # injected - reader >= 20/200 (10 points)
    "random_seed": 20261016,
}

# Point the shared Gate 1 machinery at Study 2's layers and output folders.
G.CFG["layers"] = CFG["layers"]
G.OUT = L.DATA_ROOT / "study2"
G.LOG = L.DATA_ROOT / "logs" / "study2.log"
REPO_RESULTS = L.REPO_ROOT / "results" / "study2"
log = G.log


class Ctx2(G.Ctx):
    """Same as the Gate 1 context, but set 0 = Gate 1 practice set, sets 1-4 = new test sets."""

    def __init__(self):
        G.LOG.parent.mkdir(parents=True, exist_ok=True)
        G.OUT.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        self.model, self.tok = L.load_model(G.CFG["repo"], G.CFG["revision"])
        log(f"loaded {G.CFG['repo']}@{G.CFG['revision']} in {time.time() - t0:.0f}s")
        s1 = json.loads((L.REPO_ROOT / "stimuli" / "gate1_stimuli.json").read_text(encoding="utf-8"))
        s2 = json.loads((L.REPO_ROOT / "stimuli" / "study2_stimuli.json").read_text(encoding="utf-8"))
        self.stim = {"practice_concepts": s1["concepts_calibration"], "test_concepts": s2["concepts_test"],
                     "baseline_words": s1["baseline_words"]}
        self.sets = [s1["sentence_sets"][0]] + s2["sentence_sets"]
        self.loc = [L.Prompt(self.tok, s, "locate") for s in self.sets]
        self.digits = L.digit_token_ids(self.tok, self.loc[0])
        for p in self.loc[1:]:
            assert L.digit_token_ids(self.tok, p) == self.digits
        self._clean, self._copy_clean, self.sent_mean, self.resid_norm = {}, {}, {}, {}
        self._vecs = None

    def vecs(self):
        if self._vecs is None:
            path = G.OUT / "concept_vectors.pt"
            words = self.stim["practice_concepts"] + self.stim["test_concepts"]
            if path.exists():
                self._vecs = torch.load(path)
            else:
                v, norms = L.concept_vectors(self.model, self.tok, words, self.stim["baseline_words"], CFG["layers"])
                self._vecs = {"vecs": {f"{w}|{lay}": t for (w, lay), t in v.items()},
                              "raw_norms": {f"{w}|{lay}": n for (w, lay), n in norms.items()}}
                torch.save(self._vecs, path)
        return self._vecs

    def alpha(self, layer, r):
        """alpha = r x median activation size at this layer on the practice set (no injection)."""
        self.clean(CFG["practice_set"])
        return r * self.resid_norm[(CFG["practice_set"], layer)]


def skey(layer, r):
    return f"L{layer}|r{r}"


# ----------------------------------------------------------------------------- practice

def stage_practice(ctx, attempt):
    rs = CFG["attempts"][attempt]
    s = CFG["practice_set"]
    settings = [(lay, r) for lay in CFG["layers"] for r in rs]
    alphas = {skey(lay, r): round(ctx.alpha(lay, r), 4) for lay, r in settings}
    G.write_status(**{f"practice{attempt}_alphas": alphas})
    # 1) injected localization trials, all settings
    jobs = []
    for lay, r in settings:
        for w in ctx.stim["practice_concepts"]:
            for k in range(10):
                key = f"P{attempt}|{skey(lay, r)}|{w}|set{s}|pos{k + 1}"
                meta = {"phase": f"P{attempt}", "layer": lay, "r": r, "alpha": alphas[skey(lay, r)],
                        "concept": w, "set": s, "pos": k + 1}
                jobs.append((key, meta, (lay, r, w, k)))
    G.run_trials(f"practice{attempt}_inj", jobs,
                 lambda t: ctx.locate_trial(s, t[3], t[0], ctx.vec(t[2], t[0]), ctx.alpha(t[0], t[1]), want_kl=True))
    # 2) copy+reader trials, setting by setting, stopping once the reader reaches 5 correct
    store = L.ResultStore(G.OUT / f"practice{attempt}_reader.jsonl")
    for lay, r in settings:
        recs = [x for x in store.records.values() if x["layer"] == lay and x["r"] == r]
        n_run = 0
        for w in ctx.stim["practice_concepts"]:
            for k in range(10):
                if sum(x["reader_correct"] for x in recs) >= CFG["reader_stop_at"]:
                    break
                key = f"P{attempt}R|{skey(lay, r)}|{w}|set{s}|pos{k + 1}"
                if store.has(key):
                    continue
                rec = ctx.copy_reader_trial(s, k, lay, ctx.vec(w, lay), ctx.alpha(lay, r))
                rec.update({"phase": f"P{attempt}R", "layer": lay, "r": r, "alpha": alphas[skey(lay, r)],
                            "concept": w, "set": s, "pos": k + 1, "key": key})
                store.add(rec)
                recs.append(rec)
                n_run += 1
                log(f"practice{attempt}_reader {skey(lay, r)} {len(recs)}/30 {key} reader={rec['reader_pred']} "
                    f"ok={rec['reader_correct']} exact={rec['exact_target']} {rec['sec']}s")
        log(f"practice{attempt}_reader {skey(lay, r)} done: reader {sum(x['reader_correct'] for x in recs)}"
            f"/{len(recs)} (ran {n_run} new)")
    return verdict_practice(attempt)


def practice_table(attempt):
    inj = G.read(f"practice{attempt}_inj")
    rdr = G.read(f"practice{attempt}_reader")
    rows = []
    for lay in CFG["layers"]:
        for r in CFG["attempts"][attempt]:
            i = [x for x in inj if x["layer"] == lay and x["r"] == r]
            d = [x for x in rdr if x["layer"] == lay and x["r"] == r]
            rk, rn = sum(x["reader_correct"] for x in d), len(d)
            rows.append({
                "layer": lay, "r": r, "alpha": i[0]["alpha"] if i else None,
                "inj_k": sum(x["correct"] for x in i), "inj_n": len(i),
                "reader_k": rk, "reader_n": rn,
                "intact": rn == 30 and rk <= CFG["intact_max"],
                "exact_copy_k": sum(x["exact_target"] for x in d),
                "inj_pos12_k": sum(x["correct"] for x in i if x["pos"] in (1, 2)),
                "inj_pos3_10_k": sum(x["correct"] for x in i if x["pos"] >= 3),
                "mean_kl": round(sum(x["kl_target"] for x in i) / len(i), 3) if i else None,
            })
    return rows


def verdict_practice(attempt):
    rows = practice_table(attempt)
    complete = all(r["inj_n"] == 30 for r in rows)
    intact = [r for r in rows if r["intact"]]
    sel = sorted(intact, key=lambda r: (-r["inj_k"], r["r"], r["layer"]))[0] if intact else None
    passed = bool(complete and sel and sel["inj_k"] >= CFG["practice_pass_inj"]
                  and sel["inj_k"] - sel["reader_k"] >= CFG["practice_margin"])
    v = {"complete": complete, "n_intact": len(intact), "selected": sel, "pass": passed, "table": rows}
    G.write_status(**{f"practice{attempt}": v})
    log(f"PRACTICE ATTEMPT {attempt} {'PASS' if passed else 'FAIL'}: {len(intact)} text-intact settings; "
        f"selected {skey(sel['layer'], sel['r']) + ' inj ' + str(sel['inj_k']) + '/30 reader ' + str(sel['reader_k']) + '/30' if sel else 'none'}")
    return v


# ----------------------------------------------------------------------------- fresh test

def stage_test(ctx, sel):
    lay, r = sel["layer"], sel["r"]
    a = ctx.alpha(lay, r)
    base = {"layer": lay, "r": r, "alpha": round(a, 4)}
    dim = ctx.model.config.hidden_size
    rvs = L.random_unit_vectors(dim, len(ctx.stim["test_concepts"]), CFG["random_seed"])
    trials = [(i, w, s, k) for i, w in enumerate(ctx.stim["test_concepts"]) for s in CFG["test_sets"] for k in range(10)]
    jobs = [(f"T|{skey(lay, r)}|{w}|set{s}|pos{k + 1}", dict(base, phase="T", concept=w, set=s, pos=k + 1), (s, k, w))
            for i, w, s, k in trials]
    G.run_trials("test_inj", jobs, lambda t: ctx.locate_trial(t[0], t[1], lay, ctx.vec(t[2], lay), a, want_kl=True))
    jobs = [(f"TR|{skey(lay, r)}|{w}|set{s}|pos{k + 1}", dict(base, phase="TR", concept=w, set=s, pos=k + 1), (s, k, w))
            for i, w, s, k in trials]
    G.run_trials("test_reader", jobs, lambda t: ctx.copy_reader_trial(t[0], t[1], lay, ctx.vec(t[2], lay), a))
    jobs = [(f"TRAND|{skey(lay, r)}|rand{i}|set{s}|pos{k + 1}", dict(base, phase="TRAND", concept=f"rand{i}", set=s,
                                                                       pos=k + 1), (s, k, i))
            for i, w, s, k in trials]
    G.run_trials("test_random", jobs, lambda t: ctx.locate_trial(t[0], t[1], lay, rvs[t[2]], a, want_kl=True))
    return verdict_test()


def verdict_test():
    inj, rdr, rnd = G.read("test_inj"), G.read("test_reader"), G.read("test_random")
    ik, rk = sum(x["correct"] for x in inj), sum(x["reader_correct"] for x in rdr)
    passed = (len(inj) == 200 and len(rdr) == 200 and ik >= CFG["test_pass_inj"]
              and rk <= CFG["test_reader_max"] and ik - rk >= CFG["test_margin"])
    v = {"inj_k": ik, "inj_n": len(inj), "reader_k": rk, "reader_n": len(rdr),
         "random_k": sum(x["correct"] for x in rnd), "random_n": len(rnd),
         "exact_copy_k": sum(x["exact_target"] for x in rdr),
         "inj_wilson95": G.wilson(ik, len(inj)), "reader_wilson95": G.wilson(rk, len(rdr)), "pass": passed}
    G.write_status(test=v)
    log(f"FRESH TEST {'PASS' if passed else 'FAIL'}: injected {ik}/{len(inj)}, reader {rk}/{len(rdr)}, "
        f"random {v['random_k']}/{v['random_n']}")
    return v


def stage_summary():
    REPO_RESULTS.mkdir(parents=True, exist_ok=True)
    for f in sorted(G.OUT.glob("*.json")) + sorted(G.OUT.glob("*.jsonl")):
        shutil.copy2(f, REPO_RESULTS / f.name)
    log(f"copied results to {REPO_RESULTS}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["all", "sanity", "practice", "test", "summary"])
    args = ap.parse_args()
    pin = json.loads((L.REPO_ROOT / "stimuli" / "model_pins.json").read_text())
    G.CFG["revision"] = pin[G.CFG["repo"]]
    if args.stage == "summary":
        stage_summary()
        return
    ctx = Ctx2()
    if args.stage in ("all", "sanity"):
        G.stage_sanity(ctx)
        stage_summary()
    sel = None
    if args.stage in ("all", "practice", "test"):
        for attempt in (1, 2):
            v = stage_practice(ctx, attempt) if args.stage != "test" else verdict_practice(attempt)
            stage_summary()
            if v["pass"]:
                sel = v["selected"]
                break
            log(f"practice attempt {attempt} failed" + ("; running pre-logged attempt 2" if attempt == 1 else ""))
        if sel is None:
            log("STOP: no text-intact setting passed after 2 attempts. Write up.")
            log("DONE")
            return
    if args.stage in ("all", "test"):
        stage_test(ctx, sel)
        stage_summary()
    log("DONE")


if __name__ == "__main__":
    main()
