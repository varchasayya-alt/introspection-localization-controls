"""Study 17 analysis (gates/PREREG_study17.md): Gate 1 verdicts, per model and layer the Study 16
analysis (yes-rates, V1-V3, Q1-Q3 with the primary "response-only" fingerprint FP-B, secondary
logit contrasts, per-question table, single-question flag, leave-one-question-out Q2; FP-A logit
contrasts as a secondary condition), the KEY COMPARISON per layer, and prediction scoring.
Reads results/study17/<model>/ (or --results); writes results/study17_summary.md + .json.

Conventions fixed before any data: dose index 0 = clean (k0), 1-3 = s25, s50, s90 (collapsed doses
share runs); yes-rate = % of all trials; broken = "neither" > 20% (random pooled); a question is
scored at the highest dose index where its conditions and dose 0 are valid; V3 = positive control
yes-rate at s90 >= 50% on test data. Logit contrasts use s90. Bootstrap: 10,000 resamples over test
draws, seed 20261073; percentile CIs. "Positive" contrast = CI entirely above 0.

Usage: python scripts/study17_analyze.py [--results DIR] [--out DIR] [--models 4b,27b]
"""
import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DOSES = [0, 1, 2, 3]
BROKEN, V2_MIN_NO, V3_MIN_YES, SEED = 20.0, 80.0, 50.0, 20261073
GATE_LETTER = {"REPORTS": "Reports (>= 20 points)", "WEAK": "Weak (5-20 points)", "DOESN'T REPORT": "Doesn't report (<= 5)"}
PRED = {"P1": "Weak (5-20 points)", "P2": "Doesn't report (<= 5)", "P3": "SUPPORTED", "P4": "b", "P5": "a", "P6": "b"}
REF_C = {"Llama-3.2-3B (Study 15)": [-0.426, [-0.460, -0.393]], "Llama-3.1-8B (Study 16)": [-1.133, [-1.191, -1.073]]}


def jread(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load(path):
    R = {}
    p = Path(path)
    if not p.exists():
        return R
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            R[r["key"]] = r
    return R


def q2_letter(d, mb):
    return "a" if (d >= 10 and mb < 10) else "b" if (d < 10 and mb >= 10) else "c" if (d >= 10 and mb >= 10) else "d"


def analyse_layer(R, lay, cqs):
    ks = sorted({r["trial"] for r in R.values() if f"det|k{r['trial']}" in R and f"ctl|k{r['trial']}" in R
                 and f"L{lay}|k0" in R[f"det|k{r['trial']}"]["out"]})
    if not ks:
        return None
    N = len(ks)
    det = [R[f"det|k{k}"]["out"] for k in ks]
    ctl = [R[f"ctl|k{k}"]["out"] for k in ks]
    cq = np.array([R[f"ctl|k{k}"]["control_q"] for k in ks])

    def key(c, d):
        return f"L{lay}|k0" if d == 0 else f"L{lay}|{c}|d{d}"

    def arr(outs, c, d, field):
        if field == "yn":
            return np.array([o[key(c, d)]["yn"] for o in outs])
        return np.array([o[key(c, d)]["cls"] == field for o in outs], float)

    conds = {"a_detection": (det, "fpB"), "b_control": (ctl, "fpB"), "d_positive": (det, "pos")}
    rands = [f"rand{j}" for j in range(5)]
    table = {}
    for n, (outs, c) in conds.items():
        table[n] = {d: {f: float(100 * arr(outs, c, d, f).mean()) for f in ("yes", "no", "neither")} for d in DOSES}
    table["c_random"] = {d: {f: float(100 * np.mean([arr(det, r, d, f) for r in rands])) for f in ("yes", "no", "neither")} for d in DOSES}
    for n in table:
        for d in DOSES:
            table[n][d]["broken"] = table[n][d]["neither"] > BROKEN

    def rise(n, d, mask=None):
        if n == "c_random":
            x = 100 * np.mean([arr(det, r, d, "yes") - arr(det, r, 0, "yes") for r in rands], axis=0)
        else:
            outs, c = conds[n]
            x = 100 * (arr(outs, c, d, "yes") - arr(outs, c, 0, "yes"))
        return x if mask is None else x[mask]

    BI = np.random.RandomState(SEED).randint(0, N, size=(10000, N))

    def ci(x):
        bs = x[BI].mean(1)
        return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]

    def scored(names):
        for d in (3, 2, 1):
            if all(not table[n][d]["broken"] and not table[n][0]["broken"] for n in names):
                return d
        return None

    V2 = table["b_control"][0]["no"] >= V2_MIN_NO
    V3 = table["d_positive"][3]["yes"] >= V3_MIN_YES
    q = {}
    d1 = scored(["a_detection"])
    if d1 is None:
        q["Q1"] = {"letter": "not scorable"}
    else:
        r = rise("a_detection", d1)
        m, c = float(r.mean()), ci(r)
        q["Q1"] = {"dose": d1, "rise": round(m, 2), "ci": c, "letter": "a" if m >= 20 else ("b" if m >= 5 else "c"),
                   "ci_crosses_boundary": any(c[0] < b <= c[1] for b in (5, 20))}
    d2 = scored(["a_detection", "b_control"])
    if d2 is None or not V2:
        q["Q2"] = {"letter": "not scorable", "V2_ok": V2}
    else:
        ra, rb = rise("a_detection", d2), rise("b_control", d2)
        dd, mb = float((ra - rb).mean()), float(rb.mean())
        cd, cb = ci(ra - rb), ci(rb)
        q["Q2"] = {"dose": d2, "rise_detection": round(float(ra.mean()), 2), "rise_control": round(mb, 2), "rise_control_ci": cb,
                   "diff": round(dd, 2), "diff_ci": cd, "letter": q2_letter(dd, mb),
                   "ci_crosses_boundary": (cd[0] < 10 <= cd[1]) or (cb[0] < 10 <= cb[1]),
                   "leave_one_question_out": {cqs[i]: q2_letter(float((ra[cq != i] - rb[cq != i]).mean()), float(rb[cq != i].mean()))
                                              for i in range(10) if (cq != i).any()}}
    d3 = scored(["a_detection", "c_random"])
    if d3 is None:
        q["Q3"] = {"letter": "not scorable"}
    else:
        ra, rc = rise("a_detection", d3), rise("c_random", d3)
        dd, cd = float((ra - rc).mean()), ci(ra - rc)
        q["Q3"] = {"dose": d3, "rise_detection": round(float(ra.mean()), 2), "rise_random": round(float(rc.mean()), 2), "diff": round(dd, 2),
                   "diff_ci": cd, "letter": "b" if dd >= 10 else ("c" if dd <= -10 else "a"),
                   "ci_crosses_boundary": any(cd[0] < b <= cd[1] for b in (-10, 10))}

    def shift(outs, c):
        return arr(outs, c, 3, "yn") - arr(outs, c, 0, "yn")
    sec = {}
    for fp in ("fpB", "fpA"):
        Ld, Lc = shift(det, fp), shift(ctl, fp)
        Lr = np.mean([shift(det, r) for r in rands], axis=0)
        s = {"L_det": [round(float(Ld.mean()), 3), ci(Ld)], "L_ctl": [round(float(Lc.mean()), 3), ci(Lc)],
             "L_rand": [round(float(Lr.mean()), 3), ci(Lr)], "C_det_minus_ctl": [round(float((Ld - Lc).mean()), 3), ci(Ld - Lc)],
             "det_minus_rand": [round(float((Ld - Lr).mean()), 3), ci(Ld - Lr)]}
        cC, cdet, cctl, cr = s["C_det_minus_ctl"][1], s["L_det"][1], s["L_ctl"][1], s["det_minus_rand"][1]
        s["reading_specificity"] = ("no lean" if (cdet[0] <= 0 <= cdet[1] and cctl[0] <= 0 <= cctl[1]) else
                                    "detection-specific lean" if cC[0] > 0 else "leans more on unrelated questions" if cC[1] < 0 else "equal lean")
        s["reading_vs_random"] = "stronger than random" if cr[0] > 0 else ("weaker than random" if cr[1] < 0 else "no different from random")
        s["C_positive"] = cC[0] > 0
        sec[fp] = s
    Lp = shift(det, "pos")
    sec["L_pos"] = [round(float(Lp.mean()), 3), ci(Lp)]
    perq = {cqs[i]: {"n": int((cq == i).sum()), **{f"yes_d{d}": round(float(100 * arr(ctl, "fpB", d, "yes")[cq == i].mean()), 1) for d in DOSES}}
            for i in range(10) if (cq == i).any()}
    dq = d2 if d2 is not None else 3
    counts = {cqs[i]: int(arr(ctl, "fpB", dq, "yes")[cq == i].sum()) for i in range(10) if (cq == i).any()}
    tot = sum(counts.values())
    top = max(counts, key=counts.get)
    flag = {"dose": dq, "total_control_yes": tot, "top_question": top, "single_question_driven": bool(tot and counts[top] > tot / 2)}
    return {"n_draws": N, "table": table, "V2_ok": V2, "V3_ok": V3, "V3_pos_yes_top": round(table["d_positive"][3]["yes"], 2),
            "questions": q, "secondary": sec, "per_question": perq, "single_question_flag": flag}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results" / "study17"))
    ap.add_argument("--out", default=str(REPO / "results"))
    ap.add_argument("--models", default="4b,27b")
    args = ap.parse_args()
    RES, O = Path(args.results), Path(args.out)
    cqs = json.loads((REPO / "stimuli" / "study10_stimuli.json").read_text(encoding="utf-8"))["control_questions"]
    out = {"models": {}}
    for m in args.models.split(","):
        d = RES / m
        fp = jread(d / "fingerprint.json")
        res = {"gate": jread(d / "gate.json"), "stopped_at_gate": (d / "STOPPED_AT_GATE").exists(),
               "doses": jread(d / "doses.json"), "fingerprint": fp, "layers": {}}
        R = load(d / "test.jsonl")
        if fp and R:
            for tag in ("a", "b"):
                lay = fp["layers"][tag]
                res["layers"][tag] = {"block": lay, "result": analyse_layer(R, lay, cqs)}
        out["models"][m] = res

    def gate_of(m):
        g = out["models"].get(m, {}).get("gate")
        return g["verdict"] if g else None

    def layer_res(m, tag):
        return (out["models"].get(m, {}).get("layers", {}).get(tag) or {}).get("result")

    key = {}
    for tag in ("a", "b"):
        r27, r4, g27_ = layer_res("27b", tag), layer_res("4b", tag), gate_of("27b")
        if g27_ is None or (g27_ == "REPORTS" and r27 is None):
            v = "PENDING (27B arm not run)"
        elif g27_ != "REPORTS" or not r27["V3_ok"]:
            v = "NOT TESTABLE"
        elif r4 is None:
            v = "PENDING (4B arm missing)"
        else:
            p27, p4 = r27["secondary"]["fpB"]["C_positive"], r4["secondary"]["fpB"]["C_positive"]
            v = ("SUPPORTED" if p27 and not p4 else "NOT SUPPORTED, positive everywhere" if p27 and p4
                 else "NOT SUPPORTED, not positive in the reporter")
        note = "both Gemma models report: within-family contrast rests on the Llama references (different protocol)" \
            if gate_of("4b") == "REPORTS" and gate_of("27b") == "REPORTS" else None
        key[tag] = {"verdict": v, "note": note,
                    "C": {m: (layer_res(m, tag) or {}).get("secondary", {}).get("fpB", {}).get("C_det_minus_ctl") for m in ("27b", "4b")}}
    out["key_comparison"] = key
    out["reference_C"] = REF_C

    sc = {}
    g27, g4 = gate_of("27b"), gate_of("4b")
    sc["P1"] = GATE_LETTER.get(g27) if g27 else "pending"
    sc["P2"] = GATE_LETTER.get(g4) if g4 else "pending"
    sc["P3"] = key["a"]["verdict"] if not key["a"]["verdict"].startswith("PENDING") else "pending"
    ra = layer_res("27b", "a")
    for p, qn in (("P4", "Q1"), ("P5", "Q2"), ("P6", "Q3")):
        if g27 is not None and (g27 != "REPORTS" or out["models"]["27b"]["stopped_at_gate"]):
            sc[p] = "not scorable"
        elif ra is None:
            sc[p] = "pending"
        elif not ra["V3_ok"]:
            sc[p] = "not scorable"
        else:
            sc[p] = ra["questions"][qn]["letter"]
    out["prediction_outcomes"] = sc
    out["prediction_correct"] = {p: (None if sc[p] in ("pending", "not scorable") else sc[p] == PRED[p]) for p in PRED}

    md = ["# Study 17 — fingerprint in Gemma-3 (generated by scripts/study17_analyze.py)\n"]
    for m, r in out["models"].items():
        md.append(f"## {m}\n")
        g = r["gate"]
        md.append(f"Gate 1: {json.dumps(g) if g else 'not judged yet'}{' — STOPPED AT GATE' if r['stopped_at_gate'] else ''}\n")
        for tag, lr in r["layers"].items():
            x = lr["result"]
            if x is None:
                md.append(f"Layer ({tag}) block {lr['block']}: no test data\n")
                continue
            md.append(f"### layer ({tag}) = block {lr['block']} — {x['n_draws']} draws; V2 {x['V2_ok']}; V3 {x['V3_ok']} "
                      f"(positive control {x['V3_pos_yes_top']}% yes at s90)\n")
            md.append("| condition | dose | yes % | no % | neither % | broken? |\n|---|---|---|---|---|---|")
            for n, t in x["table"].items():
                for d in DOSES:
                    md.append(f"| {n} | {['0', 's25', 's50', 's90'][d]} | {t[d]['yes']:.1f} | {t[d]['no']:.1f} | {t[d]['neither']:.1f} | "
                              f"{'BROKEN' if t[d]['broken'] else 'ok'} |")
            md += ["", "```", json.dumps({"questions": x["questions"], "secondary": x["secondary"],
                                          "single_question_flag": x["single_question_flag"]}, indent=1), "```", ""]
    md += ["## KEY COMPARISON\n", "```", json.dumps({"key_comparison": key, "reference_C": REF_C}, indent=1), "```", "",
           "## Prediction scoring\n", "| # | researcher | outcome | correct? |", "|---|---|---|---|"]
    for p in PRED:
        c = out["prediction_correct"][p]
        md.append(f"| {p} | {PRED[p]} | {sc[p]} | {'—' if c is None else ('yes' if c else 'no')} |")
    O.mkdir(parents=True, exist_ok=True)
    (O / "study17_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (O / "study17_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\n".join(md).encode("ascii", "replace").decode()[-3000:])


if __name__ == "__main__":
    main()
