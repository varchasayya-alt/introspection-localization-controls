"""Study 16 analysis (gates/PREREG_study16.md): Study 15's primary analysis (yes-rates, rises, V1-V3,
Q1-Q3, prediction scoring) plus the pre-registered secondary outcome (yes - no logit shift at s90:
L_det, L_ctl, L_rand, differences with CIs, readings), per-control-question yes-rates with the
single-question flag, a leave-one-question-out Q2 check, and a 3B (Study 15) vs 8B table.
Reads results/study16/{test.jsonl, doses.json, direction.json} (+ results/study15_summary.json and
results/study15/test.jsonl for the comparison); writes results/study16_summary.md + .json.

Conventions fixed before any data (as Study 15): dose index 0 = clean, 1-3 = s25, s50, s90; yes-rate
= % of all trials; broken = "neither" > 20% (random pooled); a question is scored at the highest dose
index where all its conditions and dose 0 are valid; V3 = positive-control yes-rate at s90 >= 50% on
test data, else Q1-Q3 not scored. Secondary logit shifts use the top dose (s90) regardless of V1.
Bootstrap: 10,000 resamples over test draws, seed from the stimuli, percentile CIs.

Usage: python scripts/study16_analyze.py [--results DIR] [--out DIR]
"""
import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DOSES = [0, 1, 2, 3]
N_RANDOM = 5
BROKEN, V2_MIN_NO, V3_MIN_YES = 20.0, 80.0, 50.0
PRED = {"Q1": "b", "Q2": "b", "Q3": "b"}


def load(path):
    R = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            R[r["key"]] = r
    return R


def key(cond, k):
    return "k0" if k == 0 else f"{cond}|d{k}"


def q2_letter(d, mb):
    return "a" if (d >= 10 and mb < 10) else "b" if (d < 10 and mb >= 10) else "c" if (d >= 10 and mb >= 10) else "d"


def logit_shifts(R):
    """L_det, L_ctl, L_rand arrays at s90 (dose 3) for a Study 15/16-style test.jsonl."""
    draws = sorted({r["draw"] for r in R.values()})
    det = [R[f"det|d{d}"]["out"] for d in draws]
    ctl = [R[f"ctl|d{d}"]["out"] for d in draws]
    sh = lambda outs, c: np.array([o[f"{c}|d3"]["yn"] - o["k0"]["yn"] for o in outs])
    return sh(det, "inj"), sh(ctl, "inj"), np.mean([sh(det, f"rand{j}") for j in range(N_RANDOM)], axis=0), sh(det, "pos")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results" / "study16"))
    ap.add_argument("--out", default=str(REPO / "results"))
    args = ap.parse_args()
    RES, O = Path(args.results), Path(args.out)
    stim = json.loads((REPO / "stimuli" / "study16_stimuli.json").read_text(encoding="utf-8"))
    doses = json.loads((RES / "doses.json").read_text(encoding="utf-8"))
    dirn = json.loads((RES / "direction.json").read_text(encoding="utf-8"))
    fr = [0.0] + [doses["doses_fraction_of_N"][t] for t in ("25", "50", "90")]
    R = load(RES / "test.jsonl")
    draws = sorted({r["draw"] for r in R.values() if f"det|d{r['draw']}" in R and f"ctl|d{r['draw']}" in R})
    N = len(draws)
    det = [R[f"det|d{d}"]["out"] for d in draws]
    ctl = [R[f"ctl|d{d}"]["out"] for d in draws]
    cq = np.array([R[f"ctl|d{d}"]["control_q"] for d in draws])

    def arr(outs, cond, k, field):
        if field == "yn":
            return np.array([o[key(cond, k)]["yn"] for o in outs])
        return np.array([o[key(cond, k)]["cls"] == field for o in outs], dtype=float)

    conds = {"a_detection": (det, "inj"), "b_control": (ctl, "inj"), "d_positive": (det, "pos")}
    table = {}
    for name, (outs, c) in conds.items():
        table[name] = {k: {f: float(100 * arr(outs, c, k, f).mean()) for f in ("yes", "no", "neither")} for k in DOSES}
        for k in DOSES:
            table[name][k]["yn_shift"] = float((arr(outs, c, k, "yn") - arr(outs, c, 0, "yn")).mean())
    table["c_random"] = {}
    for k in DOSES:
        t = {f: float(100 * np.mean([arr(det, f"rand{j}", k, f) for j in range(N_RANDOM)])) for f in ("yes", "no", "neither")}
        t["yn_shift"] = float(np.mean([arr(det, f"rand{j}", k, "yn") - arr(det, f"rand{j}", 0, "yn") for j in range(N_RANDOM)]))
        table["c_random"][k] = t
    for name in table:
        for k in DOSES:
            table[name][k]["broken"] = table[name][k]["neither"] > BROKEN

    def rise(name, k, mask=None):
        if name == "c_random":
            x = 100 * np.mean([arr(det, f"rand{j}", k, "yes") - arr(det, f"rand{j}", 0, "yes") for j in range(N_RANDOM)], axis=0)
        else:
            outs, c = conds[name]
            x = 100 * (arr(outs, c, k, "yes") - arr(outs, c, 0, "yes"))
        return x if mask is None else x[mask]

    rng = np.random.RandomState(stim["bootstrap_seed"])
    BI = rng.randint(0, N, size=(10000, N))

    def ci(x):
        bs = x[BI].mean(1)
        return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]

    def scored_dose(names):
        for k in (3, 2, 1):
            if all(not table[n][k]["broken"] and not table[n][0]["broken"] for n in names):
                return k
        return None

    V2 = table["b_control"][0]["no"] >= V2_MIN_NO
    pos_top = table["d_positive"][3]["yes"]
    V3 = pos_top >= V3_MIN_YES
    out = {"n_draws": N, "layer": dirn["layer"], "typical_norm_N": dirn["typical_norm_N"], "dose_fractions": fr,
           "doses_collapsed": doses.get("collapsed"), "table": table, "V2_control_no_at_0": round(table["b_control"][0]["no"], 2),
           "V2_ok": V2, "V3_positive_yes_at_top": round(pos_top, 2), "V3_ok": V3}

    q = {}
    k1 = scored_dose(["a_detection"])
    if k1 is None:
        q["Q1"] = {"letter": "not scorable"}
    else:
        r = rise("a_detection", k1)
        m, c = float(r.mean()), ci(r)
        q["Q1"] = {"dose": k1, "rise": round(m, 2), "ci": c, "letter": "a" if m >= 20 else ("b" if m >= 5 else "c"),
                   "ci_crosses_boundary": any(c[0] < b <= c[1] for b in (5, 20))}
    k2 = scored_dose(["a_detection", "b_control"])
    if k2 is None or not V2:
        q["Q2"] = {"letter": "not scorable", "dose": k2, "V2_ok": V2}
    else:
        ra, rb = rise("a_detection", k2), rise("b_control", k2)
        d, mb = float((ra - rb).mean()), float(rb.mean())
        cd, cb = ci(ra - rb), ci(rb)
        loqo = {}
        for qi in range(10):
            mk = cq != qi
            loqo[stim["control_questions"][qi]] = q2_letter(float((ra[mk] - rb[mk]).mean()), float(rb[mk].mean()))
        q["Q2"] = {"dose": k2, "rise_detection": round(float(ra.mean()), 2), "rise_control": round(mb, 2),
                   "rise_control_ci": cb, "diff": round(d, 2), "diff_ci": cd, "letter": q2_letter(d, mb),
                   "ci_crosses_boundary": (cd[0] < 10 <= cd[1]) or (cb[0] < 10 <= cb[1]),
                   "leave_one_question_out_letters": loqo}
    k3 = scored_dose(["a_detection", "c_random"])
    if k3 is None:
        q["Q3"] = {"letter": "not scorable"}
    else:
        ra, rc = rise("a_detection", k3), rise("c_random", k3)
        d = float((ra - rc).mean())
        cd = ci(ra - rc)
        q["Q3"] = {"dose": k3, "rise_detection": round(float(ra.mean()), 2), "rise_random_mean": round(float(rc.mean()), 2),
                   "diff": round(d, 2), "diff_ci": cd, "letter": "b" if d >= 10 else ("c" if d <= -10 else "a"),
                   "ci_crosses_boundary": any(cd[0] < b <= cd[1] for b in (-10, 10))}
    out["questions"] = q

    # --- per-question control yes-rates and the single-question flag
    perq = {}
    for qi, qq in enumerate(stim["control_questions"]):
        mk = cq == qi
        perq[qq] = {"n": int(mk.sum()), "k0_yn": round(float(arr(ctl, "inj", 0, "yn")[mk].mean()), 3),
                    **{f"yes_d{k}": round(float(100 * arr(ctl, "inj", k, "yes")[mk].mean()), 1) for k in DOSES}}
    kq = k2 if k2 is not None else 3
    yes_counts = {qq: int(arr(ctl, "inj", kq, "yes")[cq == qi].sum()) for qi, qq in enumerate(stim["control_questions"])}
    tot = sum(yes_counts.values())
    top_q = max(yes_counts, key=yes_counts.get)
    out["per_question"] = perq
    out["single_question_flag"] = {"dose": kq, "total_control_yes": tot, "top_question": top_q,
                                   "top_share": round(yes_counts[top_q] / tot, 3) if tot else None,
                                   "single_question_driven": bool(tot and yes_counts[top_q] > tot / 2)}

    # --- secondary: logit shifts at s90
    Ld, Lc, Lr, Lp = logit_shifts(R)

    def reading_spec(dd, cdet, cctl):
        if cdet[0] <= 0 <= cdet[1] and cctl[0] <= 0 <= cctl[1]:
            return "no lean"
        return "detection-specific lean" if dd[0] > 0 else ("leans more on unrelated questions" if dd[1] < 0 else "equal lean")

    sec = {"L_det": [round(float(Ld.mean()), 3), ci(Ld)], "L_ctl": [round(float(Lc.mean()), 3), ci(Lc)],
           "L_rand": [round(float(Lr.mean()), 3), ci(Lr)], "L_pos": [round(float(Lp.mean()), 3), ci(Lp)],
           "L_det_minus_L_ctl": [round(float((Ld - Lc).mean()), 3), ci(Ld - Lc)],
           "L_det_minus_L_rand": [round(float((Ld - Lr).mean()), 3), ci(Ld - Lr)]}
    sec["reading_specificity"] = reading_spec(sec["L_det_minus_L_ctl"][1], sec["L_det"][1], sec["L_ctl"][1])
    dr = sec["L_det_minus_L_rand"][1]
    sec["reading_vs_random"] = "stronger than random" if dr[0] > 0 else ("weaker than random" if dr[1] < 0 else "no different from random")
    out["secondary_logit"] = sec

    # --- comparison with Study 15 (3B)
    cmp_ = None
    s15p = REPO / "results" / "study15_summary.json"
    if s15p.exists() and (REPO / "results" / "study15" / "test.jsonl").exists():
        s15 = json.loads(s15p.read_text(encoding="utf-8"))
        l15 = logit_shifts(load(REPO / "results" / "study15" / "test.jsonl"))
        cmp_ = {"3B": {"layer": f"block {s15['layer']} of 28", "doses": s15["dose_fractions"][1:],
                       "pos_yes": s15["table"]["d_positive"]["3"]["yes"], "det_yes": s15["table"]["a_detection"]["3"]["yes"],
                       "ctl_yes": s15["table"]["b_control"]["3"]["yes"], "rand_yes": s15["table"]["c_random"]["3"]["yes"],
                       "L": [round(float(x.mean()), 3) for x in l15],
                       "letters": [s15["questions"][k]["letter"] for k in ("Q1", "Q2", "Q3")]},
                "8B": {"layer": f"block {dirn['layer']} of 32", "doses": fr[1:], "pos_yes": table["d_positive"][3]["yes"],
                       "det_yes": table["a_detection"][3]["yes"], "ctl_yes": table["b_control"][3]["yes"],
                       "rand_yes": table["c_random"][3]["yes"], "L": [sec[k][0] for k in ("L_det", "L_ctl", "L_rand", "L_pos")],
                       "letters": [q[k]["letter"] for k in ("Q1", "Q2", "Q3")]}}
    out["comparison_3B_8B"] = cmp_

    # --- markdown
    nm = {"a_detection": "(a) detection, injected direction", "b_control": "(b) control questions, injected direction",
          "c_random": "(c) detection, random directions (mean of 5)", "d_positive": "(d) detection, yes − no direction"}
    md = ["# Study 16 — push the button on Llama-3.1-8B-Instruct (generated by scripts/study16_analyze.py)\n",
          f"{N} clean test draws; block {dirn['layer']}; typical norm N = {dirn['typical_norm_N']:.3f}; doses (fractions of N): "
          f"s25 = {fr[1]:.4f}, s50 = {fr[2]:.4f}, s90 = {fr[3]:.4f}{' (COLLAPSED)' if doses.get('collapsed') else ''}. "
          "Greedy first token, % of trials.\n",
          "| condition | dose | yes % | no % | neither % | broken? | yes − no logit shift |", "|---|---|---|---|---|---|---|"]
    for n in table:
        for k in DOSES:
            t = table[n][k]
            md.append(f"| {nm[n]} | {['0', 's25', 's50', 's90'][k]} | {t['yes']:.1f} | {t['no']:.1f} | {t['neither']:.1f} | "
                      f"{'**BROKEN**' if t['broken'] else 'ok'} | {t['yn_shift']:.2f} |")
    md += ["", f"V2 (control 'no' at dose 0 ≥ 80%): {out['V2_control_no_at_0']}% → {'ok' if V2 else 'FAIL'}",
           f"V3 (positive control 'yes' at s90 ≥ 50% on test): {out['V3_positive_yes_at_top']}% → "
           f"{'ok' if V3 else 'FAIL: study inconclusive, Q1–Q3 not scored'}", "",
           "## Pre-registered questions\n", "```", json.dumps(q, indent=1), "```", "",
           "## Secondary (pre-registered): yes − no logit shift at s90 (nats, [95% CI])\n", "```", json.dumps(sec, indent=1), "```", "",
           "## Control questions: 'yes' % by question and dose\n", "| question | n | clean yes−no logit | 0 | s25 | s50 | s90 |",
           "|---|---|---|---|---|---|---|"]
    for qq, v in perq.items():
        md.append(f"| {qq} | {v['n']} | {v['k0_yn']} | {v['yes_d0']} | {v['yes_d1']} | {v['yes_d2']} | {v['yes_d3']} |")
    md += ["", f"Single-question flag: {json.dumps(out['single_question_flag'])}", ""]
    if cmp_:
        md += ["## 3B (Study 15) vs 8B (Study 16), at the top dose\n",
               "| | 3B | 8B |", "|---|---|---|"]
        for lab, k in (("steering layer", "layer"), ("positive control yes %", "pos_yes"), ("detection yes %", "det_yes"),
                       ("control yes %", "ctl_yes"), ("random yes %", "rand_yes")):
            md.append(f"| {lab} | {cmp_['3B'][k]} | {cmp_['8B'][k]} |")
        md.append(f"| doses (× N) | {', '.join(f'{x:.4f}' for x in cmp_['3B']['doses'])} | {', '.join(f'{x:.4f}' for x in cmp_['8B']['doses'])} |")
        md.append(f"| logit shift det / ctl / random / positive | {' / '.join(map(str, cmp_['3B']['L']))} | {' / '.join(map(str, cmp_['8B']['L']))} |")
        md.append(f"| Q1 / Q2 / Q3 | {' / '.join(cmp_['3B']['letters'])} | {' / '.join(cmp_['8B']['letters'])} |")
        md.append("")
    md += ["## Prediction scoring\n", "| # | researcher | outcome | correct? |", "|---|---|---|---|"]
    score = {}
    for k, p in PRED.items():
        x = q[k]["letter"]
        score[k] = None if (x == "not scorable" or not V3) else x == p
        shown = x if V3 else f"{x} (not scored: V3 failed)"
        md.append(f"| {k} | {p} | {shown} | {'—' if score[k] is None else ('yes' if score[k] else 'no')} |")
    out["prediction_correct"] = score
    O.mkdir(parents=True, exist_ok=True)
    (O / "study16_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (O / "study16_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\n".join(md).encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
