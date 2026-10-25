"""Study 14 analysis: yes-rates by condition and dose, rises with bootstrap CIs, validity rules V1-V3,
Q1-Q3 by the bars in gates/PREREG_study14.md §6, and prediction scoring (predictions.md, Study 14).
Reads results/study14/{test.jsonl, calibration.json, sanity.json}; writes results/study14_summary.md
+ .json.

Conventions fixed before any test data:
- yes-rate = % of trials whose greedy first token is a yes token (out of all trials, "neither"
  counts as not-yes); no-rate and neither-rate likewise.
- broken (V1): neither-rate > 20% for that condition at that dose; random is pooled over its 5
  directions. The k = 0 run is shared by (a), (c), (d); if it is broken, nothing on the detection
  question is scorable.
- scored dose for a question = the highest k in {8, 4, 2, 1} at which every condition entering the
  question is valid (and k = 0 too); none -> not scorable.
- V3 uses the positive control's rise at Q1's scored dose (whatever its own neither-rate, which is
  reported beside it).
- CIs: 10,000 bootstrap resamples over test draws (seed from the stimuli), percentile 2.5 / 97.5.

Usage: python scripts/study14_analyze.py [--results DIR] [--out DIR]
"""
import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DOSES = [0, 1, 2, 4, 8]
N_RANDOM = 5
BROKEN, V2_MIN_NO, V3_MIN = 20.0, 80.0, 20.0
PRED = {"Q1": "a", "Q2": "b", "Q3": "b"}


def load(path):
    R = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            R[r["key"]] = r
    return R


def key(cond, k):
    return "k0" if k == 0 else f"{cond}|k{k}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results" / "study14"))
    ap.add_argument("--out", default=str(REPO / "results"))
    args = ap.parse_args()
    RES, O = Path(args.results), Path(args.out)
    stim = json.loads((REPO / "stimuli" / "study14_stimuli.json").read_text(encoding="utf-8"))
    cal = json.loads((RES / "calibration.json").read_text(encoding="utf-8"))
    R = load(RES / "test.jsonl")
    draws = sorted({r["draw"] for r in R.values() if f"det|d{r['draw']}" in R and f"ctl|d{r['draw']}" in R})
    N = len(draws)
    det = [R[f"det|d{d}"]["out"] for d in draws]
    ctl = [R[f"ctl|d{d}"]["out"] for d in draws]

    # per-trial arrays: cond -> dose -> (N,) or (N, 5) for random
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
        t["yn_shift"] = float(np.mean([(arr(det, f"rand{j}", k, "yn") - arr(det, f"rand{j}", 0, "yn")) for j in range(N_RANDOM)]))
        table["c_random"][k] = t
    for name in table:
        for k in DOSES:
            table[name][k]["broken"] = table[name][k]["neither"] > BROKEN

    # per-draw rise vectors at dose k (points)
    def rise(name, k):
        if name == "c_random":
            return 100 * np.mean([arr(det, f"rand{j}", k, "yes") - arr(det, f"rand{j}", 0, "yes") for j in range(N_RANDOM)], axis=0)
        outs, c = conds[name]
        return 100 * (arr(outs, c, k, "yes") - arr(outs, c, 0, "yes"))

    rng = np.random.RandomState(stim["bootstrap_seed"])
    BI = rng.randint(0, N, size=(10000, N))

    def ci(x):
        bs = x[BI].mean(1)
        return [round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]

    def scored_dose(names):
        for k in (8, 4, 2, 1):
            if all(not table[n][k]["broken"] and not table[n][0]["broken"] for n in names):
                return k
        return None

    out = {"n_draws": N, "steering_layer": cal["steering_layer"], "auc_at_layer": cal["auc_at_layer"],
           "direction_norm": cal["direction_norm"], "direction_norm_ratio": cal["direction_norm_ratio"], "table": table}
    V2 = table["b_control"][0]["no"] >= V2_MIN_NO
    out["V2_control_no_at_k0"] = round(table["b_control"][0]["no"], 2)
    out["V2_ok"] = V2

    # Q1
    k1 = scored_dose(["a_detection"])
    q = {}
    if k1 is None:
        q["Q1"] = {"letter": "not scorable"}
    else:
        r = rise("a_detection", k1)
        m, c = float(r.mean()), ci(r)
        letter = "a" if m >= 20 else ("b" if m >= 5 else "c")
        pr = rise("d_positive", k1)
        v3 = float(pr.mean()) >= V3_MIN
        q["Q1"] = {"dose": k1, "rise": round(m, 2), "ci": c, "letter": letter,
                   "ci_crosses_boundary": any(c[0] < b <= c[1] for b in (5, 20)),
                   "positive_control_rise": round(float(pr.mean()), 2), "positive_control_ci": ci(pr),
                   "positive_control_neither_pct": round(table["d_positive"][k1]["neither"], 2), "V3_ok": v3,
                   "inconclusive_push_too_weak": (letter == "c" and not v3)}
    # Q2
    k2 = scored_dose(["a_detection", "b_control"])
    if k2 is None or not V2:
        q["Q2"] = {"letter": "not scorable", "dose": k2, "V2_ok": V2}
    else:
        ra, rb = rise("a_detection", k2), rise("b_control", k2)
        d, mb = float((ra - rb).mean()), float(rb.mean())
        if d >= 10 and mb < 10:
            letter = "a"
        elif d < 10 and mb >= 10:
            letter = "b"
        elif d >= 10 and mb >= 10:
            letter = "c"
        else:
            letter = "d"
        cd, cb = ci(ra - rb), ci(rb)
        q["Q2"] = {"dose": k2, "rise_detection": round(float(ra.mean()), 2), "rise_control": round(mb, 2),
                   "rise_control_ci": cb, "diff": round(d, 2), "diff_ci": cd, "letter": letter,
                   "ci_crosses_boundary": (cd[0] < 10 <= cd[1]) or (cb[0] < 10 <= cb[1])}
    # Q3
    k3 = scored_dose(["a_detection", "c_random"])
    if k3 is None:
        q["Q3"] = {"letter": "not scorable"}
    else:
        ra, rc = rise("a_detection", k3), rise("c_random", k3)
        d = float((ra - rc).mean())
        letter = "b" if d >= 10 else ("c" if d <= -10 else "a")
        cd = ci(ra - rc)
        q["Q3"] = {"dose": k3, "rise_detection": round(float(ra.mean()), 2), "rise_random_mean": round(float(rc.mean()), 2),
                   "rise_random_per_direction": [round(float(100 * (arr(det, f"rand{j}", k3, "yes") - arr(det, f"rand{j}", 0, "yes")).mean()), 2)
                                                 for j in range(N_RANDOM)],
                   "diff": round(d, 2), "diff_ci": cd, "letter": letter,
                   "ci_crosses_boundary": any(cd[0] < b <= cd[1] for b in (-10, 10))}
    out["questions"] = q
    out["rises_all_doses"] = {n: {k: [round(float(rise(n, k).mean()), 2), ci(rise(n, k))] for k in DOSES[1:]} for n in table}

    # --- markdown
    nm = {"a_detection": "(a) detection, injected direction", "b_control": "(b) control questions, injected direction",
          "c_random": "(c) detection, random directions (mean of 5)", "d_positive": "(d) detection, yes − no direction"}
    md = ["# Study 14 — push the button (generated by scripts/study14_analyze.py)\n",
          f"Llama-3.2-3B-Instruct; {N} clean test draws; steering layer {cal['steering_layer']} (held-out probe AUC "
          f"{cal['auc_at_layer']}); |v| = {cal['direction_norm']:.3f} ({cal['direction_norm_ratio']:.3f} × mean residual norm "
          "at the answer position). Greedy first token, % of trials.\n",
          "| condition | k | yes % | no % | neither % | broken? | yes − no logit shift |", "|---|---|---|---|---|---|---|"]
    for n in table:
        for k in DOSES:
            t = table[n][k]
            md.append(f"| {nm[n]} | {k} | {t['yes']:.1f} | {t['no']:.1f} | {t['neither']:.1f} | {'**BROKEN**' if t['broken'] else 'ok'} | {t['yn_shift']:.2f} |")
    md += ["", "## Rise in yes-rate from k = 0 (points, [95% CI])\n", "| condition | k=1 | k=2 | k=4 | k=8 |", "|---|---|---|---|---|"]
    for n in table:
        md.append(f"| {nm[n]} | " + " | ".join(f"{out['rises_all_doses'][n][k][0]} {out['rises_all_doses'][n][k][1]}" for k in DOSES[1:]) + " |")
    md += ["", f"V2 (control answered 'no' at k = 0 ≥ 80%): {out['V2_control_no_at_k0']}% → {'ok' if V2 else 'FAIL'}", "",
           "## Pre-registered questions\n", "```", json.dumps(q, indent=1), "```", "",
           "## Prediction scoring\n", "| # | researcher | outcome | correct? |", "|---|---|---|---|"]
    score = {}
    for k, p in PRED.items():
        x = q[k]["letter"]
        score[k] = None if x == "not scorable" else x == p
        md.append(f"| {k} | {p} | {x} | {'—' if score[k] is None else ('yes' if score[k] else 'no')} |")
    out["prediction_correct"] = score
    O.mkdir(parents=True, exist_ok=True)
    (O / "study14_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (O / "study14_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\n".join(md).encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
