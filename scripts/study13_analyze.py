"""Study 13 analysis: per-model localization main test, yes/no report, probe AUC, and the size-trend
rules in gates/study13_plan.md §4; prediction scoring by the rules committed in predictions.md
(d99992c). 8B reference reused from Study 4 (localization) and Study 11 (yes/no, probes).
Writes results/study13_summary.md + .json.

Usage: python scripts/study13_analyze.py [--results DIR] [--data DIR] [--out DIR]
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import introspect_lib as L  # noqa: E402
from gate1_analyze import load  # noqa: E402
from study11_analyze import auc, fit_predict  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SETTINGS = {"1b_l1": "Llama-3.2-1B, layer 1 (primary)", "3b_l2": "Llama-3.2-3B, layer 2 (primary)",
            "1b_l2": "Llama-3.2-1B, layer 2 (secondary)"}
PRED = {"P1": "c", "P2": "b", "P3": "b", "P4": "a", "P5": "a", "P6": "a"}


def report_stats(R, seed):
    rec = {(r["question"], r["draw_index"], r["injected"]): r["p_yes"] for r in R}
    draws = sorted({r["draw_index"] for r in R})
    per = [100 * (rec[("det", d, True)] - rec[("det", d, False)]) for d in draws]
    rng = random.Random(seed)
    bs = sorted(sum(rng.choice(per) for _ in per) / len(per) for _ in range(10000))
    m = lambda q, i: 100 * sum(rec[(q, d, i)] for d in draws) / len(draws)
    return {"hits": round(m("det", True), 2), "fa": round(m("det", False), 2), "report": round(sum(per) / len(per), 2),
            "report_ci": [round(bs[250], 2), round(bs[9749], 2)], "ctl_inj": round(m("ctl", True), 2),
            "ctl_none": round(m("ctl", False), 2)}


def probe(states_dir, R):
    draws = sorted({r["draw_index"] for r in R})
    A, B = [d for d in draws if d < 40], [d for d in draws if d >= 40]
    out = {}
    for q in ("det", "ctl"):
        X = {d: {x: torch.load(states_dir / f"{q}_d{d}_{x}.pt", weights_only=True).float() for x in ("none", "inj")} for d in draws}
        nl = X[draws[0]]["none"].shape[0]
        rows = lambda ds, l: (torch.stack([X[d][x][l] for d in ds for x in ("none", "inj")]),
                              [x == "inj" for d in ds for x in ("none", "inj")])
        folds = [A[i::5] for i in range(5)]
        cv = []
        for l in range(nl):
            s_all, y_all = [], []
            for f in folds:
                Xtr, ytr = rows([d for d in A if d not in f], l)
                Xte, yte = rows(f, l)
                s_all += fit_predict(Xtr, ytr, Xte)
                y_all += yte
            cv.append(auc(s_all, y_all))
        best = max(range(nl), key=lambda l: cv[l])
        Xtr, ytr = rows(A, best)
        Xte, yte = rows(B, best)
        a = round(auc(fit_predict(Xtr, ytr, Xte), yte), 3)
        out[q] = {"layer": best, "auc_B": a, "decodable": a >= 0.75}
    return out


def trend(v1, v3, v8, rise, step, flat):
    if None in (v1, v3, v8):
        return "not testable"
    if v8 - v1 >= rise and v3 >= v1 - step and v8 >= v3 - step:
        return "rises"
    if max(v1, v3, v8) - min(v1, v3, v8) <= flat:
        return "flat"
    return "mixed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results"))
    ap.add_argument("--data", default=str(L.DATA_ROOT / "study13"))
    ap.add_argument("--out", default=str(REPO / "results"))
    args = ap.parse_args()
    RES, DATA, O = Path(args.results), Path(args.data), Path(args.out)
    seed = json.loads((REPO / "stimuli" / "study13_stimuli.json").read_text(encoding="utf-8"))["bootstrap_seed"]
    torch.manual_seed(0)
    rows, allres = [], {}
    for s, label in SETTINGS.items():
        D = RES / "study13" / s
        st = json.loads((D / "status.json").read_text()) if (D / "status.json").exists() else {}
        cal = json.loads((D / "calibration.json").read_text()) if (D / "calibration.json").exists() else {}
        R = load(D / "yesno.jsonl")
        r = {"label": label, "sanity": st.get("sanity"), "calibration": cal, "loc": st.get("localization")}
        if len(R) == 320:
            r["yesno"] = report_stats(R, seed)
            r["probe"] = probe(DATA / s / "states", R)
        allres[s] = r
    # 8B reference
    s4 = json.loads((RES / "study4" / "status.json").read_text())["result"]
    R8 = load(RES / "study11" / "llama" / "trials.jsonl")
    p8 = json.loads((RES / "study11_summary.json").read_text())["results"]["llama"]["probes"]
    allres["8b_l2"] = {"label": "Llama-3.1-8B, layer 2 (reference: Studies 4, 11)", "sanity": "PASS",
                       "calibration": {"layer": 2, "mean_norm": 2.287, "alpha": 5.0, "ratio": 2.186},
                       "loc": {"A": s4["inj_k"]["A"], "C": s4["inj_k"]["C"], "reader": s4["reader_k"]["A"],
                               "exact_n": s4["A_exact_n"], "exact_inj": s4["A_exact_inj_k"],
                               "exact_reader": s4["A_exact_reader_k"], "main_test": s4["main_test"],
                               "intact_score_pct": round(100 * s4["A_exact_inj_k"] / s4["A_exact_n"], 1)},
                       "yesno": report_stats(R8, seed),
                       "probe": {q: {"layer": p8[q]["layer_A_cv"], "auc_B": p8[q]["test_auc_B"], "decodable": p8[q]["decodable"]}
                                 for q in ("det", "ctl")}}
    md = ["# Study 13 — size curve in the Llama family (generated by scripts/study13_analyze.py)\n",
          "| setting | layer, α (ratio) | arm A (concept) | reader | exact copies: injected / reader | main test | intact score | arm C (random) | hits | FA | **report** [95% CI] | control yes (inj/none) | probe AUC det / ctl |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in ("1b_l1", "3b_l2", "8b_l2", "1b_l2"):
        r = allres.get(s, {})
        cal, lo, yn, pr = r.get("calibration") or {}, r.get("loc") or {}, r.get("yesno") or {}, r.get("probe") or {}
        md.append(f"| {r.get('label', s)} | {cal.get('layer')}, {round(cal.get('alpha', 0), 2)} ({cal.get('ratio')}) | "
                  f"{lo.get('A', '—')}/200 | {lo.get('reader', '—')} | {lo.get('exact_n', '—')}: {lo.get('exact_inj', '—')} / {lo.get('exact_reader', '—')} | "
                  f"**{lo.get('main_test', '—')}** | {lo.get('intact_score_pct')} | {lo.get('C', '—')}/200 | {yn.get('hits')} | {yn.get('fa')} | "
                  f"**{yn.get('report')}** {yn.get('report_ci')} | {yn.get('ctl_inj')} / {yn.get('ctl_none')} | "
                  f"{pr.get('det', {}).get('auc_B')} / {pr.get('ctl', {}).get('auc_B')} |")
    g = lambda s, k1, k2: (allres.get(s, {}).get(k1) or {}).get(k2)
    t_loc = trend(g("1b_l1", "loc", "intact_score_pct"), g("3b_l2", "loc", "intact_score_pct"), g("8b_l2", "loc", "intact_score_pct"), 20, 5, 10)
    t_rep = trend(g("1b_l1", "yesno", "report"), g("3b_l2", "yesno", "report"), g("8b_l2", "yesno", "report"), 10, 3, 5)
    md.append(f"\n**Size trend, intact-text localization: {t_loc}. Size trend, report (hits − FA): {t_rep}.**\n")
    # --- prediction scoring (rules d99992c)
    out = {}
    bins = lambda a: "a" if a >= 150 else "b" if a >= 100 else "c" if a >= 50 else "d"
    mt = lambda lo: {"PASS": "a", "NOT TESTABLE": "d"}.get(lo["main_test"], "b" if lo.get("main_fail_reason") == "injected < 50%" else "c")
    for qa, qm, s in (("P1", "P2", "1b_l1"), ("P3", "P4", "3b_l2")):
        lo = g(s, "loc", "A")
        if lo is not None:
            out[qa], out[qm] = bins(lo), mt(allres[s]["loc"])
        elif allres.get(s, {}).get("sanity"):
            out[qa] = out[qm] = "not scorable"
    out["P5"] = {"rises": "a", "flat": "b"}.get(t_loc, "c")
    out["P6"] = {"rises": "a", "flat": "b"}.get(t_rep, "c")
    md.append("## Prediction scoring (rules committed before the predictions, d99992c)\n")
    md.append("| # | researcher | outcome letter | correct? |\n|---|---|---|---|")
    for q in PRED:
        x = out.get(q, "pending")
        ok = "—" if x in ("pending", "not scorable") else ("yes" if x == PRED[q] else "no")
        md.append(f"| {q} | {PRED[q]} | {x} | {ok} |")
    md.append("")
    O.mkdir(parents=True, exist_ok=True)
    (O / "study13_summary.md").write_text("\n".join(md), encoding="utf-8")
    (O / "study13_summary.json").write_text(json.dumps({"results": allres, "trend_localization": t_loc,
                                                        "trend_report": t_rep, "prediction_outcomes": out}, indent=1), encoding="utf-8")
    print("\n".join(md).encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
