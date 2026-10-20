"""Study 11 analysis: logit-lens verdicts and linear probes, by the rules in gates/study11_plan.md;
prediction scoring by the rules committed in predictions.md (c7eb51d). Reads the per-run lens values
from results/study11/<model>/trials.jsonl and the saved states from $INTROSPECTION_DATA/study11.
Writes results/study11_summary.md + .json.

Usage: python scripts/study11_analyze.py [--results DIR] [--data DIR] [--out DIR]
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

REPO = Path(__file__).resolve().parents[1]
MODELS = ("sft", "dpo", "final", "llama")
NAMES = {"sft": "Tülu SFT", "dpo": "Tülu DPO", "final": "Tülu final", "llama": "Llama-3.1-8B-Instruct"}
PEAK_LAYERS = range(8, 32)
BAR_NATS, SHRINK, AUC_BAR, L2 = 1.0, 0.25, 0.75, 0.05
PRED = {"P1": "a", "P2": "a", "P3": "a", "P4": "a", "P5": "a", "P6": "a"}


def auc(score, y):
    pos = [s for s, t in zip(score, y) if t]
    neg = [s for s, t in zip(score, y) if not t]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def fit_predict(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    w = torch.zeros(A.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    yt = torch.tensor(ytr, dtype=torch.float32)
    opt = torch.optim.LBFGS([w, b], max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(A @ w + b, yt) + L2 * (w @ w)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return (B @ w + b).tolist()


def bootstrap_ci(vals, seed, n=10000):
    rng = random.Random(seed)
    bs = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(n))
    return round(bs[int(0.025 * n)], 3), round(bs[int(0.975 * n) - 1], 3)


def analyse(name, R, Dstates, seed):
    rec = {(r["question"], r["draw_index"], r["injected"]): r for r in R}
    draws = sorted({r["draw_index"] for r in R})
    A = [d for d in draws if d < 40]
    B = [d for d in draws if d >= 40]
    shift = {q: {d: [i - n for i, n in zip(rec[(q, d, True)]["lens"], rec[(q, d, False)]["lens"])] for d in draws}
             for q in ("det", "ctl")}
    D = {d: [x - y for x, y in zip(shift["det"][d], shift["ctl"][d])] for d in draws}
    nl = len(D[draws[0]])
    mean = lambda ds, l: sum(D[d][l] for d in ds) / len(ds)
    peak = max(PEAK_LAYERS, key=lambda l: mean(A, l))
    dB, d32 = mean(B, peak), mean(B, nl - 1)
    ci = bootstrap_ci([D[d][peak] for d in B], seed)
    if dB >= BAR_NATS and ci[0] > 0:
        verdict = "SUPPRESSED" if d32 <= SHRINK * dB else "REACHES OUTPUT"
    else:
        verdict = "ABSENT"
    res = {"peak_layer_A": peak, "D_peak_B": round(dB, 3), "D_peak_B_ci": ci, "D_final_B": round(d32, 3),
           "verdict": verdict,
           "curve_all": {q: [round(sum(shift[q][d][l] for d in draws) / len(draws), 3) for l in range(nl)] for q in shift},
           "D_curve_all": [round(mean(draws, l), 3) for l in range(nl)]}
    # probes
    probes = {}
    for q in ("det", "ctl"):
        X = {d: {x: torch.load(Dstates / "states" / f"{q}_d{d}_{x}.pt", weights_only=True).float() for x in ("none", "inj")}
             for d in draws}
        rows = lambda ds, l: (torch.stack([X[d][x][l] for d in ds for x in ("none", "inj")]),
                              [x == "inj" for d in ds for x in ("none", "inj")])
        folds = [A[i::5] for i in range(5)]
        cv = []
        for l in range(nl):
            s_all, y_all = [], []
            for f in folds:
                tr = [d for d in A if d not in f]
                Xtr, ytr = rows(tr, l)
                Xte, yte = rows(f, l)
                s_all += fit_predict(Xtr, ytr, Xte)
                y_all += yte
            cv.append(auc(s_all, y_all))
        best = max(range(nl), key=lambda l: cv[l])
        test_curve = []
        for l in range(nl):
            Xtr, ytr = rows(A, l)
            Xte, yte = rows(B, l)
            test_curve.append(round(auc(fit_predict(Xtr, ytr, Xte), yte), 3))
        probes[q] = {"layer_A_cv": best, "cv_auc_A": round(cv[best], 3), "test_auc_B": test_curve[best],
                     "decodable": test_curve[best] >= AUC_BAR, "test_curve_B": test_curve}
    res["probes"] = probes
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results" / "study11"))
    ap.add_argument("--data", default=str(L.DATA_ROOT / "study11"))
    ap.add_argument("--out", default=str(REPO / "results"))
    args = ap.parse_args()
    RES, DATA, O = Path(args.results), Path(args.data), Path(args.out)
    seed = json.loads((REPO / "stimuli" / "study11_stimuli.json").read_text(encoding="utf-8"))["bootstrap_seed"]
    torch.manual_seed(0)
    md = ["# Study 11 — suppressed or absent? Logit lens and probes (generated by scripts/study11_analyze.py)\n",
          "Lens values in nats (yes minus no). D = detection-question yes-shift minus control-question yes-shift "
          "(injected minus none, paired by draw). Peak layer chosen on split A (layers 8–31), tested on split B.\n",
          "| model | sanity | peak layer (A) | D at peak (B) [95% CI] | D at output (B) | lens verdict | probe det.: layer, AUC (B) | probe ctl.: layer, AUC (B) |",
          "|---|---|---|---|---|---|---|---|"]
    out, allres = {}, {}
    for name in MODELS:
        st = json.loads((RES / name / "status.json").read_text()) if (RES / name / "status.json").exists() else {}
        R = load(RES / name / "trials.jsonl")
        if len(R) < 320:
            md.append(f"| {NAMES[name]} | {st.get('sanity', 'not run')} | not complete ({len(R)}/320) | | | | | |")
            continue
        r = analyse(name, R, DATA / name, seed)
        allres[name] = r
        p = r["probes"]
        md.append(f"| {NAMES[name]} | {st.get('sanity')} | {r['peak_layer_A']} | {r['D_peak_B']} {r['D_peak_B_ci']} | "
                  f"{r['D_final_B']} | **{r['verdict']}** | {p['det']['layer_A_cv']}, **{p['det']['test_auc_B']}** | "
                  f"{p['ctl']['layer_A_cv']}, **{p['ctl']['test_auc_B']}** |")
    md.append("")
    for name, r in allres.items():
        md.append(f"## {NAMES[name]}: curves (every 4th layer; lens shifts over all 80 draws; probe AUC on B)\n")
        md.append("| layer | " + " | ".join(str(l) for l in range(0, 33, 4)) + " |\n|---|" + "---|" * 9)
        md.append("| yes-shift, detection | " + " | ".join(str(r["curve_all"]["det"][l]) for l in range(0, 33, 4)) + " |")
        md.append("| yes-shift, control | " + " | ".join(str(r["curve_all"]["ctl"][l]) for l in range(0, 33, 4)) + " |")
        md.append("| D | " + " | ".join(str(r["D_curve_all"][l]) for l in range(0, 33, 4)) + " |")
        md.append("| probe AUC, detection | " + " | ".join(str(r["probes"]["det"]["test_curve_B"][l]) for l in range(0, 33, 4)) + " |")
        md.append("| probe AUC, control | " + " | ".join(str(r["probes"]["ctl"]["test_curve_B"][l]) for l in range(0, 33, 4)) + " |")
        md.append("")
    # --- prediction scoring (rules c7eb51d)
    letter = {"SUPPRESSED": "a", "REACHES OUTPUT": "b", "ABSENT": "c"}
    for q, nm in (("P1", "sft"), ("P2", "dpo"), ("P3", "final"), ("P6", "llama")):
        if nm in allres:
            out[q] = letter[allres[nm]["verdict"]]
    if all(n in allres for n in ("sft", "dpo", "final")):
        k = sum(allres[n]["probes"]["det"]["decodable"] for n in ("sft", "dpo", "final"))
        out["P4"] = "a" if k == 3 else ("b" if k else "c")
    if "final" in allres:
        dd = allres["final"]["probes"]["det"]["test_auc_B"] - allres["final"]["probes"]["ctl"]["test_auc_B"]
        out["P5"] = "a" if abs(dd) <= 0.10 else ("b" if dd > 0 else "c")
    md.append("## Prediction scoring (rules committed before the predictions, c7eb51d)\n")
    md.append("| # | researcher | outcome letter | correct? |\n|---|---|---|---|")
    for q in PRED:
        x = out.get(q, "pending")
        ok = "—" if x in ("pending", "not scorable") else ("yes" if x == PRED[q] else "no")
        md.append(f"| {q} | {PRED[q]} | {x} | {ok} |")
    md.append("")
    O.mkdir(parents=True, exist_ok=True)
    (O / "study11_summary.md").write_text("\n".join(md), encoding="utf-8")
    (O / "study11_summary.json").write_text(json.dumps({"results": allres, "prediction_outcomes": out}, indent=1), encoding="utf-8")
    print("\n".join(md).encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
