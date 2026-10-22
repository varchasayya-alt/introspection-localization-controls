"""Study 13, deviation D6 (logs/deviations.md): Llama-3.2-1B localization with S7 counting spans 2-10.

Runs after the main Study 13 queue, with the frozen study13.py code (unchanged): same vectors,
strength, draws, arms A (+ copy-back/reader) and C, and the same main test (target-span exactness).
Only the S7 gate differs: a clean copy must reproduce spans 2-10 exactly in >= 19/20 draws.
Outputs go to $INTROSPECTION_DATA/study13/<setting>/d6/ and results/study13/<setting>_d6/.

Usage: python scripts/study13_d6.py --setting 1b_l1|1b_l2
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import introspect_lib as L  # noqa: E402
import study4 as S4  # noqa: E402
import study13 as Z  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True, choices=["1b_l1", "1b_l2"])
    ap.add_argument("--results-root", default=str(L.REPO_ROOT / "results"))
    args = ap.parse_args()
    Z.configure(args.setting)
    base_out = Z.OUT
    ctx = Z.Ctx13(args.setting)
    Z.stage_vectors(ctx)      # no-ops if the main queue already made them
    Z.stage_calibrate(ctx)
    ctx.load_vectors_and_alpha()
    d6 = base_out / "d6"
    d6.mkdir(parents=True, exist_ok=True)
    ok, bad = 0, []
    for di, d in enumerate(ctx.stim["loc_draws"]):
        spans, _, cp, _, _ = ctx.loc_prompts(d["sentences"])
        cl = L.clean_pass(ctx.model, cp, want_final=True)
        perc, _ = S4.perceived_with_fix(ctx.model, ctx.tok, cp, {"final": cl["final"]}, 0)
        if all(perc[j] == spans[j] for j in range(1, 10)):
            ok += 1
        else:
            bad.append({"draw": di, "diffs": [(j + 1, perc[j]) for j in range(1, 10) if perc[j] != spans[j]]})
    res = {"deviation": "D6", "S7_spans_2_10_exact_draws": ok, "S7_failures": bad, "localization_ok": ok >= Z.CFG["s7_min_draws"]}
    (d6 / "sanity_d6.json").write_text(json.dumps(res, indent=2))
    Z.log(f"D6 SANITY: S7 (spans 2-10) {ok}/20 -> localization {'runs' if res['localization_ok'] else 'STOPPED'}")
    if res["localization_ok"]:
        Z.OUT = d6  # stores and status for the deviation run go to the d6 folder
        Z.stage_localization(ctx)
    dest = Path(args.results_root) / "study13" / f"{args.setting}_d6"
    dest.mkdir(parents=True, exist_ok=True)
    for f in sorted(d6.glob("*.json")) + sorted(d6.glob("*.jsonl")):
        shutil.copy2(f, dest / f.name)
    Z.log(f"D6 DONE: results copied to results/study13/{args.setting}_d6")


if __name__ == "__main__":
    main()
