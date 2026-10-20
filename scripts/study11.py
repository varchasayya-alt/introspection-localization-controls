"""Study 11 — is the yes/no report suppressed or absent? Plan: gates/study11_plan.md.

Study 10's prompts and injection (scripts/study10.py), on 80 fresh draws with one injected position
each. For every run (detection / control question, with / without injection) the residual stream at
the answer position is saved at every layer (hidden_states[0..32], fp16, outside the repository),
and the logit-lens yes-minus-no value is computed per layer. Probes are fitted in the analysis.

Usage:
  python scripts/study11.py --model sft|dpo|final|llama --stage all|sanity|trials|summary
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
import study6 as S6  # noqa: E402
import study10 as T  # noqa: E402

CFG = {"layer": 2, "alpha": 5.0, "tol_logit": 1e-3, "tol_state_rel": 1e-3}
MODELS = {"sft": (7, "allenai/Llama-3.1-Tulu-3-8B-SFT"), "dpo": (8, "allenai/Llama-3.1-Tulu-3-8B-DPO"),
          "final": (6, "allenai/Llama-3.1-Tulu-3-8B"), "llama": (4, "meta-llama/Llama-3.1-8B-Instruct")}
STIM = L.REPO_ROOT / "stimuli" / "study11_stimuli.json"
OUT = LOG = REPO_RESULTS = None


def configure(name):
    global OUT, LOG, REPO_RESULTS
    OUT = L.DATA_ROOT / "study11" / name
    LOG = L.DATA_ROOT / "logs" / f"study11_{name}.log"
    REPO_RESULTS = L.REPO_ROOT / "results" / "study11" / name


def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    L.log(msg, LOG)


def load_vectors(name, vec_study):
    vstim = json.loads((L.REPO_ROOT / "stimuli" / f"study{vec_study}_stimuli.json").read_text(encoding="utf-8"))
    if name != "llama":
        return S6.load_tulu_vectors(vstim)  # checks the SHA-256 hashes
    S3.VEC_DIR = Path(vstim["vector_dir"].replace("<INTROSPECTION_DATA>", str(L.DATA_ROOT)))
    for f, h in vstim["vector_sha256"].items():
        assert hashlib.sha256((S3.VEC_DIR / f).read_bytes()).hexdigest() == h, f
    return S3.load_their_vectors(vstim["concepts"])


class Ctx11:
    def __init__(self, name):
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "states").mkdir(exist_ok=True)
        vec_study, repo = MODELS[name]
        rev = json.loads((L.REPO_ROOT / "stimuli" / "model_pins.json").read_text())[repo]
        t0 = time.time()
        self.model, self.tok = L.load_model(repo, rev)
        log(f"loaded {repo}@{rev} in {time.time() - t0:.0f}s")
        self.stim = json.loads(STIM.read_text(encoding="utf-8"))
        self.vecs = load_vectors(name, vec_study)
        self.y, self.n = T.yes_no_ids(self.tok)
        with torch.inference_mode():
            self.W_y = self.model.lm_head.weight[self.y].float()
            self.W_n = self.model.lm_head.weight[self.n].float()

    def prompts(self, d):
        spans = S3.marker_spans(d["sentences"])
        ctrl = self.stim["control_questions"][d["control_q"]] + " Answer yes or no."
        return T.P10(self.tok, spans, T.Q_DETECT), T.P10(self.tok, spans, ctrl)

    @torch.inference_mode()
    def lens(self, states, logits):
        """Yes-minus-no (log-sum-exp over the variants) per layer: lm_head(final_norm(h)) for
        layers 0..31, the real output logits for layer 32."""
        h = L.final_norm(self.model)(states[:-1].float().unsqueeze(0))[0]
        ly, ln = h @ self.W_y.T, h @ self.W_n.T
        out = (torch.logsumexp(ly, -1) - torch.logsumexp(ln, -1)).tolist()
        out.append(float(torch.logsumexp(logits[self.y], -1) - torch.logsumexp(logits[self.n], -1)))
        return out


@torch.inference_mode()
def full_states(model, P):
    out = model(input_ids=P.tensor(), use_cache=True, output_hidden_states=True, logits_to_keep=1)
    return out.logits[0, -1].float(), torch.stack([h[0, -1] for h in out.hidden_states]), out.past_key_values


@torch.inference_mode()
def suffix_states(model, P, start, cache, vec, idx):
    import copy
    c = copy.deepcopy(cache)
    n_remove = c.get_seq_length() - start
    if n_remove > 0:
        c.crop(-n_remove)
    with L.inject(model, CFG["layer"], vec, [p - start for p in idx]):
        out = model(input_ids=P.tensor()[:, start:], past_key_values=c, use_cache=True,
                    output_hidden_states=True, logits_to_keep=1)
    return out.logits[0, -1].float(), torch.stack([h[0, -1] for h in out.hidden_states])


@torch.inference_mode()
def full_states_injected(model, P, vec, idx):
    with L.inject(model, CFG["layer"], vec, idx):
        out = model(input_ids=P.tensor(), use_cache=False, output_hidden_states=True, logits_to_keep=1)
    return out.logits[0, -1].float(), torch.stack([h[0, -1] for h in out.hidden_states])


def write_status(**kw):
    path = OUT / "status.json"
    st = json.loads(path.read_text()) if path.exists() else {}
    st.update(kw)
    st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(json.dumps(st, indent=2))
    return st


def stage_sanity(ctx):
    m, res, a = ctx.model, {}, CFG["alpha"]
    pd, pc = ctx.prompts(ctx.stim["draws"][0])
    assert pd.text.count(S6.BOS) == 1 and pd.ids[: pd.user_tok[0][0]] == pc.ids[: pc.user_tok[0][0]]
    res["S1_n_tokens"] = [len(pd.ids), len(pc.ids)]
    res["S2_yes_ids"], res["S2_no_ids"] = ctx.y, ctx.n
    for P in (pd, pc):
        for k in range(10):
            lab, sen = set(P.arm[k]["label"]), set(P.arm[k]["sentence"])
            assert lab and sen and not (lab & sen) and sorted(lab | sen) == P.user_tok[k], k
    idx = pd.arm[3]["sentence"]
    lg0, st0, cache = full_states(m, pd)
    lg1, st1 = full_states_injected(m, pd, torch.zeros(m.config.hidden_size), idx)
    res["S4_alpha0_max_abs_diff"] = float((lg0 - lg1).abs().max())
    assert res["S4_alpha0_max_abs_diff"] == 0.0
    rv = L.random_unit_vectors(m.config.hidden_size, 1, seed=999)[0] * a
    lf, sf = full_states_injected(m, pd, rv, idx)
    ls, ss = suffix_states(m, pd, idx[0], cache, rv, idx)
    yn = ctx.y + ctx.n
    res["S5_maxdiff_yesno"] = float((lf[yn] - ls[yn]).abs().max())
    res["S5_layer16_rel"] = float((sf[16].float() - ss[16].float()).norm() / sf[16].float().norm())
    assert res["S5_maxdiff_yesno"] < CFG["tol_logit"] and res["S5_layer16_rel"] < CFG["tol_state_rel"]
    lens = ctx.lens(st0, lg0)
    # hidden_states[32] must already be normalised: unembedding it directly reproduces the output
    h_final = st0[32].float()
    direct = float(torch.logsumexp(h_final @ ctx.W_y.T, -1) - torch.logsumexp(h_final @ ctx.W_n.T, -1))
    res["S9_lens_final_vs_output"] = abs(direct - lens[32])
    assert res["S9_lens_final_vs_output"] < CFG["tol_logit"]
    res["S6_vector_hashes_ok"] = True
    (OUT / "sanity.json").write_text(json.dumps(res, indent=2))
    log(f"SANITY PASSED: S4 0.0, S5 {res['S5_maxdiff_yesno']:.1e} / {res['S5_layer16_rel']:.1e}, "
        f"S9 {res['S9_lens_final_vs_output']:.1e}")
    write_status(sanity="PASS")


def stage_trials(ctx):
    m, a = ctx.model, CFG["alpha"]
    st = L.ResultStore(OUT / "trials.jsonl")
    t0 = time.time()
    for di, d in enumerate(ctx.stim["draws"]):
        keys = [f"{q}|d{di}|{x}" for q in ("det", "ctl") for x in ("none", "inj")]
        if all(st.has(k) for k in keys):
            continue
        c, t, pos = d["concept"], d["vec_type"], d["position"]
        meta = {"draw_index": di, "split": "A" if di < 40 else "B", "concept": c, "vec_type": t,
                "control_q": d["control_q"], "pos": pos}
        vec = a * ctx.vecs[(c, t)]
        for q, P in zip(("det", "ctl"), ctx.prompts(d)):
            lg, sts, cache = full_states(m, P)
            for x in ("none", "inj"):
                key = f"{q}|d{di}|{x}"
                if st.has(key):
                    continue
                t1 = time.time()
                if x == "inj":
                    idx = P.arm[pos - 1]["sentence"]
                    lg, sts = suffix_states(m, P, idx[0], cache, vec, idx)
                torch.save(sts.to(torch.float16).clone(), OUT / "states" / f"{q}_d{di}_{x}.pt")
                py, mass = T.p_yes(lg, ctx.y, ctx.n)
                st.add(dict(meta, key=key, question=q, injected=x == "inj", p_yes=round(py, 6),
                            yes_no_mass=round(mass, 6), lens=[round(v, 5) for v in ctx.lens(sts, lg)],
                            sec=round(time.time() - t1, 2)))
        log(f"draw {di + 1}/80 done ({c}|{t}|pos{pos}); {(time.time() - t0) / 60:.1f} min")
    write_status(complete=len(st.records) == 320)
    log(f"TRIALS DONE: {len(st.records)}/320 runs")


def stage_summary():
    REPO_RESULTS.mkdir(parents=True, exist_ok=True)
    for f in sorted(OUT.glob("*.json")) + sorted(OUT.glob("*.jsonl")):
        shutil.copy2(f, REPO_RESULTS / f.name)
    log(f"copied results to results/study11/{REPO_RESULTS.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--stage", required=True, choices=["all", "sanity", "trials", "summary"])
    args = ap.parse_args()
    configure(args.model)
    if args.stage == "summary":
        stage_summary()
        return
    ctx = Ctx11(args.model)
    if args.stage in ("all", "sanity"):
        stage_sanity(ctx)
    if args.stage in ("all", "trials"):
        stage_trials(ctx)
    stage_summary()
    log("DONE")


if __name__ == "__main__":
    main()
