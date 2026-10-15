"""Gate 0 hardware timing: seconds per forward pass (~300 tokens) on this CPU.

Builds each model from its config.json only (random weights, no weight download).
Speed depends on architecture shape, not weight values, so this measures the real
cost of the actual 1B / 3B / 7B architectures we would use.

Usage: python scripts/gate0_timing.py [--repos ...] [--dtypes bf16 fp32] [--threads 12]
"""
import argparse
import gc
import json
import platform
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM

# "bf16w": weights STORED in bf16 (half the RAM), but every matmul is done in fp32 by
# up-casting each Linear's weight just before use. This CPU (AVX2, no AVX512-BF16/AMX)
# runs native bf16 matmuls much slower than fp32.
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32, "bf16w": torch.bfloat16}


def _fp32_compute_linear_forward(self, x):
    w = self.weight.float()
    b = self.bias.float() if self.bias is not None else None
    return torch.nn.functional.linear(x, w, b)


def make_bf16_storage_fp32_compute(model):
    """Keep Linear weights in bf16; cast everything else (embeddings, norms) to fp32 so the
    residual stream is fp32; Linear layers up-cast their weight on the fly."""
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            m.forward = _fp32_compute_linear_forward.__get__(m, torch.nn.Linear)
        else:
            for name, p in list(m.named_parameters(recurse=False)):
                p.data = p.data.float()
    return model


def build_random(repo, dtype, dname="bf16"):
    cfg = AutoConfig.from_pretrained(repo)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
    model = model.to_empty(device="cpu")
    # Fill with small normal values: uninitialised memory can hold NaNs/denormals
    # that make CPU math artificially slow.
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.02)
        for b in model.buffers():
            if b.is_floating_point():
                b.zero_()
    # Rotary inv_freq buffers must be recomputed after meta init.
    for m in model.modules():
        if hasattr(m, "inv_freq") and hasattr(m, "rope_init_fn"):
            inv_freq, scale = m.rope_init_fn(m.config, "cpu")
            m.inv_freq = inv_freq
            m.original_inv_freq = inv_freq
            m.attention_scaling = scale
    if dname == "bf16w":
        make_bf16_storage_fp32_compute(model)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, cfg, n_params


def time_forward(model, vocab, seq_len, reps, warmup):
    ids = torch.randint(0, min(vocab, 30000), (1, seq_len))
    with torch.inference_mode():
        for _ in range(warmup):
            model(input_ids=ids, use_cache=False)
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            model(input_ids=ids, use_cache=False)
            times.append(time.perf_counter() - t0)
    times.sort()
    return {"median_s": times[len(times) // 2], "min_s": times[0], "max_s": times[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="+", required=True)
    ap.add_argument("--dtypes", nargs="+", default=["bf16"])
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--seq-len", type=int, default=300)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    info = {
        "torch": torch.__version__,
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    print(json.dumps(info))
    results = []
    for repo in args.repos:
        for dname in args.dtypes:
            t0 = time.perf_counter()
            model, cfg, n = build_random(repo, DTYPES[dname], dname)
            build_s = time.perf_counter() - t0
            r = time_forward(model, cfg.vocab_size, args.seq_len, args.reps, args.warmup)
            r.update(repo=repo, dtype=dname, params_B=round(n / 1e9, 2),
                     layers=cfg.num_hidden_layers, seq_len=args.seq_len, build_s=round(build_s, 1))
            print(json.dumps(r), flush=True)
            results.append(r)
            del model
            gc.collect()
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"info": info, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
