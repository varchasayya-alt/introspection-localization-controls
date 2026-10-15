"""Shared code for the sentence-localization experiments.

Conventions (frozen for Gate 1; see gates/gate1_plan.md):
- "Layer L" = the residual stream at the OUTPUT of decoder block L (model.model.layers[L]),
  i.e. hidden_states[L + 1] in Hugging Face numbering. Concept vectors are read there and
  injected there.
- Weights stored in bf16, all math in fp32 (deviation D1).
- Answer = argmax over the logits of the tokens "1".."10" at the last position.
"""
import copy
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("HF_HOME", r"%USERPROFILE%\.cache\huggingface")  # outside OneDrive

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("INTROSPECTION_DATA", r"%USERPROFILE%\IntrospectionData"))

INTRO = (
    "I am an interpretability researcher studying how language models like you work. "
    "I want to assess how well you can detect where I inject thoughts. "
    "I will show you ten sentences. While you read exactly one of them, I will inject a thought "
    "(the pattern of activity for a concept) directly into your internal activations. "
    "Your task is to tell me which sentence the injected thought is located in."
)
Q_LOCATE = "Do you detect an injected thought? Where is the injected thought located?"
PREFIX_LOCATE = "Yes, I detect an injected thought. It is located in SENTENCE "
Q_COPY = "Please repeat all ten sentences exactly as written above, in the same format."
CONCEPT_TEMPLATE = "Tell me about {word}"


# ----------------------------------------------------------------------------- model

def _fp32_compute_linear_forward(self, x):
    w = self.weight.float()
    b = self.bias.float() if self.bias is not None else None
    return torch.nn.functional.linear(x, w, b)


def make_bf16_storage_fp32_compute(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            m.forward = _fp32_compute_linear_forward.__get__(m, torch.nn.Linear)
        else:
            for _, p in list(m.named_parameters(recurse=False)):
                p.data = p.data.float()
    return model


def load_model(repo, revision=None, threads=12):
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(repo, revision=revision, dtype=torch.bfloat16)
    make_bf16_storage_fp32_compute(model)
    model.eval()
    return model, tok


def decoder_layers(model):
    return model.model.layers


def final_norm(model):
    return model.model.norm


# ----------------------------------------------------------------------------- prompts

def sentences_block(sentences):
    return "\n".join(f"SENTENCE {i + 1}: {s}" for i, s in enumerate(sentences))


def _template_ids(tok, msgs):
    out = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    if hasattr(out, "keys") and "input_ids" in out:
        out = out["input_ids"]
    if isinstance(out, torch.Tensor):
        out = out.tolist()
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


class Prompt:
    """A tokenized prompt plus the token indices of each sentence."""

    def __init__(self, tok, sentences, mode="locate", strict=True):
        # strict=False: allow sentences with no tokens (e.g. an empty "perceived" copy in the
        # reader control); such prompts are only read, never injected into.
        self.strict = strict
        self.sentences = list(sentences)
        self.mode = mode
        body = sentences_block(sentences)
        user2 = body + "\n\n" + (Q_LOCATE if mode == "locate" else Q_COPY)
        self.msgs = [
            {"role": "user", "content": INTRO},
            {"role": "assistant", "content": "Ok."},
            {"role": "user", "content": user2},
        ]
        head = tok.apply_chat_template(self.msgs, tokenize=False, add_generation_prompt=True)
        # Tulu-3's chat template omits BOS; open-instruct adds it separately at training time
        # (--add_bos for Llama-based models), so we prepend it when the template leaves it out.
        self.added_bos = bool(tok.bos_token) and not head.startswith(tok.bos_token)
        if self.added_bos:
            head = tok.bos_token + head
        self.head_text = head
        tail = PREFIX_LOCATE if mode == "locate" else body
        self.text = head + tail
        enc = tok(self.text, add_special_tokens=False, return_offsets_mapping=True)
        self.ids = enc["input_ids"]
        self.offsets = enc["offset_mapping"]
        # sentence char spans inside the user message (first copy of the block)
        b0 = self.text.find(body)
        assert b0 >= 0
        self.user_spans = self._spans(b0, body, sentences)
        self.user_tok = [self._tok_idx(*sp) for sp in self.user_spans]
        if mode == "copy":
            b1 = len(head)
            assert self.text[b1:] == body
            self.resp_spans = self._spans(b1, body, sentences)
            self.resp_tok = [self._tok_idx(*sp) for sp in self.resp_spans]

    @staticmethod
    def _spans(base, body, sentences):
        spans, pos = [], 0
        for i, s in enumerate(sentences):
            line = f"SENTENCE {i + 1}: {s}"
            j = body.find(line, pos)
            assert j >= 0
            cs = base + j + len(f"SENTENCE {i + 1}: ")
            spans.append((cs, cs + len(s)))
            pos = j + len(line)
        return spans

    def _tok_idx(self, cs, ce):
        idx = [i for i, (a, b) in enumerate(self.offsets) if b > a and a < ce and b > cs]
        if self.strict:
            assert idx, (cs, ce)
        return idx

    def tensor(self):
        return torch.tensor([self.ids])


def digit_token_ids(tok, prompt):
    """Token id for each answer "1".."10" after the locate prefix; checks single-token-ness."""
    ids = []
    for n in range(1, 11):
        full = tok(prompt.text + str(n), add_special_tokens=False)["input_ids"]
        if full[: len(prompt.ids)] != prompt.ids or len(full) != len(prompt.ids) + 1:
            raise ValueError(f'answer "{n}" is not a single token after the prefix')
        ids.append(full[-1])
    return ids


# ----------------------------------------------------------------------------- hooks

@contextmanager
def inject(model, layer, vec, positions):
    """Add `vec` (already scaled) to the residual stream at the output of block `layer`,
    at the given positions (relative to the tokens fed in this forward call)."""
    pos = torch.as_tensor(positions, dtype=torch.long)

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        h[:, pos, :] += vec.to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    handle = decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def capture_final(model):
    """Capture the final-norm output (pre-unembedding hidden states) of a forward call."""
    store = {}

    def hook(mod, inp, out):
        store["h"] = out[0] if isinstance(out, tuple) else out

    handle = final_norm(model).register_forward_hook(hook)
    try:
        yield store
    finally:
        handle.remove()


def logits_at(model, hidden_rows):
    """Unembed selected final-norm hidden rows -> fp32 logits."""
    return model.lm_head(hidden_rows).float()


# ----------------------------------------------------------------------------- forward helpers

@torch.inference_mode()
def clean_pass(model, prompt, want_hidden=False, want_final=False):
    ids = prompt.tensor()
    with capture_final(model) as cap:
        out = model(input_ids=ids, use_cache=True, output_hidden_states=want_hidden, logits_to_keep=1)
    return {
        "logits": out.logits[0, -1].float(),
        "cache": out.past_key_values,
        "hidden": out.hidden_states if want_hidden else None,
        "final": cap["h"][0] if want_final else None,
    }


@torch.inference_mode()
def run_suffix(model, prompt, start, cache, layer=None, vec=None, inj_abs=None, want_final=False):
    """Exact speed-up: reuse the clean KV cache for tokens [0, start) (identical because the
    model can't look ahead) and run only tokens [start, end), optionally injecting at absolute
    token positions `inj_abs` (all must be >= start)."""
    c = copy.deepcopy(cache)
    c.crop(start)
    ids = prompt.tensor()[:, start:]
    with capture_final(model) as cap:
        if layer is not None and vec is not None:
            rel = [p - start for p in inj_abs]
            assert min(rel) >= 0
            with inject(model, layer, vec, rel):
                out = model(input_ids=ids, past_key_values=c, use_cache=True, logits_to_keep=1)
        else:
            out = model(input_ids=ids, past_key_values=c, use_cache=True, logits_to_keep=1)
    return {"logits": out.logits[0, -1].float(), "final": cap["h"][0] if want_final else None}


@torch.inference_mode()
def run_full(model, prompt, layer=None, vec=None, inj_abs=None):
    ids = prompt.tensor()
    if layer is not None and vec is not None:
        with inject(model, layer, vec, inj_abs):
            out = model(input_ids=ids, use_cache=False, logits_to_keep=1)
    else:
        out = model(input_ids=ids, use_cache=False, logits_to_keep=1)
    return out.logits[0, -1].float()


# ----------------------------------------------------------------------------- vectors

@torch.inference_mode()
def last_token_states(model, tok, text, layers):
    enc = tok(text, return_tensors="pt")  # includes BOS like normal raw-text use
    out = model(**enc, output_hidden_states=True, logits_to_keep=1)
    return {L: out.hidden_states[L + 1][0, -1].float().clone() for L in layers}


def concept_vectors(model, tok, words, baseline_words, layers):
    """IFT 'simple' method: last-token state of "Tell me about {word}" minus the mean over the
    baseline words, at each layer; returned unit-length plus the raw difference norm."""
    base = [last_token_states(model, tok, CONCEPT_TEMPLATE.format(word=w), layers) for w in baseline_words]
    mean = {L: torch.stack([b[L] for b in base]).mean(0) for L in layers}
    vecs, raw_norms = {}, {}
    for w in words:
        s = last_token_states(model, tok, CONCEPT_TEMPLATE.format(word=w), layers)
        for L in layers:
            d = s[L] - mean[L]
            raw_norms[(w, L)] = float(d.norm())
            vecs[(w, L)] = d / d.norm()
    return vecs, raw_norms


def random_unit_vectors(dim, n, seed):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        v = torch.randn(dim, generator=g)
        out.append(v / v.norm())
    return out


# ----------------------------------------------------------------------------- storage

class ResultStore:
    """Append-only JSONL; every record is flushed and fsync'd, so a crash loses at most the
    trial in progress. Re-running skips keys that are already stored."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records = {}
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path, "rb+") as f:  # make sure a crash-truncated line is terminated
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    f.write(b"\n")
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a half-written last line from a crash
                    self.records[r["key"]] = r

    def has(self, key):
        return key in self.records

    def add(self, rec):
        rec["t_saved"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.records[rec["key"]] = rec


def log(msg, path=None):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
