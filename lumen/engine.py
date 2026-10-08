"""Run a real language model with every fixed-weight matrix multiply done by a simulated optical engine.

Instead of asking "is the error as small as N-bit weights?", run GPT-2 / Qwen2.5 on WikiText-2 and
measure perplexity (how surprised the model is by real text; lower is better) with the engine's
physical errors on every layer.

What stays digital: attention scores (query x key, scores x value) multiply two activations, not a
fixed weight. Embedding lookups, layer norms, softmax and activation functions are digital too.

The engine, per matrix W [out, in]:
  weights    each row stored scaled to its own maximum (corrected digitally after reading),
             `weight_levels` levels, plus a small per-cell calibration residual
  inputs     each token's vector scaled to its maximum and quantized to `in_bits`
  incoherent intensity-encoded light is never negative, so signed weights use + and - rows and
             signed inputs take two passes. Shot noise follows ALL the light that lands, not the
             answer: var = e * sum_j |x_j| |w_ij|, e = photo-electrons per full-scale multiply-add,
             plus the detector's own read noise `read_e` (electrons rms).
  coherent   one laser and homodyne detection: negative numbers are a phase flip, one pass. The
             reading's noise is the local oscillator's, fixed:
             X = sqrt(p) * sum_j x_j w_ij + N(0, 1/4); p = detected photons per full-scale input.
             Phase drift between calibrations turns into weight error |w| * N(0, drift).
  both       an `adc_bits` converter per row whose full scale is set on the first (calibration)
             batch with `headroom`; values beyond it clip. Row gain drift.
"""
import math
import time

import torch
import torch.nn as nn


class Engine:
    def __init__(self, kind="incoherent", e=None, in_bits=6, weight_levels=256, adc_bits=12,
                 gain_drift=0.005, phase_drift=0.0, read_e=60.0, headroom=4.0, clip=True, seed=0):
        """kind: incoherent | coherent | digital. e: photo-electrons (incoherent) or detected photons
        (coherent) per full-scale multiply-add; None = no shot or read noise."""
        self.kind, self.e, self.in_bits, self.weight_levels = kind, e, in_bits, weight_levels
        self.levels = 2 ** adc_bits
        self.gain_drift, self.phase_drift, self.read_e = gain_drift, phase_drift, read_e
        self.headroom, self.clip = headroom, clip
        self.gen = torch.Generator().manual_seed(seed)
        self.light = 0.0            # running sum of detected light (e- or photons) and of multiply-adds
        self.macs = 0.0

    def randn(self, *shape):
        return torch.randn(*shape, generator=self.gen)


class OpticalLinear(nn.Module):
    """y = x W^T + b with W fixed. weight given as [out, in]."""

    def __init__(self, weight, bias, eng):
        super().__init__()
        self.eng = eng
        w = weight.detach().float()
        self.bias = None if bias is None else bias.detach().float()
        if eng.kind == "digital":
            self.w = w
            return
        m = w.abs().amax(1, keepdim=True).clamp_min(1e-12)
        wh = torch.round(w / m * (eng.weight_levels - 1)) / (eng.weight_levels - 1)
        wh = wh + (0.01 / 16) * wh.abs().mean() * eng.randn(*wh.shape)            # per-cell calibration residual
        if eng.kind == "coherent" and eng.phase_drift > 0:
            wh = wh + wh.abs() * eng.phase_drift * eng.randn(*wh.shape)
        self.wh, self.m = wh, m[:, 0]
        self.wabs = wh.abs()
        self.gain = 1 + eng.gain_drift * eng.randn(wh.shape[0])
        self.full = None            # per-row converter full scale, set on the first batch

    def forward(self, x):
        eng = self.eng
        shp = x.shape
        x = x.reshape(-1, shp[-1]).float()
        if eng.kind == "digital":
            y = x @ self.w.T
            return (y if self.bias is None else y + self.bias).reshape(*shp[:-1], -1)
        mx = x.abs().amax(1, keepdim=True).clamp_min(1e-12)
        q = 2 ** eng.in_bits
        xh = torch.round(x / mx * q) / q
        s = xh @ self.wh.T                                                  # the noiseless answer
        n_in = x.shape[1]
        eng.macs += s.numel() * n_in
        if eng.e is not None:
            if eng.kind == "incoherent":
                light = (xh.abs() @ self.wabs.T) * eng.e                        # both passes
                noise = torch.sqrt(light + 2 * eng.read_e ** 2) * torch.randn(s.shape, generator=eng.gen)
                read = s * eng.e + noise
                eng.light += float(light.sum())
                unit = eng.e
            else:
                sq = math.sqrt(eng.e)
                read = s * sq + 0.5 * torch.randn(s.shape, generator=eng.gen)
                eng.light += float(eng.e * (xh.abs() @ (self.wh ** 2).T).sum())
                unit = sq
            if self.full is None:
                self.full = read.abs().amax(0) * eng.headroom + 1e-9
            width = 2 * self.full / eng.levels
            read = torch.round(read / width) * width
            if eng.clip:
                read = torch.maximum(torch.minimum(read, self.full), -self.full)
            s = read / unit
        y = s * self.gain * self.m * mx
        return (y if self.bias is None else y + self.bias).reshape(*shp[:-1], -1)


def opticalize(model, eng, lm_head=True):
    """Swap every fixed-weight matrix multiply in a GPT-2 or Qwen/Llama-style model."""
    from transformers.pytorch_utils import Conv1D
    swaps = []
    for name, mod in model.named_modules():
        for cname, child in mod.named_children():
            full = f"{name}.{cname}" if name else cname
            if not lm_head and full.endswith("lm_head"):
                continue
            if isinstance(child, Conv1D):              # GPT-2 stores W as [in, out]
                swaps.append((mod, cname, OpticalLinear(child.weight.T, child.bias, eng)))
            elif isinstance(child, nn.Linear):
                swaps.append((mod, cname, OpticalLinear(child.weight, child.bias, eng)))
    for mod, cname, new in swaps:
        setattr(mod, cname, new)
    return len(swaps)


@torch.no_grad()
def perplexity(model, ids, ctx=512, windows=24):
    nll, n = 0.0, 0
    for k in range(windows):
        chunk = ids[k * ctx:(k + 1) * ctx][None]
        if chunk.shape[1] < 2:
            break
        out = model(chunk, labels=chunk)
        nll += float(out.loss) * (chunk.shape[1] - 1)
        n += chunk.shape[1] - 1
    return math.exp(nll / n)


def wikitext2_ids(tok, chars=400_000):
    """WikiText-2 test text, cached at data/wikitext2_test.txt (fetched with `datasets` the first time)."""
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "data" / "wikitext2_test.txt"
    if not path.exists():
        from datasets import load_dataset
        path.parent.mkdir(exist_ok=True)
        path.write_text("".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]), encoding="utf-8")
    return tok(path.read_text(encoding="utf-8")[:chars], return_tensors="pt").input_ids[0]


def evaluate(model_name, eng_kwargs, windows=24, ctx=512):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).eval()
    if getattr(model.config, "tie_word_embeddings", False):
        model.lm_head.weight = nn.Parameter(model.lm_head.weight.detach().clone())   # untie before swapping
    ids = wikitext2_ids(tok)
    t0 = time.time()
    eng = None
    if eng_kwargs is not None:
        eng = Engine(**eng_kwargs)
        opticalize(model, eng)
    ppl = perplexity(model, ids, ctx, windows)
    res = {"model": model_name, "engine": eng_kwargs, "ppl": ppl, "tokens": windows * ctx,
           "seconds": time.time() - t0}
    if eng is not None and eng.e is not None:
        res["light_per_mac"] = eng.light / max(eng.macs, 1)
    return res
