"""Reproduction: Wang et al., "An optical neural network using less than 1 photon per
multiplication", Nature Communications 13, 123 (2022), doi:10.1038/s41467-021-27774-8.

Their build: an OLED phone screen shows the input vector, a reflective SLM holds the weights,
a lens images one onto the other, and a photon-counting-class detector (Hamamatsu MPPC, read
as an analog voltage) adds each dot product. A 784-100-100-10 MNIST network ran all three
matrix-vector products optically; bias, ReLU and argmax were electronic. Trained digitally
with 5-bit weights and 4-bit activations (activations 7-bit at inference), no hardware
retraining. Their stated results: ~99% at ~3.2 detected photons per multiplication, ~90% at
~0.64, on the first 130 MNIST test images, matching a shot-noise-only simulation.
Dot products of 711 x 711 = 505,521 elements: RMS error ~6% at 0.001 photons per
multiplication, <1% at 0.1, a floor of ~0.2% from 2 photons up.

What this checks: Lumen's photon-noise physics and how a real network degrades with it.
Assumptions (their supplementary sections were not readable): signed weights by offset,
w' = (w / w_max + 1) / 2 on the SLM, the offset removed electronically using the known input
sum; the same photon budget per multiplication in every layer.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hardware import ste_round
from .workloads import mnist

LAYERS = (784, 100, 100, 10)
PAPER = {"accuracy": {3.2: 0.99, 0.64: 0.90},          # their text: "~99%", "~90%" (">90%")
         "dot_error": {0.001: 0.06, 0.1: 0.01, 2.0: 0.002}}


def q_ste(x, bits, hi, stochastic=False):
    """Round onto 2^bits levels in [0, hi]. Their training used random rounding between the two
    nearest levels ("random digitization"), which hardens the network against noise."""
    n = 2 ** bits - 1
    u = x.clamp(0, hi) / hi * n
    r = torch.floor(u + torch.rand_like(u)) if stochastic else torch.round(u)
    return (u + (r - u).detach()) / n * hi


class QMLP(nn.Module):
    """Their network: 5-bit weights (symmetric), 4-bit activations in training."""

    def __init__(self):
        super().__init__()
        self.l = nn.ModuleList(nn.Linear(a, b) for a, b in zip(LAYERS, LAYERS[1:]))

    def qweight(self, l):
        s = l.weight.abs().max().detach()
        return ste_round(l.weight / s * 15) / 15 * s          # 5 bits: -15..15

    def forward(self, x, act_bits=4, stochastic=False):
        for i, l in enumerate(self.l):
            x = F.linear(x, self.qweight(l), l.bias)
            if i < len(self.l) - 1:
                x = q_ste(F.relu(x), act_bits, 4.0, stochastic)
        return x


def augment(x):
    """Their input augmentation: rotation +/-5 deg, shift and zoom +/-4%, a random 3x3 blur."""
    n = x.shape[0]
    a = (torch.rand(n) * 2 - 1) * math.radians(5)
    z = 1 + (torch.rand(n) * 2 - 1) * 0.04
    t = (torch.rand(n, 2) * 2 - 1) * 0.08
    th = torch.stack([torch.stack([torch.cos(a) / z, -torch.sin(a) / z, t[:, 0]], 1),
                      torch.stack([torch.sin(a) / z, torch.cos(a) / z, t[:, 1]], 1)], 1)
    img = x.view(n, 1, 28, 28)
    img = F.grid_sample(img, F.affine_grid(th, img.shape, align_corners=False), align_corners=False)
    k = torch.rand(n, 1, 3, 3)
    k = k / k.sum((2, 3), keepdim=True)
    k = torch.where(torch.rand(n, 1, 1, 1) < 0.5, k, F.pad(torch.ones(n, 1, 1, 1), (1, 1, 1, 1)))
    img = F.conv2d(img.view(1, n, 28, 28), k, padding=1, groups=n).view(n, 784)
    return img


def train(seed=0, epochs=4, log=print, hardened=False):
    """hardened: their recipe: full-precision warm-up, random rounding, input augmentation."""
    torch.manual_seed(seed)
    xtr, ytr, _, _ = mnist()
    m = QMLP()
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    n = len(xtr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        warm = hardened and ep < epochs // 3
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            xb = augment(xtr[idx]) if hardened else xtr[idx]
            out = m(xb, act_bits=12 if warm else 4, stochastic=hardened and not warm)
            loss = F.cross_entropy(out, ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        log(f"  epoch {ep + 1}/{epochs}")
    return m


EPS = 0.02          # the SLM's lowest transmission (their SI section 11)
PER_ROW = True
PHOTON_J = 6.62607015e-34 * 299792458.0 / 525e-9      # green OLED pixels


def optical_mvm(x, w, photons_per_mult, gen, excess=1.0):
    """Their SI section 11 scheme. Weights shifted and scaled into [EPS, 1] on the SLM; inputs
    x >= 0 on the OLED. Light gives w'.x for every row plus one extra block at full
    transmission giving 1.x (measured once, reused for every row). Electronics then form
    w.x = c1 (w'.x) + c2 (1.x). Each detector reading is photon-counted (Poisson; variance x
    `excess` for a noisier detector). Scale set so the average multiplication (reference block
    included) delivers `photons_per_mult` detected photons. Returns the result and the photons."""
    if PER_ROW:      # their Eq. 2 shifts and rescales each weight vector on its own
        lo, hi = w.min(1).values[None, :], w.max(1).values[None, :]
        wp = EPS + (1 - EPS) * (w - lo.T) / (hi - lo).T
    else:
        lo, hi = w.min(), w.max()
        wp = EPS + (1 - EPS) * (w - lo) / (hi - lo)
    c1 = (hi - lo) / (1 - EPS)
    c2 = lo - EPS * c1
    y_opt = x @ wp.T
    ref = x.sum(1, keepdim=True)
    n_mult = x.shape[1] * (w.shape[0] + 1)
    kappa = photons_per_mult * n_mult / (y_opt.sum(1, keepdim=True) + ref).clamp_min(1e-12)

    def count(mean):
        if excess == 1.0:
            return torch.poisson(mean, generator=gen)
        return mean + (mean * excess).sqrt() * torch.randn(mean.shape, generator=gen)

    y_hat = count(y_opt * kappa) / kappa
    ref_hat = count(ref * kappa) / kappa
    photons = float(((y_opt.sum(1) + ref[:, 0]) * kappa[:, 0]).mean())
    return c1 * y_hat + c2 * ref_hat, photons


@torch.no_grad()
def optical_accuracy(m, photons, n_images=None, seed=0, excess=1.0, repeats=1):
    """Accuracy, and detected optical energy per inference (J)."""
    _, _, xte, yte = mnist()
    if n_images:
        xte, yte = xte[:n_images], yte[:n_images]
    gen = torch.Generator().manual_seed(seed)
    accs, energy = [], 0.0
    for _ in range(repeats):
        x = (torch.round(xte * 127) / 127)                    # the OLED's ~7-bit levels
        total = 0.0
        per_layer = photons if isinstance(photons, (list, tuple)) else [photons] * len(m.l)
        for i, l in enumerate(m.l):
            y, ph_ = optical_mvm(x, m.qweight(l), per_layer[i], gen, excess)
            total += ph_
            x = y + l.bias
            if i < len(m.l) - 1:
                x = q_ste(F.relu(x), 7, 4.0)
        accs.append(float((x.argmax(1) == yte).float().mean()))
        energy = total * PHOTON_J
    return float(np.mean(accs)), energy


@torch.no_grad()
def digital_accuracy(m, n_images=None):
    _, _, xte, yte = mnist()
    if n_images:
        xte, yte = xte[:n_images], yte[:n_images]
    return float((m(xte, act_bits=7).argmax(1) == yte).float().mean())


@torch.no_grad()
def dot_error(photons, n=711 * 711, pairs=100, seed=0, excess=1.0):
    """RMS error of optical dot products of length n (relative to the true dot product),
    non-negative image-like vectors and weights, as in their Fig. 2."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(pairs, n, generator=g)
    w = torch.rand(pairs, n, generator=g)
    y = (x * w).sum(1)
    mean = photons * n * (x * w).sum(1) / (x * w).sum(1)    # photons per multiplication x n
    k = mean / y
    counts = torch.poisson(k * y, generator=g) if excess == 1.0 else k * y + (k * y * excess).sqrt() * torch.randn(pairs, generator=g)
    return float(((counts / k - y) / y).pow(2).mean().sqrt())


def report(log=print):
    """The whole check, as run for the README: dot-product physics, then the network."""
    log("dot-product error, 505,521 elements (paper: ~6% at 0.001, <1% at 0.1, ~0.2% floor from 2 up)")
    for p in (0.001, 0.1, 2.0):
        log(f"  {p:>6} photons/mult: shot noise only {dot_error(p):.4f}   with their detector (SNR halved) {dot_error(p, excess=4.0):.4f}")
    log("training their network with their recipe (warm-up, random rounding, augmentation) ...")
    m = train(epochs=9, log=lambda *_: None, hardened=True)
    log(f"digital: first 130 {digital_accuracy(m, 130):.3f}, all 10,000 {digital_accuracy(m):.4f}   (paper: 99%)")
    log("optical (paper: ~99% at 3.2 photons/mult, ~90% at 0.64; ~1 pJ detected per inference at 99%)")
    out = {}
    for p in (0.64, 3.2, 10, 30):
        a130, e = optical_accuracy(m, p, n_images=130, repeats=20)
        a10k, _ = optical_accuracy(m, p)
        out[p] = (a130, a10k, e)
        log(f"  {p:>5} photons/mult | first 130 {a130:.3f} | all 10,000 {a10k:.3f} | {e * 1e12:.3f} pJ per inference")
    return out
