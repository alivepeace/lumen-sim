"""Building blocks for simulated optical hardware, written as differentiable torch code so a
network can be trained *through* the simulated hardware.

Rounding uses a straight-through estimator: the forward pass rounds exactly like the
hardware, the backward pass lets the gradient through as if it did not.
"""
import math

import torch
import torch.nn.functional as F

from .physics import Q, KB, T_ROOM


def ste_round(x):
    return x + (torch.round(x) - x).detach()


def quantize(x, levels):
    n = levels - 1
    return ste_round(x.clamp(0, 1) * n) / n






def tia(i_pos, i_neg, *, rf, vos, bw, c_in, en, gbw, swing, noise, gen):
    """Two photodiodes wired to oppose each other into one transimpedance amplifier.
    The wire subtracts (Kirchhoff); the amplifier turns current into voltage.
    Noise: shot noise of every current reaching the node and resistor noise (filtered by
    the feedback capacitor), plus op-amp voltage noise (not filtered; see physics).
    Returns voltage relative to mid-rail, clipped at the rails."""
    i = i_pos - i_neg
    v = i * rf + vos
    if noise:
        bn = bw * math.pi / 2
        var_i = 2 * Q * (i_pos.abs() + i_neg.abs()) * bn + 4 * KB * T_ROOM * bn / rf
        c_f = 1 / (2 * math.pi * rf * bw)
        var_v = var_i * rf ** 2 + en ** 2 * (1 + c_in / c_f) * gbw * math.pi / 2
        v = v + var_v.sqrt() * torch.randn(i.shape, generator=gen, dtype=i.dtype)
    return v.clamp(-swing, swing)


def adc(v, *, full_scale, bits, enob, noise, gen):
    """Microcontroller ADC: v in [-full_scale, full_scale] mapped onto its range,
    rounded to `bits`, with extra noise so the effective bits equal `enob`."""
    n = 2 ** int(bits)
    u = (v / full_scale + 1) / 2 * (n - 1)
    if noise:
        lsb_noise = 2 ** (bits - enob) / math.sqrt(12)
        extra = math.sqrt(max(lsb_noise ** 2 - 1 / 12, 0.0))
        u = u + extra * torch.randn(u.shape, generator=gen, dtype=u.dtype)
    u = ste_round(u.clamp(0, n - 1))
    return (u / (n - 1) * 2 - 1) * full_scale
