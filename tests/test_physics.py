"""Textbook checks: if these fail, trust nothing."""
import math

import numpy as np
import pytest
import torch

from lumen import physics as ph
from lumen.engine import Engine, OpticalLinear
from lumen.hardware import adc, tia
from lumen.validate_wang import predict

DT = torch.float64


def test_johnson_noise_textbook():
    # a 1 MOhm resistor at 300 K: 0.129 pA/rtHz
    assert math.sqrt(ph.johnson_noise_var(1e6, 1.0)) == pytest.approx(1.287e-13, rel=1e-3)


def test_tia_bandwidth_textbook():
    # GBW 1 MHz, Rf 1 MOhm, Cin 100 pF -> 39.9 kHz
    assert ph.tia_max_bandwidth(1e6, 1e6, 100e-12) == pytest.approx(39_894, rel=1e-3)


def test_opamp_noise_textbook():
    # a 28 nV/rtHz, 1 MHz op-amp in a 4.1 MOhm, 1 kHz stage with 146 pF in: noise gain 4.8,
    # ~77 uV rms at the output (hand calculation)
    c_f = 1 / (2 * math.pi * 4.1e6 * 1000)
    assert 1 + 146e-12 / c_f == pytest.approx(4.76, rel=0.01)
    assert math.sqrt(ph.opamp_output_noise_var(28e-9, 146e-12, c_f, 1e6)) == pytest.approx(77e-6, rel=0.03)


def test_shot_limited_bits():
    assert ph.shot_limited_bits(2 ** 20) == pytest.approx(10.0)


def test_infinite_glowing_plane():
    """A detector over an endless glowing plane of exitance M receives M * A_det.
    Sums the geometry law over a fine grid; checks the geometry law."""
    h, area, m = 0.004, 7.5e-6, 2.0
    step = 0.0001
    g = np.arange(-0.4, 0.4, step) + step / 2
    xx, yy = np.meshgrid(g, g)
    received = (ph.lambert_share(area, h, xx ** 2 + yy ** 2 + h ** 2) * m * step ** 2).sum()
    assert received == pytest.approx(m * area, rel=0.01)


def test_tia_noise_matches_formula():
    gen = torch.Generator().manual_seed(0)
    n = 200_000
    i = torch.full((n,), 1e-6, dtype=DT)
    z = torch.zeros(n, dtype=DT)
    rf, bw, cin, en, gbw = 1e6, 1000.0, 146e-12, 28e-9, 1e6
    v = tia(i, z, rf=rf, vos=0.0, bw=bw, c_in=cin, en=en, gbw=gbw, swing=1e9, noise=True, gen=gen)
    bn = ph.noise_bandwidth(bw)
    c_f = 1 / (2 * math.pi * rf * bw)
    want = math.sqrt((ph.shot_noise_var(1e-6, bn) + ph.johnson_noise_var(rf, bn)) * rf ** 2
                     + ph.opamp_output_noise_var(en, cin, c_f, gbw))
    assert float(v.std()) == pytest.approx(want, rel=0.01)
    assert float(v.mean()) == pytest.approx(1.0, rel=1e-3)


def test_adc_effective_bits():
    gen = torch.Generator().manual_seed(0)
    v = torch.linspace(-1, 1, 200_001, dtype=DT)
    out = adc(v, full_scale=1.0, bits=12, enob=9.0, noise=True, gen=gen)
    err = (out - v) / (2 / 4095)            # in LSB
    enob = 12 - math.log2(float(err.std()) * math.sqrt(12))
    assert enob == pytest.approx(9.0, abs=0.1)


def test_guard_energy_and_speed():
    with pytest.raises(ph.PhysicsViolation):
        ph.check_passive(1.0, 1.01, "test")
    with pytest.raises(ph.PhysicsViolation):     # 100 kHz from a 3 MOhm TIA on a 1 MHz op-amp
        ph.check_bandwidth(100e3, ph.tia_max_bandwidth(1e6, 3e6, 146e-12), "test")


def test_engine_digital_is_exact():
    torch.manual_seed(0)
    w, x = torch.randn(32, 64), torch.randn(8, 64)
    lin = OpticalLinear(w, None, Engine(kind="digital"))
    assert torch.allclose(lin(x), x @ w.T, atol=1e-5)


def test_engine_noiseless_close_to_exact():
    torch.manual_seed(0)
    w, x = torch.randn(32, 64), torch.randn(8, 64)
    lin = OpticalLinear(w, None, Engine(kind="incoherent", e=None, in_bits=12, weight_levels=4096, gain_drift=0.0))
    y, ref = lin(x), x @ w.T
    assert float((y - ref).norm() / ref.norm()) < 0.01


def test_engine_shot_noise_falls_with_light():
    torch.manual_seed(0)
    w, x = torch.randn(64, 256), torch.randn(64, 256)
    ref = x @ w.T
    err = []
    for e in (10.0, 1000.0):
        lin = OpticalLinear(w, None, Engine(kind="incoherent", e=e, read_e=0.0, gain_drift=0.0, adc_bits=20))
        err.append(float((lin(x) - ref).norm() / ref.norm()))
    assert err[1] < err[0] / 5          # 100x the light: ~10x less shot noise


def test_wang2022_dot_product_error():
    """Wang et al. 2022 measured about 6 % RMS error at 0.001 photons per multiply and under 1 % at 0.1."""
    assert 0.03 < predict(505_521, 0.001, trials=60) < 0.12
    assert predict(505_521, 0.1, trials=60) < 0.01
