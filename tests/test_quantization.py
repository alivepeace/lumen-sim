"""Deterministic quantizer contracts, independent of LM and physics validation."""
import pytest
import torch

from lumen.engine import Engine, OpticalLinear
from lumen.hardware import adc


def ideal_engine(kind="incoherent", **kwargs):
    eng = Engine(kind=kind, gain_drift=0.0, **kwargs)
    # Isolate the quantization grid from calibration residual / physical noise.
    eng.randn = lambda *shape: torch.zeros(*shape)
    return eng


@pytest.mark.parametrize("kind", ["incoherent", "coherent"])
@pytest.mark.parametrize("bits", [6, 10])
def test_input_signed_levels(kind, bits):
    q = 2 ** bits
    grid = torch.arange(-q, q + 1).float() / q
    # Every vector has a full-scale anchor; scalar vectors would normalize to +/-1.
    x = torch.stack((grid, torch.ones_like(grid)), dim=-1)
    lin = OpticalLinear(torch.eye(2), None, ideal_engine(kind, in_bits=bits))
    out = lin(x)[:, 0]
    assert torch.equal(out, grid)
    assert out.unique().numel() == 2 * q + 1
    # Half-step ties go to even grid indices, symmetrically about zero.
    ties = torch.tensor([-1.5, -0.5, 0.5, 1.5]) / q
    out = lin(torch.stack((ties, torch.ones_like(ties)), dim=-1))[:, 0]
    assert torch.equal(out, torch.tensor([-2., 0., 0., 2.]) / q)
    assert torch.equal(lin(torch.zeros(1, 2)), torch.zeros(1, 2))
    assert torch.equal(lin(x * 8), lin(x) * 8)


@pytest.mark.parametrize("kind", ["incoherent", "coherent"])
def test_weight_signed_levels(kind):
    n = 255
    grid = torch.arange(-n, n + 1).float() / n
    # A single row ensures all weights share the same normalization scale.
    lin = OpticalLinear(grid[None], None, ideal_engine(kind))
    assert torch.equal(lin.wh[0], grid)
    assert lin.wh.unique().numel() == 511
    # Exercise rounding away from exact grid points, including zero rows.
    w = torch.tensor([[-1., -1.5/n, -0.5/n, 0.5/n, 1.5/n, 1.], [0.] * 6])
    lin = OpticalLinear(w * 4, None, ideal_engine(kind))
    expected = torch.tensor([[-1., -2/n, 0., 0., 2/n, 1.], [0.] * 6])
    torch.testing.assert_close(lin.wh, expected, rtol=0, atol=0)


def test_engine_adc_includes_both_endpoints(monkeypatch):
    monkeypatch.setattr(torch, "randn", lambda shape, **kwargs: torch.zeros(shape))
    eng = ideal_engine(e=1.0, read_e=0.0, in_bits=6, adc_bits=6)
    lin = OpticalLinear(torch.tensor([[1., 0.]]), None, eng)
    lin.full = torch.ones(1)  # Fixed calibration range for this contract.
    grid = torch.arange(-32, 33).float() / 32
    x = torch.stack((grid, torch.ones_like(grid)), dim=-1)
    out = lin(x)[:, 0]
    assert torch.equal(out, grid)
    assert out.unique().numel() == 65
    lin.full = torch.tensor([0.5])
    assert torch.equal(lin(x)[[0, -1], 0], torch.tensor([-0.5, 0.5]))


def test_hardware_adc_uses_total_bits():
    v = torch.linspace(-1, 1, 4097)
    out = adc(v, full_scale=1., bits=6, enob=6., noise=False, gen=None)
    assert out.unique().numel() == 64
    assert out.min() == -1 and out.max() == 1


@pytest.mark.parametrize("flag", ["--in-bits", "--input-fractional-bits"])
def test_cli_preserves_legacy_configuration(monkeypatch, capsys, flag):
    from lumen import engine
    from lumen.__main__ import main
    calls = []
    def evaluate(model, kwargs, windows, ctx):
        calls.append(kwargs)
        return {"engine": kwargs}
    monkeypatch.setattr(engine, "evaluate", evaluate)
    main(["llm", flag, "6"])
    assert calls == [None, {"kind": "incoherent", "e": None, "in_bits": 6}]
    assert '"in_bits": 6' in capsys.readouterr().out
