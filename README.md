# Lumen simulator

A simulator for AI hardware that computes with light. It answers one question before anything gets built: can an optical engine run a real language model as well as a GPU?

Light does multiply-adds almost for free, but every reading carries photon shot noise, detector noise, converter limits and drift. Lumen puts those flaws into every fixed-weight matrix multiply of a real model (GPT-2, Qwen2.5 and other Hugging Face causal LMs) and measures what happens to perplexity on WikiText-2.

```
pip install -r requirements.txt
python -m lumen test                         # textbook physics checks: if these fail, trust nothing
python -m lumen reproduce                    # re-run a published optical neural network experiment
python -m lumen llm --model gpt2 --light 1e5 # a real model through the simulated engine
```

## What is modeled

- **Incoherent engines** (light intensity, never negative): signed weights as + and - rows, signed inputs in two passes, shot noise on all the light that lands, detector read noise.
- **Coherent engines** (one laser, homodyne detection): signs as phase, noise set by the local oscillator, phase drift turned into weight error.
- **Both:** per-row weight scaling and limited weight levels, input quantization, a converter whose full scale is set on a calibration batch (values beyond it clip), and row gain drift.
- **Guardrails** (`lumen/physics.py`): no passive stage may put out more light than it takes in, no amplifier may run faster than its parts allow, and shot noise can never be switched off.

What stays digital: attention scores (two activations multiplied together), embeddings, norms, softmax and activation functions.

## Checked against

- Textbook formulas: resistor and op-amp noise, transimpedance bandwidth, shot noise, converter effective bits (`tests/`).
- A published experiment: Wang et al., "An optical neural network using less than 1 photon per multiplication", Nature Communications 13, 123 (2022). The dot-product error vs detected photons matches their measurements (about 6 % at 0.001 photons per multiply, under 1 % at 0.1).

## One thing it shows right away

Real models have a few huge activation values. Scaling each input vector to its largest value and quantizing it crushes the ordinary values: GPT-2 at `in_bits=6` (6 fractional input bits, 129 signed levels) goes from perplexity 47 to over 3,000 with no noise at all, while `in_bits=10` (2,049 signed levels) keeps it near 52. Input precision and how the engine handles loud values matter as much as the light budget.

## Quantization conventions and benchmark compatibility

The engine's historical settings describe grid resolution, **not total signed
hardware bit widths**. For normalized values in [-1, 1]:

| Engine setting | Grid before physical errors | Signed levels (including endpoints) |
| --- | --- | --- |
| `in_bits=B` | `round(x * 2**B) / 2**B` | `2**(B+1) + 1`; B=6: 129; B=10: 2,049 |
| `weight_levels=L` | `round(w * (L-1)) / (L-1)` | `2*L - 1`; default L=256: 511 |
| `adc_bits=A` | step `2*full / 2**A`, centered at zero | `2**A + 1` with clipping; default A=12: 4,097 |

Inputs scale per vector and weights per row, with a minimum scale of `1e-12`.
Rounding uses PyTorch's round-to-nearest, ties-to-even rule. Weight level counts
apply before the always-present calibration residual and optional phase drift;
those errors move weights off the grid. The `digital` engine bypasses all these quantizers. The engine ADC runs only
when `e` is set. With `clip=False` its output is not restricted to a finite range. The
separate `lumen.hardware.adc` uses a conventional `2**bits` total-level grid;
these two converter implementations have different conventions.

Use `python -m lumen llm --input-fractional-bits 6` for explicit CLI terminology.
`--in-bits 6` remains an equivalent alias. The CLI defaults to 10 fractional
input bits; the Python `Engine` defaults to `in_bits=6`. Python configuration
keys and result JSON (`in_bits`, `weight_levels`, `adc_bits`), defaults, random
number draws, and numerical equations remain unchanged to preserve old runs.

The GPT-2 figures above are historical reported results under these legacy
grids, not new measurements or benchmarks of conventional 6-bit/10-bit signed
hardware. Encoding 129 or 2,049 distinct values requires at least 8 or 12 total
binary bits respectively; 511 weight levels require at least 9. These counts
exclude scale storage and do not specify an optical hardware encoding. A future
change to conventional total-bit quantization would require rerunning both
GPT-2 precision claims; the published perplexities must not be transferred to
that different quantizer. This clarification does not independently validate
the perplexity results or the Wang 2022 physics model.

## License

MIT
