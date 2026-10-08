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

Real models have a few huge activation values. Scaling each input vector to its largest value and quantizing it crushes the ordinary values: GPT-2 at 6 input bits goes from perplexity 47 to over 3,000 with no noise at all, while 10 bits keeps it near 52. Input precision and how the engine handles loud values matter as much as the light budget.

## License

MIT
