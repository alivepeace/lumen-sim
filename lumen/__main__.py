"""python -m lumen <command>

  test        textbook physics checks: if these fail, trust nothing
  reproduce   re-run the Wang et al. 2022 optical neural network experiment in simulation
  llm         run a real language model through the simulated optical engine (perplexity)
"""
import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lumen", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["test", "reproduce", "llm"])
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--kind", default="incoherent", choices=["incoherent", "coherent", "digital"])
    ap.add_argument("--light", type=float, default=None,
                    help="photo-electrons (incoherent) or photons (coherent) per full-scale multiply-add; omit for no noise")
    ap.add_argument("--input-fractional-bits", "--in-bits", dest="in_bits", type=int, default=10,
                    help="fractional input bits B on [-1, 1]: 2**(B+1)+1 signed levels, not total hardware bits; default 10; --in-bits is the legacy alias")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=512)
    a = ap.parse_args(argv)
    if a.cmd == "test":
        import pytest
        sys.exit(pytest.main([str(Path(__file__).resolve().parent.parent / "tests"), "-q"]))
    elif a.cmd == "reproduce":
        from . import wang2022
        print(json.dumps({str(k): v for k, v in wang2022.report().items()}, indent=1, default=str))
    elif a.cmd == "llm":
        from .engine import evaluate
        base = evaluate(a.model, None, a.windows, a.ctx)
        eng = evaluate(a.model, {"kind": a.kind, "e": a.light, "in_bits": a.in_bits}, a.windows, a.ctx)
        print(json.dumps({"exact": base, "optical": eng}, indent=1))


if __name__ == "__main__":
    main()
