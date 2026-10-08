"""Validation 1: reproduce Wang et al., Nat. Commun. 13, 123 (2022) "An optical neural network using
less than 1 photon per multiplication" with OUR noise model (incoherent light, shot noise on all the
detected light, detector at half the shot-noise SNR as their SI reports, a fixed crosstalk floor).
Their setup: non-negative image vectors x (display) times non-negative weights w (SLM), summed on one
detector; N up to 711^2; measured RMS dot-product error vs detected photons per multiplication:
<6 % at 0.001, <1 % at 0.1, ~0.2 % floor at >=2 (Fig. 2b,c). Error here = RMS of (measured - true) /
true dot product. Their exact normalization and vectors (STL-10) differ; this is a first check."""
import numpy as np


def predict(N, photons, snr_penalty=2.0, floor=0.002, trials=400, seed=0):
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(trials):
        x = rng.random(N); w = rng.random(N)
        y = x @ w
        mean_light = photons * N                     # detected photons for this dot product (per-MAC average)
        lam = mean_light * (x * w).sum() / (x * w).sum()
        meas = rng.poisson(lam) / mean_light * y
        noise_extra = (snr_penalty - 1) * np.sqrt(lam) / mean_light * y * rng.standard_normal()
        meas = meas + noise_extra + floor * y * rng.standard_normal()
        errs.append((meas - y) / y)
    return float(np.sqrt(np.mean(np.square(errs))))


if __name__ == "__main__":
    import json, sys
    rows = []
    for N in (10_000, 100_000, 505_521):
        for p in (0.001, 0.01, 0.1, 1, 2, 10):
            e = predict(N, p)
            rows.append(dict(N=N, photons=p, predicted_rms=e))
            print(f"N={N:>7} photons/MAC {p:>6}: predicted RMS error {e:.3%}", file=sys.stderr)
    print(json.dumps(rows))
