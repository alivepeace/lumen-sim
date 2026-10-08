"""Tasks the hardware is judged on. Same task, same split, for every system."""
import numpy as np
import torch
from sklearn.datasets import load_digits


def digits(seed=0, test_frac=0.2):
    """8x8 handwritten digits (1,797 images, 10 classes), pixels scaled to 0..1."""
    d = load_digits()
    x = (d.data / 16.0).astype(np.float64)
    y = d.target.astype(np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    n_test = int(len(x) * test_frac)
    te, tr = idx[:n_test], idx[n_test:]
    t = lambda a: torch.from_numpy(a)
    return t(x[tr]), t(y[tr]), t(x[te]), t(y[te])


# multiply-adds in the logical network (no bias), the number every system is credited with
DIGITS_NET = (64, 16, 10)


def macs(layers):
    return sum(a * b for a, b in zip(layers, layers[1:]))


_MNIST = None


def mnist():
    """MNIST (60,000 train, 10,000 test), pixels 0..1, in the standard order (test image 0 first).
    Downloaded once through scikit-learn's OpenML cache."""
    global _MNIST
    if _MNIST is None:
        from sklearn.datasets import fetch_openml
        d = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff",
                         data_home=str(__import__("pathlib").Path(__file__).resolve().parent.parent / "data"))
        x = torch.from_numpy((d.data / 255.0).astype(np.float32))
        y = torch.from_numpy(d.target.astype(np.int64))
        _MNIST = (x[:60000], y[:60000], x[60000:], y[60000:])
    return _MNIST
