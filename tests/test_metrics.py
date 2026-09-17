import numpy as np
from fpkit.eval.metrics import evaluate


def test_perfect_separation():
    r = evaluate(genuine=[0.9, 0.8, 0.95], impostor=[0.1, 0.2, 0.3])
    assert r.eer == 0.0 and r.zero_fmr == 0.0


def test_gaussian_overlap_eer():
    rng = np.random.default_rng(0)
    g = rng.normal(1.0, 1.0, 50_000)
    i = rng.normal(-1.0, 1.0, 50_000)
    # two unit gaussians 2 sigma apart -> EER = Phi(-1) ~= 15.87%
    assert abs(evaluate(g, i).eer - 0.1587) < 0.01
