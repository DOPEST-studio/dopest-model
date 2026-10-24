import numpy as np

from dopest_model.calibrate import fit_temperature, nll


def test_recovers_the_temperature_of_overconfident_logits():
    """Labels drawn from softmax(z / 4): scaling the logits by 4 again should be undone by T close to 4."""
    rng = np.random.default_rng(0)
    logits, gold = [], []
    for _ in range(4000):
        z = rng.normal(0, 6, size=rng.integers(2, 6))
        p = np.exp(z / 4 - (z / 4).max())
        logits.append(z)
        gold.append(int(rng.choice(len(z), p=p / p.sum())))
    t = fit_temperature(logits, gold)
    assert 3.6 < t < 4.4
    assert nll(logits, gold, t) <= min(nll(logits, gold, x) for x in (1.0, 2.0, 3.0, 5.0, 8.0))


def test_never_sharpens_below_one():
    """Underconfident logits would want T < 1; the fit stops at 1."""
    logits = [np.array([0.2, 0.0]), np.array([0.0, 0.2])]
    assert abs(fit_temperature(logits, [0, 1]) - 1.0) < 1e-3
    assert fit_temperature([], []) == 1.0
