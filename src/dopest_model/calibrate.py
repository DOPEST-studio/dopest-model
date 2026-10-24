"""Per-question-type temperature scaling of letter logits.

The Jeff paper fits one temperature per question type by NLL on held-out data and never goes below 1
(Gemma 4 31B zero-shot: choice 5.96, yes/no 7.24, score 7.24). Temperatures fitted here are applied in
evaluate.py before the softmax.
"""
from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


NOUL_FLOOR = 0.801
"""JevBench counts a yes/no answer with 0.2 < P(yes) < 0.8 as an abstention (wrong). The commit floor moves such an
answer to the floor on its own side (Torchcast Decision 4B ships 0.801); the answer itself does not change."""


def commit_floor(p_yes: float, floor: float = NOUL_FLOOR) -> float:
    if max(p_yes, 1.0 - p_yes) < floor:
        return floor if p_yes >= 0.5 else 1.0 - floor
    return p_yes


def nll(logits_list: list[np.ndarray], gold: list[int], temperature: float) -> float:
    """Mean negative log-likelihood of the gold option at a given temperature."""
    return float(np.mean([-np.log(softmax(lg, temperature)[g] + 1e-12) for lg, g in zip(logits_list, gold)]))


def fit_temperature(logits_list: list[np.ndarray], gold: list[int]) -> float:
    """Temperature T >= 1 that minimizes the NLL of the gold options.

    logits_list: one array of K soft-capped letter logits per item (K varies between items).
    gold:        index of the correct option in each array.

    Golden-section search over the inverse temperature b = 1/T in [1/50, 1]: the NLL is convex in b (a log-sum-exp
    of linear functions of b minus a linear term), so it has a single minimum, and b <= 1 is exactly T >= 1.
    """
    if not logits_list:
        return 1.0
    f = lambda b: nll(logits_list, gold, 1.0 / b)
    lo, hi, g = 1.0 / 50, 1.0, (5 ** 0.5 - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = f(a), f(b)
    for _ in range(40):                          # interval shrinks by 0.618 per step: ~1e-8 after 40 steps
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = f(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = f(b)
    return 1.0 / ((lo + hi) / 2)
