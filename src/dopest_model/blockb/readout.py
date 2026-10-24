"""Letter readout for the vLLM verifiers: pure functions, so they are tested without a GPU."""
from __future__ import annotations

import math

from ..prompt import LETTERS


def _logaddexp(a: float, b: float) -> float:
    m = max(a, b)
    return m if m == -math.inf else m + math.log(math.exp(a - m) + math.exp(b - m))


def letter_distribution(top_logprobs: dict, k: int, decode=None) -> list[float] | None:
    """Distribution over the first k option letters from one position's top logprobs.

    'A' and ' A' are pooled; a letter outside the top list gets zero mass. None when no letter is in the list.
    """
    mass = {}
    for tid, lp in top_logprobs.items():
        text = lp.decoded_token if lp.decoded_token is not None else (decode([tid]) if decode else "")
        text = text.strip()
        if len(text) == 1 and text in LETTERS[:k]:
            mass[text] = _logaddexp(mass.get(text, -math.inf), lp.logprob)
    if not mass:
        return None
    top = max(mass.values())
    w = [math.exp(mass[L] - top) if L in mass else 0.0 for L in LETTERS[:k]]
    return [x / sum(w) for x in w]


def final_continuation(gen: list[int], end_id: int, to_final: list[int], final_only: list[int]) -> list[int]:
    """Tokens to append to a gpt-oss prompt so that the next token is the answer in the final channel.

    gen is the first pass's output, stopped at the analysis message's <|end|> or cut by the token budget;
    to_final is <|end|><|start|>assistant<|channel|>final<|message|>, final_only the last three of those tokens.
    """
    if not gen or gen[:len(final_only)] == final_only:     # no analysis message: open the final channel directly
        return list(final_only)
    if end_id in gen:
        gen = gen[:gen.index(end_id)]
    return gen + list(to_final)
