"""One-forward-pass letter readout with MLX: final hidden state -> soft-capped logits of the option letters.

Training (train.py) and evaluation (predict.py) share letter_logits(), so the trained quantity is exactly the
evaluated one.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx_lm import load

from .prompt import LETTERS


def letter_token_ids(tokenizer) -> list[int]:
    ids = []
    for letter in LETTERS:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != letter:
            raise ValueError(f"the letter {letter!r} is not a single token for this tokenizer")
        ids.append(encoded[0])
    return ids


def text_model(model):
    return getattr(model, "language_model", model)         # multimodal wrappers hold the text model


def letter_logits(lm, tokens: mx.array, letter_ids: mx.array) -> mx.array:
    """Soft-capped logits of the given letter tokens at the last position of each sequence: (batch, k)."""
    hidden = lm.model(tokens)[:, -1, :]                       # already final-normed
    if lm.tie_word_embeddings:
        head = lm.model.embed_tokens
        logits = hidden @ head.weight[letter_ids].T if hasattr(head, "weight") and head.weight.ndim == 2 and \
            head.weight.dtype != mx.uint32 else head.as_linear(hidden)[:, letter_ids]
    else:
        logits = lm.lm_head(hidden)[:, letter_ids]
    cap = lm.final_logit_softcapping
    return mx.tanh(logits / cap) * cap if cap is not None else logits


def prompt_tokens(tokenizer, msgs: list[dict]) -> list[int]:
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tokenizer.encode(prompt, add_special_tokens=False)


class LetterReader:
    def __init__(self, model_path: str, adapter_path: str | None = None, cache_gb: float = 4.0):
        mx.set_cache_limit(int(cache_gb * 1024 ** 3))          # variable prompt lengths would grow the cache unbounded
        self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
        self.lm = text_model(self.model)
        self.letter_ids = letter_token_ids(self.tokenizer)

    def prompt_tokens(self, msgs: list[dict]) -> list[int]:
        return prompt_tokens(self.tokenizer, msgs)

    def letter_logits(self, tokens: list[int], k: int) -> np.ndarray:
        """Soft-capped next-token logits of the first k letters (temperature not applied)."""
        picked = letter_logits(self.lm, mx.array([tokens]), mx.array(self.letter_ids[:k]))[0].astype(mx.float32)
        mx.eval(picked)
        return np.array(picked)
