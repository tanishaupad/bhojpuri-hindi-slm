"""Shared text preparation for tokenizer training, evaluation, and inference."""

from __future__ import annotations


def normalize_tokenizer_input(text: str) -> str:
    """Collapse every Unicode whitespace boundary to one ordinary space.

    SentencePiece reads its training corpus one sentence per line, so literal
    newlines cannot become ordinary vocabulary pieces under the identity
    character normalizer.  Applying this function consistently prevents held-
    out line breaks from being emitted as unknown tokens.
    """

    return " ".join(text.split())
