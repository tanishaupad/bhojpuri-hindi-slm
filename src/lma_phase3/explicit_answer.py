"""Frozen, format-based answer extraction; never search premises for the gold answer."""

from __future__ import annotations

import re

from lma_phase3.evaluate import normalize_answer

SCORING_VERSION = "explicit_answer_v1"


def extract_explicit_answer(generated: str) -> str:
    """Extract the leading answer span, independent of choices and ground truth.

    With a कारण: marker, the entire preceding span is the answer field. Otherwise
    use the first sentence (।, ., !, ? or newline), or the complete bare response.
    Remove one leading उत्तर:/जवाब: label and surrounding terminal punctuation.
    No paraphrase matching, answer lookup, or later-clause search is performed.
    """
    text = generated.strip()
    text = re.sub(r"^(?:उत्तर|जवाब)\s*:\s*", "", text)
    parts = re.split(r"कारण\s*:", text, maxsplit=1)
    field = parts[0] if len(parts) == 2 else re.split(r"[।.!?\n]", text, maxsplit=1)[0]
    return normalize_answer(field.strip().strip("।.!?;:"))


def score_explicit_answer(generated: str, reference: str, semantics: dict) -> dict:
    """Keep explicit answer accuracy separate from whole-explanation verification."""
    answer = extract_explicit_answer(generated)
    correct = bool(answer) and answer == normalize_answer(reference)
    return {
        "answer_scoring_version": SCORING_VERSION,
        "extracted_answer": answer,
        "explicit_answer_correct": correct,
        "explicit_answer_and_explanation_verified": (
            correct and semantics["explanation_fully_verified"]
        ),
    }
