from __future__ import annotations

from lma_phase1.audit import sample_score


def test_sample_score_is_deterministic_and_seeded():
    assert sample_score("seed", "doc") == sample_score("seed", "doc")
    assert sample_score("seed", "doc") != sample_score("other", "doc")
