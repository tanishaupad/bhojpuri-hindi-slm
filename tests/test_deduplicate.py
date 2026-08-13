from __future__ import annotations

from lma_phase1.deduplicate import hamming_distance, simhash


def test_simhash_is_deterministic():
    text = "यह एक पर्याप्त लंबा परीक्षण वाक्य है जो बार बार समान परिणाम देता है"
    assert simhash(text) == simhash(text)


def test_similar_text_has_closer_hash_than_unrelated_text():
    first = "आज मौसम बहुत अच्छा है और बच्चे मैदान में खेल रहे हैं"
    edited = "आज मौसम बहुत अच्छा है तथा बच्चे मैदान में खेल रहे हैं"
    unrelated = "कंप्यूटर विज्ञान में एल्गोरिदम समस्याओं को हल करते हैं"
    assert hamming_distance(simhash(first), simhash(edited)) < hamming_distance(
        simhash(first), simhash(unrelated)
    )
