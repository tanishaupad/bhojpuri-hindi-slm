import pytest

from lma_phase3.score_answer_presence import contains_answer, score_row


@pytest.mark.parametrize(
    "text,answer,expected",
    [
        ("राम की उम्र 20 साल है।", "राम", True),
        ("“राम”", "राम", True),
        ("रामू बड़ा है।", "राम", False),
        ("नाम सही है।", "ना", False),
        ("हाँ, दोनों बराबर हैं।", "हाँ", True),
        ("क़", "क़", True),
        ("उत्तर नहीं है।", "नहीं", True),
        ("नहीं", "हाँ", False),
        ("राम_श्याम", "राम", False),
        ("", "राम", False),
    ],
)
def test_whole_answer_presence(text, answer, expected):
    assert contains_answer(text, answer) is expected


def test_mentions_of_multiple_candidates_are_credited_but_flagged():
    row = {
        "id": "x",
        "task_family": "direct",
        "answer_text": "राम",
        "problem": {"entities": ["राम", "श्याम"], "query": [0, 1]},
    }
    result = score_row({"generated": "राम और श्याम दोनों।"}, row, {})
    assert result["answer_present"]
    assert result["correct_and_alternative_present"]
    assert not result["correct_present_without_alternative"]
    assert not result["strict_exact_match"]
