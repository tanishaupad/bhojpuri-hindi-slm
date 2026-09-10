import json
from pathlib import Path

import pytest

from lma_phase3.proof_checker import reference_explanation
from lma_phase3.semantic_check import score_semantics

ROOT = Path(__file__).resolve().parents[1]


def example(family="direct", language="hindi"):
    base = json.loads((ROOT / language / "configs/phase3_templates.json").read_text())
    rows = [
        json.loads(s)
        for s in (ROOT / f"report/samples/generated/phase3/pilot_v1/{language}_development.jsonl")
        .read_text()
        .splitlines()
    ]
    return next(r for r in rows if r["task_family"] == family), base


@pytest.mark.parametrize("family", ["direct", "extremum", "equality", "transitivity"])
@pytest.mark.parametrize("language", ["hindi", "bhojpuri"])
def test_proof_does_not_require_markers_or_exact_layout(family, language):
    row, base = example(family, language)
    proof = reference_explanation(row, base).replace("कारण: ", "")
    assert score_semantics(row, proof, base)["joint_fully_verified"]
    sentences = [s for s in proof.split("।") if s.strip()]
    assert score_semantics(row, "।".join(reversed(sentences)), base)["joint_fully_verified"]


def test_introductory_fact_is_checked_without_being_mistaken_for_answer():
    row, base = example()
    text = "यह बात तो तय है कि " + row["premises"][0]
    result = score_semantics(row, text, base)
    assert result["supported_claim_found"]
    assert result["answer_status"] == "unresolved"
    assert not result["sufficient_supported_justification"]


def test_negated_meta_claim_does_not_become_an_assertion():
    row, base = example()
    text = "यह गलत है कि " + row["premises"][0]
    result = score_semantics(row, text, base)
    assert not result["claim_coverage"] and not result["full_text_coverage"]


def test_answer_can_be_expressed_as_comparison_without_repeating_name_as_label():
    row, base = example("transitivity")
    text = reference_explanation(row, base).split("कारण:")[1]
    result = score_semantics(row, text, base)
    assert result["joint_fully_verified"] and not result["explicit_answers"]


def test_wrong_numbers_fail_and_uninterpreted_text_is_separate():
    row, base = example()
    text = reference_explanation(row, base).replace("कारण: ", "")
    wrong = text.replace(str(row["problem"]["values"][0]), "999")
    assert score_semantics(row, wrong, base)["has_unsupported_claim"]
    extended = score_semantics(row, text + " असमर्थित अतिरिक्त वाक्य।", base)
    assert extended["joint_supported"] and not extended["joint_fully_verified"]


def test_natural_superlative_with_values_is_supported():
    row, base = example("extremum")
    attr = "उम्र" if row["domain"] == "age" else "लंबाई"
    word = "ज़्यादा" if row["direction"] == "greater" else "कम"
    text = " ".join(row["premises"]) + f" {row['answer_text']} की {attr} सबसे {word} है।"
    assert score_semantics(row, text, base)["joint_fully_verified"]


def test_name_in_unrelated_prose_is_not_an_answer():
    row, base = example()
    result = score_semantics(row, f"{row['answer_text']} के बारे में एक कहानी है।", base)
    assert result["answer_status"] == "unresolved"


def test_questions_are_not_assertions():
    row, base = example()
    text = row["premises"][0].rstrip("।") + "?"
    assert not score_semantics(row, text, base)["claim_coverage"]
    assert score_semantics(row, row["answer_text"] + "?", base)["answer_status"] == "unresolved"


def test_answer_sentence_does_not_require_colon():
    row, base = example()
    assert score_semantics(row, f"सही उत्तर {row['answer_text']} है।", base)["answer_correct"]


def test_coordinated_subject_or_list_is_not_a_standalone_answer():
    row, base = example()
    name = row["answer_text"]
    text = f"{name} और {name} की उम्र 40 साल है।"
    result = score_semantics(row, text, base)
    assert result["answer_status"] == "unresolved"
    assert not result["full_text_coverage"]
    assert score_semantics(row, f"{name}, कहानी अधूरी है।", base)["answer_status"] == "unresolved"


def test_trailing_name_in_truncated_prose_remains_ambiguous():
    row, base = example()
    text = row["premises"][0] + " " + row["answer_text"]
    assert score_semantics(row, text, base)["answer_status"] == "unresolved"
    assert score_semantics(row, row["answer_text"], base)["answer_correct"]
