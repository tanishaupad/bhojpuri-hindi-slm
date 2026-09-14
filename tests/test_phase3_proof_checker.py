import json
from pathlib import Path

import pytest
import torch

from lma_phase2.config import ModelConfig
from lma_phase2.model import DecoderOnlyTransformer
from lma_phase3.cached_decode import cached_step, greedy_cached
from lma_phase3.evaluate import greedy_batch
from lma_phase3.proof_checker import check_output, reference_explanation

ROOT = Path(__file__).resolve().parents[1]


def pilot(language, family):
    base = json.loads((ROOT / language / "configs/phase3_templates.json").read_text())
    rows = [
        json.loads(s)
        for s in (
            ROOT / "report/samples/generated/phase3/pilot_v1" / f"{language}_development.jsonl"
        )
        .read_text()
        .splitlines()
    ]
    return next(r for r in rows if r["task_family"] == family), base


@pytest.mark.parametrize("language", ["hindi", "bhojpuri"])
@pytest.mark.parametrize("family", ["direct", "extremum", "equality", "transitivity"])
def test_valid_proofs_and_reordered_sentences(language, family):
    row, base = pilot(language, family)
    text = reference_explanation(row, base)
    assert check_output(row, text, base)["joint_correct"]
    answer, reason = text.split("कारण:")
    sentences = [s.strip() for s in reason.split("।") if s.strip()]
    reordered = answer + "कारण: " + "। ".join(reversed(sentences)) + "।"
    assert check_output(row, reordered, base)["joint_correct"]


def test_wrong_answer_does_not_invalidate_separately_correct_explanation():
    row, base = pilot("hindi", "direct")
    text = reference_explanation(row, base)
    wrong = next(n for n in row["problem"]["entities"] if n != row["answer_text"])
    text = wrong + text[len(row["answer_text"]) :]
    result = check_output(row, text, base)
    assert result["explanation_verified"]
    assert not result["answer_correct"] and not result["joint_correct"]


def test_invented_value_is_parsed_but_unsupported():
    row, base = pilot("hindi", "direct")
    text = reference_explanation(row, base).replace(str(row["problem"]["values"][0]), "999")
    result = check_output(row, text, base)
    assert result["explanation_parseable"] and not result["explanation_verified"]
    assert result["status"] == "unsupported_claim"


def test_negation_is_never_accepted_as_positive_relation():
    row, base = pilot("hindi", "transitivity")
    text = reference_explanation(row, base).replace("ज़्यादा है", "ज़्यादा नहीं है")
    result = check_output(row, text, base)
    assert result["status"] == "unparsed_text"
    assert not result["joint_correct"]


def test_endpoint_assertion_without_chain_is_insufficient():
    row, base = pilot("hindi", "transitivity")
    text = reference_explanation(row, base)
    final = [s.strip() for s in text.split("कारण:")[1].split("।") if s.strip()][-1]
    result = check_output(row, row["answer_text"] + "। कारण: " + final + "।", base)
    assert result["status"] == "insufficient_justification"
    assert not result["joint_correct"]


def test_extremum_requires_comparison_covering_every_competitor():
    row, base = pilot("hindi", "extremum")
    text = reference_explanation(row, base)
    sentences = text.split("कारण:")[1].split("।")
    shortened = row["answer_text"] + "। कारण: " + "।".join(sentences[:-2]) + "।"
    assert not check_output(row, shortened, base)["explanation_verified"]


def test_wrong_equality_conclusion_is_unsupported():
    row, base = pilot("bhojpuri", "equality")
    assert row["answer_label"] is True
    text = reference_explanation(row, base).replace("बराबर बा", "बराबर नइखे")
    assert check_output(row, text, base)["status"] == "unsupported_claim"


def test_unknown_extra_text_cannot_hide_behind_a_valid_proof():
    row, base = pilot("hindi", "direct")
    result = check_output(row, reference_explanation(row, base) + " कहीं भी कुछ भी।", base)
    assert not result["explanation_parseable"] and not result["joint_correct"]
    result = check_output(row, row["answer_text"], base)
    assert result["answer_correct"] and result["status"] == "missing_explanation"


def test_reversed_relation_contradicts_premises():
    row, base = pilot("hindi", "transitivity")
    a, b = row["problem"]["edges"][0]
    names = row["problem"]["entities"]
    domain = base["domains"][row["domain"]]
    reverse = domain["relation"].format(left=names[b], right=names[a])
    result = check_output(row, reference_explanation(row, base) + " " + reverse, base)
    assert result["status"] == "unsupported_claim"


def test_less_than_equivalent_of_greater_than_is_accepted():
    row, base = pilot("hindi", "direct")
    text = reference_explanation(row, base)
    names = row["problem"]["entities"]
    values = row["problem"]["values"]
    high = values.index(max(values))
    low = 1 - high
    attr = "उम्र" if row["domain"] == "age" else "लंबाई"
    text = text.replace(
        f"{names[high]} की {attr} {names[low]} से ज़्यादा है",
        f"{names[low]} की {attr} {names[high]} से कम है",
    )
    assert check_output(row, text, base)["joint_correct"]


def test_cached_logits_and_generations_match_uncached_model():
    torch.manual_seed(81)
    config = ModelConfig(
        vocabulary_size=31,
        context_length=32,
        model_dimension=16,
        number_of_layers=2,
        number_of_heads=4,
        feed_forward_dimension=32,
        dropout=0,
        attention_dropout=0,
        embedding_dropout=0,
    )
    model = DecoderOnlyTransformer(config).eval()
    tokens = torch.randint(0, 31, (3, 7))
    logits, cache = cached_step(model, tokens)
    torch.testing.assert_close(logits, model(tokens).logits[:, -1])
    for _ in range(4):
        new = torch.randint(0, 31, (3, 1))
        tokens = torch.cat((tokens, new), dim=1)
        logits, cache = cached_step(model, new, cache)
        torch.testing.assert_close(logits, model(tokens).logits[:, -1], rtol=1e-4, atol=1e-6)
    assert greedy_cached(model, tokens.tolist(), 3, "cpu", 8) == greedy_batch(
        model, tokens.tolist(), 3, "cpu", 8
    )


def test_batch_progress_survives_torn_last_line(tmp_path):
    from lma_phase3.evaluate_proofs import append_progress, read_progress

    path = tmp_path / "progress.jsonl"
    assert read_progress(path) == []
    append_progress(path, [{"id": "a"}, {"id": "b"}])
    with path.open("a") as f:
        f.write('{"id":')
    assert read_progress(path) == [{"id": "a"}, {"id": "b"}]
    path.write_text('{broken\n{"id":"b"}\n')
    with pytest.raises(ValueError, match="before the last line"):
        read_progress(path)
