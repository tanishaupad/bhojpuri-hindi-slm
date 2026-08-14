from collections import Counter

from lma_phase1.io import write_jsonl
from lma_phase1.preprocess import iter_processed, rejection_reasons
from lma_phase1.schema import Document


def make_document(source_name: str = "native_source") -> Document:
    return Document(
        document_id="doc-1",
        language="bho",
        text="ई भोजपुरी के एगो साफ आ स्वाभाविक वाक्य हवे। " * 10,
        source_name=source_name,
        collection_type="downloaded",
        collection_method="test",
        license="test-license",
        collected_at="2026-08-14T00:00:00Z",
    )


def test_rejection_reasons_accepts_clean_devanagari_text() -> None:
    config = {
        "minimum_characters": 120,
        "minimum_words": 20,
        "minimum_devanagari_ratio": 0.7,
        "maximum_latin_ratio": 0.2,
        "maximum_digit_ratio": 0.25,
    }

    reasons, metrics = rejection_reasons(make_document(), config)

    assert reasons == []
    assert metrics["devanagari_ratio"] == 1.0
    assert Counter(reasons) == Counter()


def test_iter_processed_excludes_configured_source(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl.gz"
    write_jsonl(input_path, [make_document("translated_partition")])
    counters = Counter()
    config = {
        "minimum_characters": 120,
        "minimum_words": 20,
        "minimum_devanagari_ratio": 0.7,
        "maximum_latin_ratio": 0.2,
        "maximum_digit_ratio": 0.25,
        "remove_repeated_lines": True,
        "excluded_source_names": ["translated_partition"],
    }

    accepted = list(
        iter_processed(
            [input_path],
            expected_language="bho",
            config=config,
            rejected_path=None,
            counters=counters,
        )
    )

    assert accepted == []
    assert counters["raw_documents"] == 1
    assert counters["rejected_source_excluded"] == 1
    assert counters["rejected_source_excluded:translated_partition"] == 1
