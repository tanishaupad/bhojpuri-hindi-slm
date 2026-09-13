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
    rejected_path = tmp_path / "rejected.jsonl.gz"
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
            rejected_path=rejected_path,
            counters=counters,
        )
    )

    assert accepted == []
    assert counters["raw_documents"] == 1
    assert counters["rejected_source_excluded"] == 1
    assert counters["rejected_source_excluded:translated_partition"] == 1


def test_iter_processed_applies_anjoria_tail_cleaning(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl.gz"
    output_document = make_document("anjoria_pre2023")
    output_document.text = (
        "समाचार\n" + "ई भोजपुरी के साफ उपयोगी लेख हवे। " * 10 + "\n0 Comments\nRecent Posts\nकचरा"
    )
    write_jsonl(input_path, [output_document])
    counters = Counter()
    config = {
        "minimum_characters": 120,
        "minimum_words": 20,
        "minimum_devanagari_ratio": 0.7,
        "maximum_latin_ratio": 0.2,
        "maximum_digit_ratio": 0.25,
        "remove_repeated_lines": True,
        "excluded_source_names": [],
        "clean_anjoria": True,
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

    assert len(accepted) == 1
    assert "Recent Posts" not in accepted[0].text
    assert counters["anjoria_truncated_tail"] == 1


def test_iter_processed_applies_hindi_wikipedia_cleaning(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl.gz"
    document = make_document("hindi_wikipedia")
    document.language = "hi"
    document.text = "यह साफ हिन्दी लेख है। " * 20 + "\n\nसन्दर्भ\n\nEnglish reference"
    write_jsonl(input_path, [document])
    counters = Counter()
    config = {
        "minimum_characters": 120,
        "minimum_words": 20,
        "minimum_devanagari_ratio": 0.7,
        "maximum_latin_ratio": 0.2,
        "maximum_digit_ratio": 0.25,
        "remove_repeated_lines": True,
        "excluded_source_names": [],
        "clean_hindi_wikipedia": True,
    }

    accepted = list(
        iter_processed(
            [input_path],
            expected_language="hi",
            config=config,
            rejected_path=None,
            counters=counters,
        )
    )

    assert len(accepted) == 1
    assert "English reference" not in accepted[0].text
    assert counters["hindi_wikipedia_truncated_trailing_sections"] == 1


def test_iter_processed_applies_nepali_wikipedia_cleaning(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl.gz"
    document = make_document("nepali_wikipedia")
    document.language = "ne"
    document.text = "यो सफा नेपाली लेख हो। " * 20 + " श्रेणी:नेपाल"
    write_jsonl(input_path, [document])
    counters = Counter()
    config = {
        "minimum_characters": 120,
        "minimum_words": 20,
        "minimum_devanagari_ratio": 0.7,
        "maximum_latin_ratio": 0.2,
        "maximum_digit_ratio": 0.25,
        "remove_repeated_lines": True,
        "excluded_source_names": [],
        "clean_nepali_wikipedia": True,
    }

    accepted = list(
        iter_processed(
            [input_path],
            expected_language="ne",
            config=config,
            rejected_path=None,
            counters=counters,
        )
    )

    assert len(accepted) == 1
    assert "श्रेणी:" not in accepted[0].text
    assert counters["nepali_wikipedia_truncated_inline_categories"] == 1


def test_iter_processed_removes_residual_wikipedia_markup(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl.gz"
    document = make_document("hindi_wikipedia")
    document.language = "hi"
    document.text = (
        "यह साफ हिन्दी लेख है। " * 20
        + '\n<div class="center">पाठ</div> colspan="3" '
        + "{{cite web |title=English citation}}"
    )
    write_jsonl(input_path, [document])
    counters = Counter()
    config = {
        "minimum_characters": 120,
        "minimum_words": 20,
        "minimum_devanagari_ratio": 0.7,
        "maximum_latin_ratio": 0.2,
        "maximum_digit_ratio": 0.25,
        "remove_repeated_lines": True,
        "excluded_source_names": [],
        "clean_hindi_wikipedia": True,
    }

    accepted = list(
        iter_processed(
            [input_path],
            expected_language="hi",
            config=config,
            rejected_path=None,
            counters=counters,
        )
    )

    assert len(accepted) == 1
    assert "<div" not in accepted[0].text
    assert "colspan" not in accepted[0].text
    assert "{{cite" not in accepted[0].text
    assert counters["hindi_wikipedia_removed_residual_templates"] == 1


def test_iter_processed_rejects_wikipedia_cleaned_to_empty(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl.gz"
    rejected_path = tmp_path / "rejected.jsonl.gz"
    document = make_document("hindi_wikipedia")
    document.language = "hi"
    document.text = "{{अपूर्ण साँचा\n" + "यह कच्चा पाठ वापस नहीं रखा जाना चाहिए। " * 20
    write_jsonl(input_path, [document])
    counters = Counter()
    config = {
        "minimum_characters": 120,
        "minimum_words": 20,
        "minimum_devanagari_ratio": 0.7,
        "maximum_latin_ratio": 0.2,
        "maximum_digit_ratio": 0.25,
        "remove_repeated_lines": True,
        "excluded_source_names": [],
        "clean_hindi_wikipedia": True,
    }

    accepted = list(
        iter_processed(
            [input_path],
            expected_language="hi",
            config=config,
            rejected_path=rejected_path,
            counters=counters,
        )
    )

    assert accepted == []
    assert counters["hindi_wikipedia_cleaned_to_empty"] == 1
    assert counters["rejected_empty_not_persisted"] == 1


def test_iter_processed_rejects_explicitly_translated_source_document(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl.gz"
    document = make_document("hindi_wikisource")
    document.language = "hi"
    document.text = "अनुवादक: परीक्षण लेखक\n" + "यह हिन्दी पुस्तक का पाठ है। " * 20
    write_jsonl(input_path, [document])
    counters = Counter()
    config = {
        "minimum_characters": 120,
        "minimum_words": 20,
        "minimum_devanagari_ratio": 0.7,
        "maximum_latin_ratio": 0.2,
        "maximum_digit_ratio": 0.25,
        "remove_repeated_lines": True,
        "excluded_source_names": [],
        "exclude_translation_marked_sources": ["hindi_wikisource"],
    }

    accepted = list(
        iter_processed(
            [input_path],
            expected_language="hi",
            config=config,
            rejected_path=None,
            counters=counters,
        )
    )

    assert accepted == []
    assert counters["rejected_translation_marker"] == 1


def test_iter_processed_counts_empty_rejection_without_serializing_it(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl.gz"
    rejected_path = tmp_path / "rejected.jsonl.gz"
    document = make_document("hindi_wikipedia")
    document.language = "hi"
    document.text = ("यह पंक्ति पूरी तरह दोहराई गई है और हटेगी।\n" * 3).strip()
    write_jsonl(input_path, [document])
    counters = Counter()
    config = {
        "minimum_characters": 120,
        "minimum_words": 20,
        "minimum_devanagari_ratio": 0.7,
        "maximum_latin_ratio": 0.2,
        "maximum_digit_ratio": 0.25,
        "remove_repeated_lines": True,
        "excluded_source_names": [],
    }

    accepted = list(
        iter_processed(
            [input_path],
            expected_language="hi",
            config=config,
            rejected_path=rejected_path,
            counters=counters,
        )
    )

    assert accepted == []
    assert counters["rejected_empty_not_persisted"] == 1
    assert not rejected_path.exists()


def test_iter_processed_applies_clean_nepali_wikipedia(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl.gz"
    document = make_document("nepali_wikipedia")
    document.language = "ne"
    document.text = (
        "thumb|280px|सन् १७२० मा फ्रान्सका एक अभिनेता\n"
        "अभिनेता त्यो पुरुष कलाकार हो जो एउटा चलचित्र वा नाटकमा कुनै चरित्रको अभिनय गर्दछ । "
        "अभिनेता परिकल्पना तथा दर्शक माझ माध्यमको कार्य गर्ने गर्छ । यो एक महत्त्वपूर्ण नेपाली लेख हो ।\n\n"
        "सन्दर्भ सामग्रीहरू\n\nश्रेणी:कलाकार"
    )
    write_jsonl(input_path, [document])
    counters = Counter()
    config = {
        "minimum_characters": 100,
        "minimum_words": 15,
        "minimum_devanagari_ratio": 0.7,
        "maximum_latin_ratio": 0.2,
        "maximum_digit_ratio": 0.25,
        "clean_nepali_wikipedia": True,
        "remove_repeated_lines": True,
    }

    accepted = list(
        iter_processed(
            [input_path],
            expected_language="ne",
            config=config,
            rejected_path=None,
            counters=counters,
        )
    )

    assert len(accepted) == 1
    assert "280px|" not in accepted[0].text
    assert "thumb|" not in accepted[0].text
    assert "सन्दर्भ सामग्रीहरू" not in accepted[0].text
    assert "श्रेणी:कलाकार" not in accepted[0].text
    assert accepted[0].text.startswith("अभिनेता त्यो पुरुष कलाकार हो")
    assert counters["nepali_wikipedia_truncated_trailing_sections"] >= 1
