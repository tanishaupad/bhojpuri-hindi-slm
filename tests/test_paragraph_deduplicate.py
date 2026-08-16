from __future__ import annotations

from lma_phase1.paragraph_deduplicate import (
    ParagraphIndex,
    clean_document_paragraphs,
    eligible_paragraph,
    paragraph_units,
)
from lma_phase1.schema import Document


def make_document(document_id: str, text: str) -> Document:
    return Document(
        document_id=document_id,
        language="bho",
        text=text,
        source_name="test",
        collection_type="manual",
        collection_method="test",
        license="test",
        collected_at="2026-08-16",
    )


def test_paragraph_units_preserve_nonempty_lines() -> None:
    assert paragraph_units("पहिला पंक्ति\n\nदोसरा पंक्ति\n") == [
        "पहिला पंक्ति",
        "दोसरा पंक्ति",
    ]


def test_short_units_are_not_eligible() -> None:
    assert not eligible_paragraph("छोट शीर्षक", 50, 5)


def test_corpus_wide_duplicate_paragraph_is_removed(tmp_path) -> None:
    repeated = "ई एगो लमहर साझा अनुच्छेद हवे जवना में पन्द्रह से अधिका शब्द जरूर मौजूद बा।"
    first = make_document("first", repeated + "\nपहिला छोट लाइन")
    second = make_document("second", repeated + "\nदोसरा छोट लाइन")
    index = ParagraphIndex(tmp_path / "paragraphs.sqlite")

    first_text, first_counts, first_hashes = clean_document_paragraphs(first, index, 50, 10)
    for paragraph_hash in first_hashes:
        index.add_paragraph(paragraph_hash, first.document_id)
    second_text, second_counts, _ = clean_document_paragraphs(second, index, 50, 10)
    index.close()

    assert repeated in first_text
    assert repeated not in second_text
    assert second_text == "दोसरा छोट लाइन"
    assert first_counts.get("paragraph_duplicates_removed", 0) == 0
    assert second_counts["paragraph_duplicates_removed"] == 1
