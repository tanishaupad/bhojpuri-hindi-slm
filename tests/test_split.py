from __future__ import annotations

from lma_phase1.schema import Document
from lma_phase1.split import assign_split


def make_document(document_id: str, group_id: str | None = None) -> Document:
    metadata = {"book_id": group_id} if group_id else {}
    return Document(
        document_id=document_id,
        language="hi",
        text="परीक्षण पाठ",
        source_name="test_source",
        collection_type="manual",
        collection_method="unit_test",
        license="CC0-1.0",
        collected_at="2026-08-13",
        metadata=metadata,
    )


def test_assignment_is_deterministic():
    document = make_document("doc_1")
    first = assign_split(document, "seed", 0.8, 0.1)
    second = assign_split(document, "seed", 0.8, 0.1)
    assert first == second


def test_documents_in_same_book_stay_together():
    first = make_document("book_page_1", "book_42")
    second = make_document("book_page_2", "book_42")
    assert assign_split(first, "seed", 0.8, 0.1) == assign_split(second, "seed", 0.8, 0.1)
