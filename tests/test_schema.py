from __future__ import annotations

import pytest

from lma_phase1.schema import Document


def make_document(**overrides):
    values = {
        "document_id": "hi_test_1",
        "language": "hi",
        "text": "यह एक परीक्षण दस्तावेज़ है।",
        "source_name": "test",
        "collection_type": "manual",
        "collection_method": "unit_test",
        "license": "CC0-1.0",
        "collected_at": "2026-08-13",
    }
    values.update(overrides)
    return Document(**values)


def test_round_trip():
    document = make_document(metadata={"page": 1})
    assert Document.from_dict(document.to_dict()) == document


def test_rejects_invalid_collection_type():
    document = make_document(collection_type="mystery")
    with pytest.raises(ValueError, match="collection_type"):
        document.validate()


def test_rejects_whitespace_in_id():
    document = make_document(document_id="bad id")
    with pytest.raises(ValueError, match="whitespace"):
        document.validate()
