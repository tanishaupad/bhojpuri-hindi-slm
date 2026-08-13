from __future__ import annotations

from lma_phase1.collectors.mediawiki import fetch_page_details, page_to_document


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payloads):
        self.payloads = iter(payloads)
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append(dict(params))
        return FakeResponse(next(self.payloads))


def test_page_conversion_preserves_revision_metadata():
    page = {
        "pageid": 42,
        "ns": 0,
        "title": "परीक्षण",
        "extract": "यह परीक्षण लेख है।",
        "fullurl": "https://hi.wikipedia.org/wiki/test",
        "revisions": [{"revid": 99, "timestamp": "2026-08-13T00:00:00Z"}],
    }
    document = page_to_document(page, "hi", "hindi_wikipedia", "CC-BY-SA-4.0", "2026-08-13")
    assert document is not None
    assert document.document_id == "hindi_wikipedia_42"
    assert document.metadata["revision_id"] == 99


def test_disambiguation_page_is_skipped():
    page = {
        "pageid": 7,
        "extract": "अर्थ",
        "pageprops": {"disambiguation": ""},
    }
    assert page_to_document(page, "hi", "hindi_wikipedia", "CC-BY-SA-4.0", "2026-08-13") is None


def test_property_continuations_are_merged_into_complete_pages():
    session = FakeSession(
        [
            {
                "query": {
                    "pages": [
                        {
                            "pageid": 1,
                            "title": "एक",
                            "fullurl": "https://example/1",
                            "extract": "एक",
                        },
                        {"pageid": 2, "title": "दो", "fullurl": "https://example/2"},
                    ]
                },
                "continue": {"excontinue": 1, "continue": "||"},
            },
            {"query": {"pages": [{"pageid": 2, "extract": "दो"}]}},
        ]
    )
    pages = fetch_page_details(
        session,
        "https://example/api.php",
        [{"pageid": 1, "title": "एक"}, {"pageid": 2, "title": "दो"}],
    )
    assert [page["extract"] for page in pages] == ["एक", "दो"]
    assert session.calls[1]["excontinue"] == 1
