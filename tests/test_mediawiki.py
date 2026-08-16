from __future__ import annotations

from lma_phase1.collectors.mediawiki import (
    enumerate_pages,
    fetch_page_details,
    page_to_document,
    wikitext_to_plaintext,
)


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


def test_category_enumeration_uses_categorymembers_continuation():
    session = FakeSession(
        [
            {
                "query": {
                    "categorymembers": [
                        {"pageid": 11, "ns": 0, "title": "बटोहीया"},
                    ]
                },
                "continue": {"cmcontinue": "page|next", "continue": "-||"},
            }
        ]
    )

    pages, continuation = enumerate_pages(
        session,
        "https://wikisource.org/w/api.php",
        namespace=0,
        batch_size=50,
        continuation=None,
        category="Category:भोजपुरी",
    )

    assert pages[0]["pageid"] == 11
    assert continuation == {"cmcontinue": "page|next"}
    assert session.calls[0]["list"] == "categorymembers"
    assert session.calls[0]["cmtitle"] == "Category:भोजपुरी"


def test_missing_extract_falls_back_to_parsed_html():
    session = FakeSession(
        [
            {
                "query": {
                    "pages": [
                        {
                            "pageid": 5,
                            "title": "बटोहीया",
                            "fullurl": "https://wikisource.org/wiki/test",
                        }
                    ]
                }
            },
            {
                "parse": {
                    "text": (
                        "<div><p>सुन्दर सुभूमि भइया भारत के देसवा से</p>"
                        '<sup class="reference">[1]</sup></div>'
                    )
                }
            },
        ]
    )

    pages = fetch_page_details(
        session,
        "https://wikisource.org/w/api.php",
        [{"pageid": 5, "title": "बटोहीया"}],
    )

    assert pages[0]["extract"] == "सुन्दर सुभूमि भइया भारत के देसवा से"
    assert pages[0]["extraction_method"] == "mediawiki_parse_html"
    assert session.calls[1]["action"] == "parse"


def test_wikitext_to_plaintext_keeps_prose_and_link_labels():
    source = "'''भारत''' [[दक्षिण एशिया]] में एक देश है। {{स्रोतहीन}}"

    assert wikitext_to_plaintext(source) == "भारत दक्षिण एशिया में एक देश है।"


def test_revision_mode_fetches_content_in_one_batched_request():
    session = FakeSession(
        [
            {
                "query": {
                    "pages": [
                        {
                            "pageid": 1,
                            "title": "भारत",
                            "fullurl": "https://example/1",
                            "revisions": [
                                {
                                    "revid": 9,
                                    "timestamp": "2026-08-16T00:00:00Z",
                                    "slots": {"main": {"content": "'''भारत''' एक देश है।"}},
                                }
                            ],
                        }
                    ]
                }
            }
        ]
    )

    pages = fetch_page_details(
        session,
        "https://example/api.php",
        [{"pageid": 1, "title": "भारत"}],
        content_mode="revision",
    )

    assert pages[0]["extract"] == "भारत एक देश है।"
    assert pages[0]["extraction_method"] == "mediawiki_revision_wikitext"
    assert session.calls[0]["rvprop"] == "ids|timestamp|content"


def test_revision_mode_renders_wikisource_pages_transclusion():
    session = FakeSession(
        [
            {
                "query": {
                    "pages": [
                        {
                            "pageid": 5,
                            "title": "उपन्यास",
                            "revisions": [
                                {
                                    "slots": {
                                        "main": {
                                            "content": '<pages index="उपन्यास.djvu" from=1 to=3 />'
                                        }
                                    }
                                }
                            ],
                        }
                    ]
                }
            },
            {"parse": {"text": "<div><p>यह पुस्तक का पूरा पाठ है।</p></div>"}},
        ]
    )

    pages = fetch_page_details(
        session,
        "https://example/api.php",
        [{"pageid": 5, "title": "उपन्यास"}],
        content_mode="revision",
        render_empty_revisions=True,
    )

    assert pages[0]["extract"] == "यह पुस्तक का पूरा पाठ है।"
    assert pages[0]["extraction_method"] == "mediawiki_parse_html"
    assert session.calls[1]["action"] == "parse"
