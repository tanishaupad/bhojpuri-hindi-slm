from lma_phase1.collectors.blogger import (
    entry_identifier,
    entry_labels,
    entry_url,
    extract_entry_text,
)


def test_extract_entry_text_removes_embed_and_normalizes_lines() -> None:
    html = """
    <div><p>पहिल पाँति</p><script>bad()</script><p>दूसर   पाँति</p></div>
    <iframe src="video"></iframe><div class="pvc_stats">123</div>
    <blockquote class="twitter-tweet">embedded post</blockquote>
    <p>स्रोत https://example.com/story</p>
    """

    assert extract_entry_text(html) == "पहिल पाँति\nदूसर पाँति\nस्रोत"


def test_entry_metadata_helpers() -> None:
    entry = {
        "id": {"$t": "tag:blogger.com,1999:blog-1.post-987"},
        "category": [{"term": "कहानी"}, {"term": "अनुवाद"}],
        "link": [
            {"rel": "self", "href": "https://example.com/feed/987"},
            {"rel": "alternate", "href": "https://example.com/post.html"},
        ],
    }

    assert entry_identifier(entry) == "post-987"
    assert entry_labels(entry) == ["कहानी", "अनुवाद"]
    assert entry_url(entry) == "https://example.com/post.html"
