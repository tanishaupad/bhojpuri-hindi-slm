from lma_phase1.collectors.web import extract_text_and_links


def test_extract_text_prefers_selector_order_over_document_order() -> None:
    html = """
    <article>
      <div class="entry-content"><p>असल लेख</p></div>
      <div class="comments">Submit a Comment</div>
    </article>
    """

    text, _, _ = extract_text_and_links(
        html, "https://example.com/post", ".entry-content, article"
    )

    assert text == "असल लेख"
