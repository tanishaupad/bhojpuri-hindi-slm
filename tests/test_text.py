from __future__ import annotations

from lma_phase1.text import (
    canonical_text,
    clean_bhojpuri_wikipedia_article,
    clean_pib_article,
    normalize_unicode,
    text_metrics,
)


def test_unicode_and_whitespace_normalization():
    text = "  नमस्ते\u00a0  दुनिया  \r\n\r\n\r\n  ठीक है  "
    assert normalize_unicode(text) == "नमस्ते दुनिया\n\nठीक है"


def test_devanagari_metrics():
    metrics = text_metrics("भोजपुरी भाषा में text 123")
    assert metrics.words == 5
    assert metrics.devanagari_letters > metrics.latin_letters
    assert metrics.digits == 3


def test_canonical_text_collapses_layout_differences():
    assert canonical_text("नमस्ते   दुनिया") == canonical_text("नमस्ते\nदुनिया")


def test_clean_pib_article_removes_metadata_footer_and_english_block():
    text = """राष्ट्रपति सचिवालय
हिन्दी शीर्षक

प्रविष्टि तिथि: 14 AUG 2026 3:19PM by PIB Delhi

यह मुख्य हिन्दी अनुच्छेद है और इसे रखा जाना चाहिए।

This is an embedded English social media post that should be removed completely.

पीके/केसी/डीवी

(रिलीज़ आईडी: 123) आगंतुक पटल : 10
इस विज्ञप्ति को इन भाषाओं में पढ़ें: English, Urdu"""

    cleaned, report = clean_pib_article(text)

    assert "यह मुख्य हिन्दी अनुच्छेद" in cleaned
    assert "प्रविष्टि तिथि" not in cleaned
    assert "embedded English" not in cleaned
    assert "रिलीज़ आईडी" not in cleaned
    assert "पीके/केसी/डीवी" not in cleaned
    assert report["removed_english_paragraphs"] == 1


def test_clean_bhojpuri_wikipedia_removes_english_external_links_tail():
    text = """अप्लेशियन परबत उत्तर अमेरिका में एगो पुरान परबत सिस्टम बाटे।

ई पहाड़ी क्षेत्र बहुत लमहर बा आ एकर इतिहास पुरान बा।

संदर्भ

External link to a forest inventory and university research center.
Another English-only resource."""

    cleaned, report = clean_bhojpuri_wikipedia_article(text)

    assert "अप्लेशियन परबत" in cleaned
    assert "External link" not in cleaned
    assert "संदर्भ" not in cleaned
    assert report["truncated_trailing_sections"] == 1


def test_clean_bhojpuri_wikipedia_removes_pure_english_paragraph():
    text = """This entire paragraph is English catalogue text and should not enter the corpus.

ई छोट भोजपुरी अनुच्छेद राखल जाई।"""

    cleaned, report = clean_bhojpuri_wikipedia_article(text)

    assert cleaned == "ई छोट भोजपुरी अनुच्छेद राखल जाई।"
    assert report["removed_english_paragraphs"] == 1
