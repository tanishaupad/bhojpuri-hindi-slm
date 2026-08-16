from __future__ import annotations

from lma_phase1.text import (
    clean_anjoria_article,
    canonical_text,
    clean_bhojpuri_wikipedia_article,
    clean_pib_article,
    has_translation_credit,
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


def test_wikipedia_cleaning_recognizes_hindi_trailing_sections():
    text = """यह मुख्य हिन्दी लेख है और इसका उपयोग प्रशिक्षण में किया जाएगा।

इन्हें भी देखें

दूसरा संबंधित लेख

सन्दर्भ

बाहरी कड़ियाँ"""

    cleaned, report = clean_bhojpuri_wikipedia_article(text)

    assert cleaned == "यह मुख्य हिन्दी लेख है और इसका उपयोग प्रशिक्षण में किया जाएगा।"
    assert report["truncated_trailing_sections"] == 1


def test_wikipedia_cleaning_removes_category_and_magic_word_lines():
    text = """यह मुख्य हिन्दी लेख है।

__NOTOC__

श्रेणी:हिन्दी साहित्य
Category:Hindi"""

    cleaned, report = clean_bhojpuri_wikipedia_article(text)

    assert cleaned == "यह मुख्य हिन्दी लेख है।"
    assert report["removed_metadata_lines"] == 3


def test_wikipedia_cleaning_removes_leaked_infobox_and_media_prefix():
    text = """{{Infobox television
| genre = सिटकॉम
| network = स्टार प्लस
}}यह मुख्य हिन्दी लेख है और इसे रखा जाना चाहिए।

right|thumb|300px|अनावश्यक चित्र कैप्शन

दूसरा उपयोगी हिन्दी अनुच्छेद। Instagram id=spam_handle"""

    cleaned, report = clean_bhojpuri_wikipedia_article(text)

    assert cleaned.startswith("यह मुख्य हिन्दी लेख")
    assert "Infobox" not in cleaned
    assert "right|thumb" not in cleaned
    assert "spam_handle" not in cleaned
    assert report["removed_leading_templates"] == 1


def test_wikipedia_cleaning_preserves_label_from_leaked_link_brackets():
    cleaned, _ = clean_bhojpuri_wikipedia_article(
        "यह [[होली ट्रिनिटी चर्च नाम का एक उपयोगी हिन्दी वाक्य है।"
    )

    assert cleaned == "यह होली ट्रिनिटी चर्च नाम का एक उपयोगी हिन्दी वाक्य है।"


def test_clean_anjoria_article_truncates_repeated_site_tail():
    text = """समाचार
ई मुख्य भोजपुरी लेख हवे आ एकरा के राखल जरूरी बा।
एहमें दोसर उपयोगी वाक्यो बा।
0 Comments
Submit a Comment
Cancel reply
पिछला लेख
Recent Posts
दोसर लेख के शीर्षक"""

    cleaned, report = clean_anjoria_article(text)

    assert cleaned.endswith("एहमें दोसर उपयोगी वाक्यो बा।")
    assert "Submit a Comment" not in cleaned
    assert "दोसर लेख के शीर्षक" not in cleaned
    assert report["truncated_tail"] == 1
    assert report["cutoff_marker"] == "0 Comments"
    assert report["removed_lines"] == 6


def test_clean_anjoria_article_leaves_body_without_tail_unchanged():
    text = "समाचार\nई पूरा लेख हवे आ एहमें कवनो साइट वाला पुछल्ला नइखे।"

    cleaned, report = clean_anjoria_article(text)

    assert cleaned == text
    assert report["truncated_tail"] == 0
    assert report["removed_lines"] == 0
    assert report["cutoff_marker"] is None


def test_translation_credit_detection_is_explicit_and_conservative():
    assert has_translation_credit("अनुवादक: राम प्रसाद")
    assert has_translation_credit("Translated by A. Reader")
    assert not has_translation_credit("इस लेख में अनुवाद के सिद्धांत की चर्चा है।")
