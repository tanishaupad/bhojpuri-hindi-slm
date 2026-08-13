from __future__ import annotations

from lma_phase1.text import canonical_text, normalize_unicode, text_metrics


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
