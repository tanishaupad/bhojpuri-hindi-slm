"""Unicode normalization and script-aware text quality measurements."""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

_HORIZONTAL_SPACE = re.compile(r"[\t\f\v\u00a0\u2000-\u200b\u202f\u205f\u3000]+")
_MULTIPLE_BLANK_LINES = re.compile(r"\n{3,}")
_MULTIPLE_SPACES = re.compile(r" {2,}")
_PIB_DATE_LINE = re.compile(r"^प्रविष्टि तिथि:")
_PIB_FOOTER_LINE = re.compile(r"^\((?:रिलीज़|रिलीज) आईडी:")
_PIB_SEPARATOR_LINE = re.compile(r"^\*{3,}$")
_PIB_SIGNATURE_LINE = re.compile(r"^[^\s/]{1,8}(?:/[^\s/]{1,8}){1,10}$")
_WIKIPEDIA_TRAILING_HEADINGS = {
    "references",
    "external links",
    "further reading",
    "notes",
    "इहो देखल जाय",
    "बाहरी कड़ी",
    "बाहरी कड़ियाँ",
    "संदर्भ",
    "स्रोत",
}


@dataclass(frozen=True, slots=True)
class TextMetrics:
    """Basic measurements used for corpus filtering and reporting."""

    characters: int
    letters: int
    words: int
    devanagari_letters: int
    latin_letters: int
    digits: int

    @property
    def devanagari_ratio(self) -> float:
        """Return the share of alphabetic characters written in Devanagari."""

        return self.devanagari_letters / self.letters if self.letters else 0.0

    @property
    def latin_ratio(self) -> float:
        """Return the share of alphabetic characters written in Latin script."""

        return self.latin_letters / self.letters if self.letters else 0.0


def normalize_unicode(text: str) -> str:
    """Normalize text to NFC while preserving meaningful line boundaries."""

    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u200c", "").replace("\u200d", "")
    text = "".join(
        character if character in {"\n", "\t"} or unicodedata.category(character) != "Cc" else " "
        for character in text
    )
    text = unicodedata.normalize("NFC", text)
    lines = []
    for line in text.splitlines():
        line = _HORIZONTAL_SPACE.sub(" ", line)
        line = _MULTIPLE_SPACES.sub(" ", line).strip()
        lines.append(line)
    return _MULTIPLE_BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def remove_repeated_lines(text: str, minimum_repeats: int = 3) -> str:
    """Remove lines repeated many times inside one document as boilerplate."""

    lines = text.splitlines()
    counts = Counter(line for line in lines if len(line) >= 20)
    repeated = {line for line, count in counts.items() if count >= minimum_repeats}
    return "\n".join(line for line in lines if line not in repeated).strip()


def clean_pib_article(text: str) -> tuple[str, dict[str, int]]:
    """Remove standard PIB metadata, footer text, and English-only embedded blocks."""

    kept_paragraphs = []
    removed_lines = 0
    removed_english_paragraphs = 0
    reached_footer = False
    for paragraph in re.split(r"\n\s*\n", text):
        kept_lines = []
        for line in paragraph.splitlines():
            line = line.strip()
            if _PIB_FOOTER_LINE.match(line) or line.startswith("इस विज्ञप्ति को इन भाषाओं में"):
                reached_footer = True
                break
            if (
                not line
                or _PIB_DATE_LINE.match(line)
                or _PIB_SEPARATOR_LINE.fullmatch(line)
                or _PIB_SIGNATURE_LINE.fullmatch(line)
            ):
                removed_lines += 1
                continue
            kept_lines.append(line)
        if reached_footer:
            break
        cleaned_paragraph = "\n".join(kept_lines).strip()
        if not cleaned_paragraph:
            continue
        metrics = text_metrics(cleaned_paragraph)
        if metrics.letters >= 20 and metrics.latin_ratio > 0.50 and metrics.devanagari_ratio < 0.40:
            removed_english_paragraphs += 1
            continue
        kept_paragraphs.append(cleaned_paragraph)
    return "\n\n".join(kept_paragraphs).strip(), {
        "removed_lines": removed_lines,
        "removed_english_paragraphs": removed_english_paragraphs,
    }


def clean_bhojpuri_wikipedia_article(text: str) -> tuple[str, dict[str, int]]:
    """Remove reference/link tails and English-dominant blocks from Bhojpuri Wikipedia."""

    kept_paragraphs = []
    removed_english_paragraphs = 0
    truncated_trailing_sections = 0
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if any(line.casefold() in _WIKIPEDIA_TRAILING_HEADINGS for line in lines):
            truncated_trailing_sections += 1
            break
        metrics = text_metrics(paragraph)
        if metrics.letters >= 20 and metrics.latin_ratio > 0.50:
            removed_english_paragraphs += 1
            continue
        kept_paragraphs.append(paragraph)
    return "\n\n".join(kept_paragraphs).strip(), {
        "removed_english_paragraphs": removed_english_paragraphs,
        "truncated_trailing_sections": truncated_trailing_sections,
    }


def text_metrics(text: str) -> TextMetrics:
    """Count words and script-specific alphabetic characters."""

    letters = devanagari = latin = digits = 0
    for character in text:
        if character.isdigit():
            digits += 1
        if not character.isalpha():
            continue
        letters += 1
        codepoint = ord(character)
        if 0x0900 <= codepoint <= 0x097F or 0xA8E0 <= codepoint <= 0xA8FF:
            devanagari += 1
        elif "LATIN" in unicodedata.name(character, ""):
            latin += 1
    return TextMetrics(
        characters=len(text),
        letters=letters,
        words=len(text.split()),
        devanagari_letters=devanagari,
        latin_letters=latin,
        digits=digits,
    )


def canonical_text(text: str) -> str:
    """Produce a conservative canonical form for duplicate comparison."""

    normalized = normalize_unicode(text).casefold()
    return " ".join(normalized.split())


def stable_text_hash(text: str) -> str:
    """Return a SHA-256 hash of canonical text."""

    return hashlib.sha256(canonical_text(text).encode("utf-8")).hexdigest()
