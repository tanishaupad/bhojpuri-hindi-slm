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
    "इन्हें भी देखें",
    "इन्हें भी देखिये",
    "इसे भी देखें",
    "सन्दर्भ",
    "सन्दर्भ सूची",
    "टिप्पणी",
    "टिप्पणियाँ",
    "बाहरी कड़ियाँ",
    "बाहरी सूत्र",
    "इहो देखल जाय",
    "बाहरी कड़ी",
    "संदर्भ",
    "स्रोत",
}
_ANJORIA_TAIL_MARKERS = {
    "submit a comment",
    "cancel reply",
    "login",
    "अंजोरिया फेसबुक पर",
    "recent posts",
    "श्रेणी-सूची",
    "अंजोरिया पर खोजीं -",
}
_ANJORIA_COMMENT_COUNT = re.compile(r"^\d+\s+comments?$", re.IGNORECASE)
_WIKIPEDIA_METADATA_LINE = re.compile(
    r"^(?:श्रेणी|category)\s*:|^__[A-Z]+__$", re.IGNORECASE
)
_WIKIPEDIA_MEDIA_LINE = re.compile(
    r"^(?:(?:left|right|center|none)\|)?thumb\|", re.IGNORECASE
)
_WIKIPEDIA_SOCIAL_HANDLE = re.compile(
    r"\b(?:instagram|facebook|twitter|youtube)\s+(?:id|handle)\s*=\s*\S+",
    re.IGNORECASE,
)
_HTML_COMMENT = re.compile(r"<!--[\s\S]*?-->")
_HTML_TAG = re.compile(r"</?[A-Za-z][^>]{0,500}>")
_WIKIPEDIA_TABLE_ATTRIBUTE = re.compile(
    r"\b(?:colspan|rowspan|class|style|scope|align|width|height)\s*=\s*"
    r'(?:"[^"]*"|\'[^\']*\'|[^\s|]*)',
    re.IGNORECASE,
)
_WIKIPEDIA_TABLE_MARKER_LINE = re.compile(r"^(?:\{\||\|\}|\|-|[!|]\s*[|!])")
_TRANSLATION_CREDIT = re.compile(
    r"(?:अनुवादक|अनूदित|अनुवादित|अनुवाद\s*[:：-]|translated\s+by|translation\s*[:：-])",
    re.IGNORECASE,
)


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

    text, removed_leading_templates = remove_leading_wikitext_templates(text)
    text, removed_residual_templates = remove_wikitext_templates(text)
    text = _HTML_COMMENT.sub("", text)
    text = _HTML_TAG.sub("", text)
    text = _WIKIPEDIA_TABLE_ATTRIBUTE.sub("", text)
    kept_paragraphs = []
    removed_english_paragraphs = 0
    removed_metadata_lines = 0
    truncated_trailing_sections = 0
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        lines = []
        for raw_line in paragraph.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if _WIKIPEDIA_METADATA_LINE.search(line):
                removed_metadata_lines += 1
                continue
            if _WIKIPEDIA_MEDIA_LINE.search(line):
                removed_metadata_lines += 1
                continue
            if _WIKIPEDIA_TABLE_MARKER_LINE.search(line):
                removed_metadata_lines += 1
                continue
            line = _WIKIPEDIA_SOCIAL_HANDLE.sub("", line).strip()
            line = line.replace("[[", "").replace("]]", "")
            if not line:
                removed_metadata_lines += 1
                continue
            lines.append(line)
        if not lines:
            continue
        if any(line.casefold() in _WIKIPEDIA_TRAILING_HEADINGS for line in lines):
            truncated_trailing_sections += 1
            break
        paragraph = "\n".join(lines)
        metrics = text_metrics(paragraph)
        if metrics.letters >= 20 and metrics.latin_ratio > 0.50:
            removed_english_paragraphs += 1
            continue
        kept_paragraphs.append(paragraph)
    return "\n\n".join(kept_paragraphs).strip(), {
        "removed_english_paragraphs": removed_english_paragraphs,
        "removed_leading_templates": removed_leading_templates,
        "removed_residual_templates": removed_residual_templates,
        "removed_metadata_lines": removed_metadata_lines,
        "truncated_trailing_sections": truncated_trailing_sections,
    }


def remove_leading_wikitext_templates(text: str) -> tuple[str, int]:
    """Remove balanced template blocks leaked at the start of rendered Wikipedia text."""

    remaining = text.lstrip()
    removed = 0
    while remaining.startswith("{{"):
        depth = 0
        index = 0
        while index < len(remaining) - 1:
            pair = remaining[index : index + 2]
            if pair == "{{":
                depth += 1
                index += 2
                continue
            if pair == "}}":
                depth -= 1
                index += 2
                if depth == 0:
                    remaining = remaining[index:].lstrip()
                    removed += 1
                    break
                continue
            index += 1
        else:
            break
    return remaining, removed


def remove_wikitext_templates(text: str) -> tuple[str, int]:
    """Remove balanced templates anywhere and truncate an unmatched template tail."""

    pieces: list[str] = []
    cursor = 0
    removed = 0
    while True:
        start = text.find("{{", cursor)
        if start < 0:
            pieces.append(text[cursor:])
            break
        pieces.append(text[cursor:start])
        depth = 1
        index = start + 2
        while index < len(text) - 1 and depth:
            pair = text[index : index + 2]
            if pair == "{{":
                depth += 1
                index += 2
            elif pair == "}}":
                depth -= 1
                index += 2
            else:
                index += 1
        removed += 1
        if depth:
            break
        cursor = index
    return "".join(pieces).strip(), removed


def clean_anjoria_article(text: str) -> tuple[str, dict[str, int | str | None]]:
    """Keep the Anjoria article body and drop its repeated navigation/comment tail."""

    lines = text.splitlines()
    cutoff = len(lines)
    marker = None
    for index, line in enumerate(lines):
        normalized = line.strip().casefold()
        if normalized in _ANJORIA_TAIL_MARKERS or _ANJORIA_COMMENT_COUNT.fullmatch(normalized):
            cutoff = index
            marker = line.strip()
            break
    kept = "\n".join(lines[:cutoff]).strip()
    removed = "\n".join(lines[cutoff:])
    return kept, {
        "truncated_tail": int(cutoff < len(lines)),
        "removed_lines": len(lines) - cutoff,
        "removed_characters": len(removed),
        "cutoff_marker": marker,
    }


def has_translation_credit(text: str, title: str | None = None) -> bool:
    """Detect explicit translation credits without guessing from fluency alone."""

    haystack = f"{title or ''}\n{text}"
    return bool(_TRANSLATION_CREDIT.search(haystack))


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
