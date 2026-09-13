"""Normalize and quality-filter raw corpus documents without losing provenance."""

from __future__ import annotations

import argparse
import hashlib
import tomllib
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from lma_phase1.io import atomic_write_json, iter_jsonl, write_jsonl
from lma_phase1.schema import Document
from lma_phase1.text import (
    clean_anjoria_article,
    clean_bhojpuri_wikipedia_article,
    clean_pib_article,
    has_translation_credit,
    normalize_unicode,
    remove_repeated_lines,
    text_metrics,
)

DEFAULTS: dict[str, Any] = {
    "minimum_characters": 200,
    "minimum_words": 30,
    "minimum_devanagari_ratio": 0.70,
    "maximum_latin_ratio": 0.20,
    "maximum_digit_ratio": 0.25,
    "remove_repeated_lines": True,
    "excluded_source_names": [],
    "exclude_translation_marked_sources": [],
    "clean_pib_boilerplate": False,
    "clean_hindi_wikipedia": False,
    "clean_bhojpuri_wikipedia": False,
    "clean_nepali_wikipedia": False,
    "clean_anjoria": False,
}


def load_config(path: Path | None) -> dict[str, Any]:
    """Load preprocessing thresholds from TOML and merge them with defaults."""

    config = dict(DEFAULTS)
    if path:
        with path.open("rb") as handle:
            config.update(tomllib.load(handle))
    return config


def rejection_reasons(
    document: Document, config: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    """Return failed quality rules and serializable text measurements."""

    metrics = text_metrics(document.text)
    nonspace_characters = max(1, sum(not char.isspace() for char in document.text))
    digit_ratio = metrics.digits / nonspace_characters
    reasons = []
    if metrics.characters < int(config["minimum_characters"]):
        reasons.append("too_short_characters")
    if metrics.words < int(config["minimum_words"]):
        reasons.append("too_few_words")
    if metrics.devanagari_ratio < float(config["minimum_devanagari_ratio"]):
        reasons.append("low_devanagari_ratio")
    if metrics.latin_ratio > float(config["maximum_latin_ratio"]):
        reasons.append("high_latin_ratio")
    if digit_ratio > float(config["maximum_digit_ratio"]):
        reasons.append("high_digit_ratio")
    serialized = {
        "characters": metrics.characters,
        "letters": metrics.letters,
        "words": metrics.words,
        "devanagari_ratio": metrics.devanagari_ratio,
        "latin_ratio": metrics.latin_ratio,
        "digit_ratio": digit_ratio,
    }
    return reasons, serialized


def iter_processed(
    input_paths: list[Path],
    expected_language: str,
    config: dict[str, Any],
    rejected_path: Path | None,
    counters: Counter[str],
) -> Iterator[Document]:
    """Yield accepted normalized records and optionally persist rejected ones."""

    rejected_batch: list[Document] = []
    excluded_source_names = {
        str(source_name) for source_name in config.get("excluded_source_names", [])
    }
    translation_marked_sources = {
        str(source_name)
        for source_name in config.get("exclude_translation_marked_sources", [])
    }
    for input_path in input_paths:
        for document in iter_jsonl(input_path):
            counters["raw_documents"] += 1
            if document.language != expected_language:
                counters["rejected_language_mismatch"] += 1
                continue
            if document.source_name in excluded_source_names:
                counters["rejected_source_excluded"] += 1
                counters[f"rejected_source_excluded:{document.source_name}"] += 1
                continue
            original_hash = hashlib.sha256(document.text.encode("utf-8")).hexdigest()
            text = normalize_unicode(document.text)
            if config.get("remove_repeated_lines", True):
                text = remove_repeated_lines(text)
            source_cleaning: dict[str, Any] = {}
            if document.source_name == "pib_hindi" and config.get("clean_pib_boilerplate", False):
                text, pib_cleaning = clean_pib_article(text)
                source_cleaning["pib"] = pib_cleaning
                for key, value in pib_cleaning.items():
                    counters[f"pib_{key}"] += value
            if document.source_name in {"hindi_wikipedia", "hindi_wikibooks"} and config.get(
                "clean_hindi_wikipedia", False
            ):
                cleaned_wikipedia_text, wikipedia_cleaning = clean_bhojpuri_wikipedia_article(text)
                text = cleaned_wikipedia_text
                if not text:
                    wikipedia_cleaning["cleaned_to_empty"] = 1
                source_cleaning[document.source_name] = wikipedia_cleaning
                for key, value in wikipedia_cleaning.items():
                    counters[f"{document.source_name}_{key}"] += value
            if document.source_name in {
                "bhojpuri_wikipedia",
                "bhojpuri_corpus:wikipedia",
            } and config.get("clean_bhojpuri_wikipedia", False):
                cleaned_wikipedia_text, wikipedia_cleaning = clean_bhojpuri_wikipedia_article(text)
                text = cleaned_wikipedia_text
                if not text:
                    wikipedia_cleaning["cleaned_to_empty"] = 1
                source_cleaning["bhojpuri_wikipedia"] = wikipedia_cleaning
                for key, value in wikipedia_cleaning.items():
                    counters[f"bhojpuri_wikipedia_{key}"] += value
            if document.source_name == "nepali_wikipedia" and config.get(
                "clean_nepali_wikipedia", False
            ):
                cleaned_wikipedia_text, wikipedia_cleaning = clean_bhojpuri_wikipedia_article(text)
                text = cleaned_wikipedia_text
                if not text:
                    wikipedia_cleaning["cleaned_to_empty"] = 1
                source_cleaning["nepali_wikipedia"] = wikipedia_cleaning
                for key, value in wikipedia_cleaning.items():
                    counters[f"nepali_wikipedia_{key}"] += value
            if document.source_name == "anjoria_pre2023" and config.get(
                "clean_anjoria", False
            ):
                text, anjoria_cleaning = clean_anjoria_article(text)
                source_cleaning["anjoria"] = anjoria_cleaning
                for key, value in anjoria_cleaning.items():
                    if isinstance(value, int):
                        counters[f"anjoria_{key}"] += value
            document.text = text
            reasons, metrics = rejection_reasons(document, config)
            if document.source_name in translation_marked_sources and has_translation_credit(
                document.text, str(document.metadata.get("title", ""))
            ):
                reasons.append("translation_marker")
            document.metadata = {
                **document.metadata,
                "preprocessing": {
                    "unicode_normalization": "NFC",
                    "original_text_sha256": original_hash,
                    "input_artifact": input_path.name,
                    "metrics": metrics,
                "source_cleaning": source_cleaning,
                },
            }
            if reasons:
                for reason in reasons:
                    counters[f"rejected_{reason}"] += 1
                if rejected_path:
                    if not document.text.strip():
                        counters["rejected_empty_not_persisted"] += 1
                        continue
                    document.metadata["preprocessing"]["rejection_reasons"] = reasons
                    rejected_batch.append(document)
                    if len(rejected_batch) >= 1000:
                        write_jsonl(rejected_path, rejected_batch, append=True)
                        rejected_batch.clear()
                continue
            counters["accepted_documents"] += 1
            counters["accepted_characters"] += metrics["characters"]
            counters["accepted_words"] += metrics["words"]
            yield document
    if rejected_path and rejected_batch:
        write_jsonl(rejected_path, rejected_batch, append=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--rejected")
    parser.add_argument("--report", required=True)
    return parser


def main() -> None:
    """Run the preprocessing CLI."""

    args = build_parser().parse_args()
    output = Path(args.output)
    rejected = Path(args.rejected) if args.rejected else None
    for target in (output, rejected):
        if target and target.exists():
            raise SystemExit(f"Refusing to append to existing output: {target}")
    config = load_config(args.config)
    counters: Counter[str] = Counter()
    documents = iter_processed(
        input_paths=[Path(path) for path in args.input],
        expected_language=args.language,
        config=config,
        rejected_path=rejected,
        counters=counters,
    )
    write_jsonl(output, documents)
    report = {
        "language": args.language,
        "inputs": args.input,
        "output": args.output,
        "thresholds": {key: config[key] for key in DEFAULTS},
        "counts": dict(sorted(counters.items())),
    }
    atomic_write_json(Path(args.report), report)
    print(
        f"accepted={counters['accepted_documents']} "
        f"rejected={counters['raw_documents'] - counters['accepted_documents']}"
    )


if __name__ == "__main__":
    main()
