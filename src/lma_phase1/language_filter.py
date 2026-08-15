"""Distinguish Bhojpuri-dominant text from Hindi using character n-grams."""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lma_phase1.io import atomic_write_json, iter_jsonl, write_jsonl
from lma_phase1.schema import Document

_SPACES = re.compile(r"\s+")


def normalize_for_language_id(text: str) -> str:
    """Keep Devanagari letters/marks and word boundaries for language ID."""

    characters = []
    for character in text:
        codepoint = ord(character)
        if 0x0900 <= codepoint <= 0x097F or 0xA8E0 <= codepoint <= 0xA8FF:
            characters.append(character)
        else:
            characters.append(" ")
    return _SPACES.sub(" ", "".join(characters)).strip()


def character_ngrams(text: str, size: int) -> Iterator[str]:
    """Yield fixed-width character n-grams."""

    if len(text) < size:
        return
    for index in range(len(text) - size + 1):
        yield text[index : index + size]


@dataclass(slots=True)
class NgramModel:
    """Additively smoothed character n-gram language model."""

    counts: Counter[str]
    total: int
    vocabulary_size: int
    alpha: float = 0.1

    def average_log_probability(self, text: str, size: int) -> float:
        """Return mean smoothed log probability for a normalized string."""

        denominator = self.total + self.alpha * self.vocabulary_size
        score = 0.0
        observed = 0
        for ngram in character_ngrams(text, size):
            score += math.log((self.counts.get(ngram, 0) + self.alpha) / denominator)
            observed += 1
        return score / observed if observed else float("-inf")


def build_models(
    bhojpuri_texts: Iterable[str],
    hindi_texts: Iterable[str],
    size: int = 4,
) -> tuple[NgramModel, NgramModel]:
    """Build balanced Bhojpuri and Hindi models from normalized strings."""

    bhojpuri_counts: Counter[str] = Counter()
    hindi_counts: Counter[str] = Counter()
    for text in bhojpuri_texts:
        bhojpuri_counts.update(character_ngrams(normalize_for_language_id(text), size))
    for text in hindi_texts:
        hindi_counts.update(character_ngrams(normalize_for_language_id(text), size))
    vocabulary_size = max(1, len(set(bhojpuri_counts).union(hindi_counts)))
    return (
        NgramModel(bhojpuri_counts, sum(bhojpuri_counts.values()), vocabulary_size),
        NgramModel(hindi_counts, sum(hindi_counts.values()), vocabulary_size),
    )


def score_difference(
    text: str, bhojpuri_model: NgramModel, hindi_model: NgramModel, size: int
) -> float:
    """Return Bhojpuri mean log probability minus Hindi mean log probability."""

    normalized = normalize_for_language_id(text)
    return bhojpuri_model.average_log_probability(
        normalized, size
    ) - hindi_model.average_log_probability(normalized, size)


def percentile(values: list[float], probability: float) -> float:
    """Return a deterministic nearest-rank percentile."""

    if not values:
        raise ValueError("cannot calculate a percentile of an empty sample")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def is_holdout(document_id: str, modulus: int) -> bool:
    """Assign a stable fraction of seed documents to calibration."""

    digest = hashlib.sha256(document_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % modulus == 0


def collect_seed_texts(
    paths: list[Path], expected_language: str, maximum_characters: int, holdout_modulus: int
) -> tuple[list[str], list[str], dict[str, int]]:
    """Load a bounded training sample and deterministic calibration holdout."""

    training = []
    holdout = []
    training_characters = 0
    examined = 0
    for path in paths:
        for document in iter_jsonl(path):
            if document.language != expected_language:
                continue
            examined += 1
            normalized = normalize_for_language_id(document.text)
            if len(normalized) < 100:
                continue
            if is_holdout(document.document_id, holdout_modulus):
                if len(holdout) < 2000:
                    holdout.append(document.text)
                continue
            if training_characters >= maximum_characters:
                continue
            remaining = maximum_characters - training_characters
            training.append(document.text[:remaining])
            training_characters += min(len(document.text), remaining)
    return training, holdout, {
        "documents_examined": examined,
        "training_documents": len(training),
        "training_characters": training_characters,
        "holdout_documents": len(holdout),
    }


def score_summary(values: list[float]) -> dict[str, float | int]:
    """Return report-ready score distribution summaries."""

    return {
        "count": len(values),
        "minimum": min(values),
        "p01": percentile(values, 0.01),
        "p05": percentile(values, 0.05),
        "median": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "maximum": max(values),
    }


def calibrated_thresholds(
    bhojpuri_scores: list[float], hindi_scores: list[float]
) -> tuple[float, float]:
    """Choose conservative thresholds from both seed-language holdouts.

    Requiring the accepted score to clear the Bhojpuri fifth percentile keeps
    the automatically accepted set high precision.  The first percentile is a
    useful lower boundary for human audit, while the corresponding Hindi upper
    tails prevent weakly Hindi-like scores from defining either cutoff.
    """

    accept = max(percentile(bhojpuri_scores, 0.05), percentile(hindi_scores, 0.995))
    borderline = max(
        percentile(bhojpuri_scores, 0.01), percentile(hindi_scores, 0.95)
    )
    return accept, min(borderline, accept)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Train, calibrate, and apply the language discriminator."""

    targets = [Path(args.accepted), Path(args.borderline), Path(args.rejected)]
    if any(path.exists() for path in targets):
        raise SystemExit("Refusing to append to an existing language-filter output")

    bho_train, bho_holdout, bho_seed_stats = collect_seed_texts(
        [Path(path) for path in args.bhojpuri_seed],
        "bho",
        args.maximum_training_characters,
        args.holdout_modulus,
    )
    hi_train, hi_holdout, hi_seed_stats = collect_seed_texts(
        [Path(path) for path in args.hindi_seed],
        "hi",
        args.maximum_training_characters,
        args.holdout_modulus,
    )
    if not bho_train or not hi_train or not bho_holdout or not hi_holdout:
        raise SystemExit("Both languages need non-empty training and holdout seed samples")
    bho_model, hi_model = build_models(bho_train, hi_train, args.ngram_size)
    bho_scores = [
        score_difference(text, bho_model, hi_model, args.ngram_size) for text in bho_holdout
    ]
    hi_scores = [
        score_difference(text, bho_model, hi_model, args.ngram_size) for text in hi_holdout
    ]
    calibrated_accept, calibrated_borderline = calibrated_thresholds(
        bho_scores, hi_scores
    )
    accept_threshold = (
        args.accept_threshold if args.accept_threshold is not None else calibrated_accept
    )
    borderline_threshold = (
        args.borderline_threshold
        if args.borderline_threshold is not None
        else calibrated_borderline
    )
    if borderline_threshold > accept_threshold:
        raise SystemExit("borderline threshold cannot exceed accept threshold")

    counters: Counter[str] = Counter()
    batches: dict[str, list[Document]] = {"accepted": [], "borderline": [], "rejected": []}
    output_paths = {
        "accepted": Path(args.accepted),
        "borderline": Path(args.borderline),
        "rejected": Path(args.rejected),
    }
    target_scores: list[float] = []
    for input_path in args.input:
        for document in iter_jsonl(Path(input_path)):
            counters["examined"] += 1
            score = score_difference(document.text, bho_model, hi_model, args.ngram_size)
            target_scores.append(score)
            if score >= accept_threshold:
                decision = "accepted"
            elif score >= borderline_threshold:
                decision = "borderline"
            else:
                decision = "rejected"
            counters[decision] += 1
            document.metadata = {
                **document.metadata,
                "hindi_bhojpuri_discrimination": {
                    "method": f"character_{args.ngram_size}gram_log_likelihood_ratio",
                    "score_bho_minus_hi": score,
                    "accept_threshold": accept_threshold,
                    "borderline_threshold": borderline_threshold,
                    "decision": decision,
                },
            }
            batches[decision].append(document)
            if len(batches[decision]) >= 1000:
                write_jsonl(output_paths[decision], batches[decision], append=True)
                batches[decision].clear()
    for decision, batch in batches.items():
        if batch:
            write_jsonl(output_paths[decision], batch, append=True)

    report = {
        "method": "character_ngram_log_likelihood_ratio",
        "ngram_size": args.ngram_size,
        "maximum_training_characters_per_language": args.maximum_training_characters,
        "seed_inputs": {
            "bhojpuri": args.bhojpuri_seed,
            "hindi": args.hindi_seed,
        },
        "seed_stats": {"bhojpuri": bho_seed_stats, "hindi": hi_seed_stats},
        "calibration_scores": {
            "bhojpuri": score_summary(bho_scores),
            "hindi": score_summary(hi_scores),
        },
        "thresholds": {
            "accepted": accept_threshold,
            "borderline": borderline_threshold,
            "default_calibration": {
                "accepted": "max(bhojpuri_holdout_p05, hindi_holdout_p99.5)",
                "borderline": "max(bhojpuri_holdout_p01, hindi_holdout_p95)",
            },
        },
        "calibration_checks": {
            "bhojpuri_holdout_accepted_fraction": sum(
                score >= accept_threshold for score in bho_scores
            )
            / len(bho_scores),
            "hindi_holdout_incorrectly_accepted_fraction": sum(
                score >= accept_threshold for score in hi_scores
            )
            / len(hi_scores),
        },
        "target_inputs": args.input,
        "target_counts": dict(sorted(counters.items())),
        "target_scores": score_summary(target_scores),
    }
    atomic_write_json(Path(args.report), report)
    return report


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bhojpuri-seed", action="append", required=True)
    parser.add_argument("--hindi-seed", action="append", required=True)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--accepted", required=True)
    parser.add_argument("--borderline", required=True)
    parser.add_argument("--rejected", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--ngram-size", type=int, default=4)
    parser.add_argument("--maximum-training-characters", type=int, default=5_000_000)
    parser.add_argument("--holdout-modulus", type=int, default=10)
    parser.add_argument("--accept-threshold", type=float)
    parser.add_argument("--borderline-threshold", type=float)
    return parser


def main() -> None:
    """Run the language-filter CLI."""

    args = build_parser().parse_args()
    report = run(args)
    print(" ".join(f"{key}={value}" for key, value in report["target_counts"].items()))


if __name__ == "__main__":
    main()
