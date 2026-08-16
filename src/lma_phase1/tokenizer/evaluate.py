"""Evaluate tokenizer candidates on held-out JSONL and create report-ready plots."""

from __future__ import annotations

import argparse
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from lma_phase1.io import atomic_write_json, iter_jsonl
from lma_phase1.text import text_metrics
from lma_phase1.tokenizer.common import normalize_tokenizer_input


def evaluate_model(model_path: str, input_path: Path, maximum_examples: int) -> dict[str, Any]:
    """Compute fertility, unknown rate, frequency statistics, and examples."""

    import sentencepiece as spm

    processor = spm.SentencePieceProcessor(model_file=model_path)
    frequencies: Counter[int] = Counter()
    total_tokens = total_words = total_characters = total_nonspace = total_unknown = 0
    total_documents = 0
    examples = []

    for document in iter_jsonl(input_path):
        tokenizer_text = normalize_tokenizer_input(document.text)
        token_ids = processor.encode(tokenizer_text, out_type=int)
        metrics = text_metrics(tokenizer_text)
        frequencies.update(token_ids)
        total_documents += 1
        total_tokens += len(token_ids)
        total_words += metrics.words
        total_characters += metrics.characters
        total_nonspace += sum(not character.isspace() for character in tokenizer_text)
        total_unknown += sum(token_id == processor.unk_id() for token_id in token_ids)
        if len(examples) < maximum_examples:
            sample = tokenizer_text[:300]
            examples.append(
                {
                    "document_id": document.document_id,
                    "text": sample,
                    "pieces": processor.encode(sample, out_type=str),
                }
            )

    observed_types = len(frequencies)
    singleton_types = sum(count == 1 for count in frequencies.values())
    rare_types = sum(count <= 5 for count in frequencies.values())
    top_tokens = [
        {"piece": processor.id_to_piece(token_id), "id": token_id, "count": count}
        for token_id, count in frequencies.most_common(50)
    ]
    return {
        "model": model_path,
        "vocabulary_size": processor.vocab_size(),
        "documents": total_documents,
        "tokens": total_tokens,
        "words": total_words,
        "characters": total_characters,
        "fertility_tokens_per_word": total_tokens / total_words if total_words else None,
        "characters_per_token": total_characters / total_tokens if total_tokens else None,
        "nonspace_characters_per_token": total_nonspace / total_tokens if total_tokens else None,
        "unknown_tokens": total_unknown,
        "unknown_token_rate": total_unknown / total_tokens if total_tokens else None,
        "observed_vocabulary_types": observed_types,
        "vocabulary_utilization": observed_types / processor.vocab_size(),
        "singleton_type_rate": singleton_types / observed_types if observed_types else None,
        "rare_type_rate_le_5": rare_types / observed_types if observed_types else None,
        "top_tokens": top_tokens,
        "examples": examples,
        "frequency_counts_descending": sorted(frequencies.values(), reverse=True),
    }


def plot_frequencies(results: list[dict[str, Any]], figure_path: Path, language: str) -> None:
    """Plot labeled log-log token rank-frequency curves for all candidates."""

    cache_root = Path(tempfile.gettempdir()) / "lma-phase1-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    import matplotlib.pyplot as plt

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    _, axis = plt.subplots(figsize=(8, 5))
    for result in results:
        counts = result["frequency_counts_descending"]
        ranks = range(1, len(counts) + 1)
        axis.plot(ranks, counts, label=f"vocab={result['vocabulary_size']}", linewidth=1.2)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_title(f"{language}: held-out token rank-frequency distribution")
    axis.set_xlabel("Token frequency rank (log scale)")
    axis.set_ylabel("Token count (log scale)")
    axis.legend()
    axis.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--input", required=True, help="Held-out validation JSONL(.gz)")
    parser.add_argument("--language", required=True)
    parser.add_argument("--maximum-examples", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--figure", required=True)
    return parser


def main() -> None:
    """Run held-out tokenizer evaluation."""

    args = build_parser().parse_args()
    try:
        import sentencepiece  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install project dependencies first: pip install -e .") from exc
    results = [
        evaluate_model(model_path, Path(args.input), args.maximum_examples)
        for model_path in args.model
    ]
    plot_frequencies(results, Path(args.figure), args.language)
    serializable = []
    for result in results:
        copy = dict(result)
        copy.pop("frequency_counts_descending")
        serializable.append(copy)
    report = {"language": args.language, "held_out_input": args.input, "candidates": serializable}
    atomic_write_json(Path(args.output), report)
    for result in results:
        print(
            f"vocab={result['vocabulary_size']} "
            f"fertility={result['fertility_tokens_per_word']:.4f} "
            f"unk_rate={result['unknown_token_rate']:.8f}"
        )


if __name__ == "__main__":
    main()
