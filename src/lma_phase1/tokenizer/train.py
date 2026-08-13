"""Train separate SentencePiece candidates from a language's training split."""

from __future__ import annotations

import argparse
from pathlib import Path

from lma_phase1.io import atomic_write_json, iter_jsonl


def prepare_training_text(input_path: Path, output_path: Path, minimum_characters: int) -> int:
    """Write one normalized paragraph per line for SentencePiece training."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as output:
        for document in iter_jsonl(input_path):
            for paragraph in document.text.splitlines():
                paragraph = " ".join(paragraph.split())
                if len(paragraph) < minimum_characters:
                    continue
                output.write(paragraph)
                output.write("\n")
                count += 1
    return count


def train_candidates(args: argparse.Namespace) -> dict[str, object]:
    """Train all requested vocabulary sizes against the same text sample."""

    try:
        import sentencepiece as spm
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install project dependencies first: pip install -e .") from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_text = output_dir / "tokenizer_training.txt"
    if training_text.exists() and not args.rebuild_training_text:
        paragraph_count = sum(1 for _ in training_text.open(encoding="utf-8"))
    else:
        paragraph_count = prepare_training_text(
            Path(args.input), training_text, args.minimum_paragraph_characters
        )
    if paragraph_count == 0:
        raise SystemExit("No training paragraphs survived tokenizer-text preparation")
    # SentencePiece accepts 0 (use every line) or a sample strictly larger than
    # 100. Small pilots therefore need 0 instead of their literal line count.
    sampled_sentences = min(args.input_sentence_size, paragraph_count)
    input_sentence_size = 0 if sampled_sentences <= 100 else sampled_sentences

    models = []
    for vocabulary_size in args.vocabulary_sizes:
        model_prefix = output_dir / f"{args.language}_{args.model_type}_{vocabulary_size}"
        model_path = model_prefix.with_suffix(".model")
        vocab_path = model_prefix.with_suffix(".vocab")
        if model_path.exists() or vocab_path.exists():
            raise SystemExit(f"Refusing to overwrite existing tokenizer candidate: {model_prefix}")
        spm.SentencePieceTrainer.train(
            input=str(training_text),
            model_prefix=str(model_prefix),
            model_type=args.model_type,
            vocab_size=vocabulary_size,
            character_coverage=args.character_coverage,
            input_sentence_size=input_sentence_size,
            shuffle_input_sentence=True,
            normalization_rule_name="identity",
            remove_extra_whitespaces=False,
            split_digits=True,
            byte_fallback=False,
            hard_vocab_limit=False,
            train_extremely_large_corpus=True,
            max_sentence_length=16_384,
            num_threads=args.num_threads,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="<bos>",
            eos_piece="<eos>",
        )
        processor = spm.SentencePieceProcessor(model_file=str(model_path))
        models.append(
            {
                "requested_vocabulary_size": vocabulary_size,
                "actual_vocabulary_size": processor.vocab_size(),
                "model": str(model_path),
                "vocabulary": str(vocab_path),
            }
        )
    return {
        "language": args.language,
        "input": args.input,
        "model_type": args.model_type,
        "character_coverage": args.character_coverage,
        "normalization_rule": "identity (corpus is pre-normalized to NFC)",
        "training_paragraphs_available": paragraph_count,
        "input_sentence_size": input_sentence_size,
        "models": models,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Training-split JSONL(.gz)")
    parser.add_argument("--language", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-type", choices=("unigram", "bpe"), default="unigram")
    parser.add_argument("--vocabulary-sizes", nargs="+", type=int, default=[16000, 24000, 32000])
    parser.add_argument("--character-coverage", type=float, default=1.0)
    parser.add_argument("--input-sentence-size", type=int, default=5_000_000)
    parser.add_argument("--minimum-paragraph-characters", type=int, default=20)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--rebuild-training-text", action="store_true")
    parser.add_argument("--report", required=True)
    return parser


def main() -> None:
    """Run candidate tokenizer training."""

    args = build_parser().parse_args()
    if any(size < 1000 for size in args.vocabulary_sizes):
        raise SystemExit("Vocabulary sizes must be at least 1,000")
    report = train_candidates(args)
    atomic_write_json(Path(args.report), report)
    for model in report["models"]:
        print(f"model={model['model']} actual_vocabulary_size={model['actual_vocabulary_size']}")


if __name__ == "__main__":
    main()
