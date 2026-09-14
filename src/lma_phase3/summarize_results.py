"""Build the Phase 3 pretrained-baseline tables from saved predictions and metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lma_phase3.generate import FAMILIES


def main() -> None:
    """Summarize the complete baseline without selecting examples to inflate accuracy."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = {
        lang: json.loads((args.results_dir / lang / "summary.json").read_text())
        for lang in ("hindi", "bhojpuri")
    }
    lines = [
        "# Pretrained reasoning baselines",
        "",
        "Both models were evaluated before reasoning finetuning on the frozen reasoning_v1 release.",
        "Each model used its own original checkpoint and tokenizer. Input files and checkpoints",
        "were verified against recorded SHA-256 hashes. No training was performed.",
        "",
        "## Overall exact-match accuracy",
        "",
        "| Split | Hindi | Bhojpuri | Examples per language |",
        "|---|---:|---:|---:|",
    ]
    splits = ("validation", "test", "challenge_names", "challenge_wording")
    for split in splits:
        h, b = [results[lang]["splits"][split]["overall"] for lang in results]
        lines.append(
            f"| {split} | {100 * h['accuracy']:.2f}% ({h['correct']}/{h['examples']}) | "
            f"{100 * b['accuracy']:.2f}% ({b['correct']}/{b['examples']}) | {h['examples']} |"
        )
    lines += [
        "",
        "## Task-level accuracy",
        "",
        "| Split / task | Hindi | Bhojpuri | Examples per language |",
        "|---|---:|---:|---:|",
    ]
    for split in splits:
        for family in FAMILIES:
            h, b = [results[lang]["splits"][split]["per_task"][family] for lang in results]
            lines.append(
                f"| {split} / {family} | {100 * h['accuracy']:.2f}% | "
                f"{100 * b['accuracy']:.2f}% | {h['examples']} |"
            )
    lines += [
        "",
        "## Protocol and interpretation",
        "",
        "Greedy decoding; stop at EOS or 16 new tokens. Exact match compares the entire generated",
        "continuation with the canonical answer after Unicode NFC and whitespace normalization only.",
        "No punctuation removal, substring extraction, or multiple-choice restriction was used.",
        "The evaluation uses float32 on the local MPS GPU. Equal-length prompts are batched without",
        "padding; only final-position vocabulary logits are projected. Tests verify equivalence to",
        "the full forward pass and per-example generation, as well as independent EOS stopping.",
        "",
        "These are corpus-pretrained language models, not instruction-finetuned models. Exact-match",
        "failure can reflect failure to emit the required short answer or stop cleanly as well as",
        "incorrect comparison reasoning. Noncanonical output therefore does not by itself establish",
        "that the relation could not be represented internally. Keep the same scoring protocol after",
        "finetuning so that before/after differences remain comparable. Test/challenge results must",
        "not be used for checkpoint or hyperparameter selection.",
        "",
        "The name challenge holds names out of synthetic training only. The wording challenge holds",
        "out one reviewed question variant, while task structures and premise forms remain familiar.",
        "",
        "## Primary-test error categories",
        "",
        "| Language | Correct | Wrong canonical answer | Noncanonical output | Empty output | EOS stops |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for lang in results:
        m = results[lang]["splits"]["test"]["overall"]
        e = m["errors"]
        lines.append(
            f"| {lang} | {m['correct']} | {e.get('wrong_canonical_answer', 0)} | "
            f"{e.get('noncanonical_output', 0)} | {e.get('empty_output', 0)} | {m['stopped_at_eos']} |"
        )
    lines += [
        "",
        "## Qualitative examples",
        "",
        "Examples below are the first saved primary-test item in each observed outcome category;",
        "this deterministic selection does not represent the frequency of each category.",
        "",
    ]
    for lang in results:
        rows = [
            json.loads(line)
            for line in (args.results_dir / lang / "test_predictions.jsonl")
            .read_text()
            .splitlines()
        ]
        for category in (
            "correct",
            "wrong_canonical_answer",
            "noncanonical_output",
            "empty_output",
        ):
            row = next((r for r in rows if r["error_category"] == category), None)
            if row is None:
                continue
            lines += [
                f"### {lang}: {category} ({row['id']})",
                "",
                row["prompt"],
                "",
                f"Expected: **{row['reference']}**",
                "",
                f"Generated: {json.dumps(row['generated'], ensure_ascii=False)}",
                "",
                f"Stop: {row['stop_reason']}",
                "",
            ]
    lines += [
        "## Artifacts",
        "",
        "Full predictions, individual split metrics and run summaries are under",
        "`report/results/generated/phase3/pretrained_v1/{hindi,bhojpuri}/`.",
        "Summaries record checkpoint, tokenizer, dataset, evaluator and prediction hashes.",
        "The full validation and primary-test sets each contain 2,000 examples per language;",
        "each challenge contains 500. All requested examples were scored.",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
