"""Write transparent symbolic-verification results without conflating coverage and truth."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from lma_phase3.generate import FAMILIES


def main() -> None:
    """Build overall/per-task tables and deterministic examples from complete runs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.results_dir / "protocol.json").read_text())
    results = {
        lang: json.loads((args.results_dir / lang / "summary.json").read_text())
        for lang in ("hindi", "bhojpuri")
    }
    lines = [
        "# Symbolic verification of answer-and-explanation generations",
        "",
        "This supplemental experiment asks each original pretrained model for a final answer followed",
        "by a short explanation. It preserves the earlier answer-only and answer-presence results.",
        "No additional language model judges the output, and no model was finetuned for this run.",
        "",
        "## Protocol",
        "",
        "The original reasoning_v1 validation, primary-test, unseen-name and unseen-wording sets are",
        "unchanged: 5,000 problems per language. An instruction requests an answer first and a reason",
        "after `कारण:`. Greedy decoding stops at EOS or 96 new tokens, instead of the old 16-token",
        "limit. Because the prompts and limit changed, this is a distinct experiment, not a revised",
        "score for the earlier generations. Output is generated from scratch using each language’s",
        "own checkpoint and tokenizer in float32 on CPU. No answer or reference proof is in the prompt.",
        "",
        "| Language | Added instruction |",
        "|---|---|",
    ]
    for lang, instruction in protocol["instructions"].items():
        lines.append(f"| {lang} | {instruction} |")
    lines += [
        "",
        "The added instruction is experiment-authored; the earlier user review covered the",
        "original question templates, not this new instruction. The 10,000 solver-derived reference",
        "explanations all pass the checker. Reference outputs need at most 51 Hindi / 59 Bhojpuri",
        "tokens including EOS, so the 96-token limit is sufficient for the target format. Maximum",
        "prompt lengths are 61 / 65 tokens, safely within the 256-token context with generation.",
        "",
        "Inference uses a KV cache over the unchanged attention layers. Unit tests compare cached",
        "and full-forward logits, and eight trained-checkpoint prompts (one per task/language) gave",
        "identical cached and uncached greedy token sequences for the 16-token cross-check.",
        "",
        "## What each checker requires",
        "",
        "| Task | Required explanation evidence |",
        "|---|---|",
        "| Direct | Both stated numeric facts and a supported comparison establishing the requested result. |",
        "| Largest/smallest | All three numeric facts and comparisons whose closure establishes the winner against every competitor. |",
        "| Equality | Both numeric facts and an explicit supported equality/inequality assertion. |",
        "| Two-hop | Both premise links and the supported endpoint conclusion; a bare conclusion is insufficient. |",
        "",
        "The parser recognizes complete Hindi/Bhojpuri value and comparison sentences, including",
        "greater/less formulations, equality/inequality, Devanagari digits, reordered sentences, and",
        "selected connective prefixes. Parsed statements are checked against numeric input values or",
        "every ranking consistent with the supplied relations. An invented value or reversed relation",
        "is unsupported. Negation outside the supported grammar cannot be accepted as an affirmative",
        "substring. The scorer never compares the explanation to one exact reference string.",
        "",
        "An unfamiliar sentence is **unverified**, not automatically factually wrong. A complete proof",
        "must contain no unknown sentences or unsupported claims. Unsupported names, mixed domains,",
        "and unrestricted prose may therefore reduce parser coverage. This is controlled-language",
        "verification, not a general natural-language correctness judge. It evaluates the stated",
        "justification, not the model’s internal reasoning process.",
        "",
        "## Overall results",
        "",
        "| Language / split | N | Answer accuracy | Explanation parseable | Explanation verified | Joint |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for lang, summary in results.items():
        for split, m in summary["metrics"].items():
            v = m["overall"]
            n = v["examples"]
            values = [
                f"{100 * v['rates'][k]:.2f}% ({v[k]}/{n})"
                for k in (
                    "answer_correct",
                    "explanation_parseable",
                    "explanation_verified",
                    "joint_correct",
                )
            ]
            lines.append(f"| {lang} / {split} | {n} | " + " | ".join(values) + " |")
    lines += [
        "",
        "All rates use the full split denominator. Answer accuracy scores only an explicitly",
        "identified canonical answer field; mention elsewhere does not count. Explanation verification",
        "is reported independently of the answer field. Joint correctness requires both. Parser",
        "coverage is the fraction with a nonempty reason whose complete text can be interpreted.",
        "Low coverage prevents a broad conclusion about explanation correctness.",
        "",
        "## Task-level results",
        "",
        "| Language / split / task | Answer correct | Parseable | Verified | Joint | N |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for lang, summary in results.items():
        for split, m in summary["metrics"].items():
            for family in FAMILIES:
                v = m["per_task"][family]
                lines.append(
                    f"| {lang} / {split} / {family} | {v['answer_correct']} | {v['explanation_parseable']} | {v['explanation_verified']} | {v['joint_correct']} | {v['examples']} |"
                )
    lines += [
        "",
        "## Primary-test verification outcomes",
        "",
        "| Language | Missing explanation | Unparsed text | Unsupported claims | Insufficient proof | Verified |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for lang, summary in results.items():
        statuses = summary["metrics"]["test"]["overall"]["statuses"]
        lines.append(
            f"| {lang} | "
            + " | ".join(
                str(statuses.get(k, 0))
                for k in (
                    "missing_explanation",
                    "unparsed_text",
                    "unsupported_claim",
                    "insufficient_justification",
                    "verified",
                )
            )
            + " |"
        )
    lines += [
        "",
        "## Examples",
        "",
        "For each language, the first primary-test example in each observed status is shown. This",
        "selection illustrates outcomes and is not representative of category frequencies.",
        "",
    ]
    for lang in results:
        with gzip.open(args.results_dir / lang / "test_predictions.jsonl.gz", "rt") as f:
            rows = [json.loads(line) for line in f]
        for status in sorted({r["checks"]["status"] for r in rows}):
            row = next(r for r in rows if r["checks"]["status"] == status)
            lines += [
                f"### {lang}: {status} ({row['id']})",
                "",
                row["prompt"],
                "",
                f"Reference: {row['reference_explanation_output']}",
                "",
                f"Generated: {json.dumps(row['generated'], ensure_ascii=False)}",
                "",
            ]
    lines += [
        "## Reproduction",
        "",
        "```bash",
        "python -m lma_phase3.evaluate_proofs \\",
        "  --artifact-root '/path/to/LMA_Individual_Project' \\",
        "  --dataset-root /path/to/reasoning_v1 \\",
        "  --output-dir /path/to/new-proof-results \\",
        "  --device cpu --batch-size 32",
        "",
        "python -m lma_phase3.summarize_proofs \\",
        "  --results-dir /path/to/new-proof-results \\",
        "  --output report/phase-3-proof-verification.md",
        "```",
        "",
        "Complete compressed generations, per-task metrics, protocol hashes, code hashes and trained-",
        "checkpoint cross-checks are saved in `report/results/generated/phase3/pretrained_proof_v2/`.",
        "",
    ]
    args.output.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
