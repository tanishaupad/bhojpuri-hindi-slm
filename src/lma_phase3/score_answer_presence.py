"""Post-hoc answer-presence scoring of saved reasoning generations.

This supplemental metric credits a reference answer mentioned anywhere as a whole
name/label. It does not identify a final conclusion or prove reasoning correctness.
The original strict exact-match baseline is preserved.
"""

from __future__ import annotations

import argparse
import gzip
import json
import unicodedata
from collections import Counter
from pathlib import Path

from lma_phase3.evaluate import file_hash, normalize_answer
from lma_phase3.generate import FAMILIES


def is_word_character(character: str) -> bool:
    """Letters, combining marks, numbers and connector punctuation continue a word."""

    category = unicodedata.category(character)
    return category[0] in ("L", "M", "N") or category == "Pc"


def contains_answer(text: str, answer: str) -> bool:
    """Match NFC/whitespace-normalized answer text bounded by non-word characters."""

    text, answer = normalize_answer(text), normalize_answer(answer)
    if not answer:
        raise ValueError("Reference answer must not be empty")
    offset = 0
    while (start := text.find(answer, offset)) != -1:
        end = start + len(answer)
        left_ok = start == 0 or not is_word_character(text[start - 1])
        right_ok = end == len(text) or not is_word_character(text[end])
        if left_ok and right_ok:
            return True
        offset = start + 1
    return False


def score_row(prediction: dict, source: dict, resource: dict) -> dict:
    """Credit answer presence and flag competing answers without interpreting prose."""

    reference = source["answer_text"]
    generated = prediction["generated"]
    if source["task_family"] == "equality":
        candidates = [resource["yes"], resource["no"]]
    else:
        problem = source["problem"]
        candidates = [problem["entities"][i] for i in problem["query"]]
    mentioned = [name for name in candidates if contains_answer(generated, name)]
    correct_present = contains_answer(generated, reference)
    others = [name for name in mentioned if normalize_answer(name) != normalize_answer(reference)]
    strict = normalize_answer(generated) == normalize_answer(reference)
    return {
        "id": source["id"],
        "task_family": source["task_family"],
        "reference": reference,
        "strict_exact_match": strict,
        "answer_present": correct_present,
        "mentioned_candidates": mentioned,
        "correct_and_alternative_present": correct_present and bool(others),
        "correct_present_without_alternative": correct_present and not others,
        "literal_substring_present": normalize_answer(reference) in normalize_answer(generated),
    }


def summarize(rows: list[dict]) -> dict:
    """Count each presence diagnostic; no candidates are required to end generation."""

    count = len(rows)
    if not count:
        raise ValueError("Cannot score an empty split")
    counts = Counter()
    for row in rows:
        for key in (
            "strict_exact_match",
            "answer_present",
            "correct_and_alternative_present",
            "correct_present_without_alternative",
            "literal_substring_present",
        ):
            counts[key] += int(row[key])
    return {
        "examples": count,
        **dict(counts),
        "answer_presence_rate": counts["answer_present"] / count,
        "strict_exact_match_rate": counts["strict_exact_match"] / count,
    }


def run(predictions_root: Path, dataset_root: Path, output: Path, root: Path) -> dict:
    """Verify saved output/data hashes, then score every original generation without rerunning."""

    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output directory; existing supplemental scores are preserved")
    output.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for lang in ("hindi", "bhojpuri"):
        summary = json.loads((predictions_root / lang / "summary.json").read_text())
        manifest_path = dataset_root / lang / "manifest.json"
        if file_hash(manifest_path) != summary["dataset_manifest_sha256"]:
            raise ValueError("Dataset manifest hash mismatch")
        resource_path = root / lang / "configs/phase3_templates.json"
        manifest = json.loads(manifest_path.read_text())
        if file_hash(resource_path) != manifest["templates_sha256"]:
            raise ValueError("Template resource hash mismatch")
        resource = json.loads(resource_path.read_text())
        results = {}
        for split in ("validation", "test", "challenge_names", "challenge_wording"):
            source_path = dataset_root / lang / f"{split}.jsonl.gz"
            prediction_path = predictions_root / lang / f"{split}_predictions.jsonl"
            if file_hash(source_path) != summary["splits"][split]["split_sha256"]:
                raise ValueError("Source split hash mismatch")
            if file_hash(prediction_path) != summary["splits"][split]["predictions_sha256"]:
                raise ValueError("Saved predictions hash mismatch")
            with gzip.open(source_path, "rt") as handle:
                sources = {r["id"]: r for r in map(json.loads, handle)}
            predictions = [json.loads(line) for line in prediction_path.read_text().splitlines()]
            if len(predictions) != len(sources) or {r["id"] for r in predictions} != set(sources):
                raise ValueError("Prediction coverage mismatch")
            rows = []
            for prediction in predictions:
                source = sources[prediction["id"]]
                if (
                    prediction["prompt"] != source["prompt"]
                    or prediction["reference"] != source["answer_text"]
                ):
                    raise ValueError("Prediction/source mismatch")
                score = score_row(prediction, source, resource)
                if score["strict_exact_match"] != prediction["correct"]:
                    raise ValueError("Strict baseline no longer reproduces")
                rows.append(score)
            path = output / f"{lang}_{split}_presence.jsonl"
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            result = {
                "overall": summarize(rows),
                "per_task": {
                    family: summarize([r for r in rows if r["task_family"] == family])
                    for family in FAMILIES
                },
                "original_predictions_sha256": file_hash(prediction_path),
                "row_scores_sha256": file_hash(path),
            }
            results[split] = result
            print(lang, split, json.dumps(result["overall"]), flush=True)
        all_results[lang] = results
    document = {
        "metric": "whole-answer presence anywhere in generated continuation",
        "status": "post-hoc supplemental diagnostic requested after observing strict baseline",
        "generation": "unchanged original greedy generations; EOS or 16-token limit",
        "normalization": "Unicode NFC and whitespace collapse; reject matches inside words",
        "limitations": [
            "Mentions may be copying, negation, or an incorrect claim.",
            "A response containing every candidate receives presence credit; flagged separately.",
            "This is not explanation correctness or final-answer accuracy.",
        ],
        "scorer_sha256": file_hash(Path(__file__)),
        "languages": all_results,
    }
    (output / "summary.json").write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    return document


def main() -> None:
    """CLI for supplemental scoring of both saved language baselines."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.predictions_root,
        args.dataset_root,
        args.output_dir,
        Path(__file__).resolve().parents[2],
    )


if __name__ == "__main__":
    main()
