"""Generate development-only reasoning examples for human language review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import sentencepiece as spm

from lma_phase1.tokenizer.common import normalize_tokenizer_input
from lma_phase3.problems import Problem, solve, verify_two_hop, world_fingerprint

FAMILIES = ("direct", "extremum", "equality", "transitivity")
GENERATOR_VERSION = "phase3-development-v1"


def balanced(rng: random.Random, choices: list, count: int) -> list:
    """Shuffle an exactly or almost equally apportioned experimental factor."""

    values = [choices[i % len(choices)] for i in range(count)]
    rng.shuffle(values)
    return values


def render(problem: Problem, resource: dict, variant: int) -> tuple[str, str, list[str]]:
    """Render only exposed premises, with a language-specific answer cue."""

    domain = resource["domains"][problem.domain]
    if problem.edges:
        premises = [
            domain["relation"].format(left=problem.entities[a], right=problem.entities[b])
            for a, b in problem.edges
        ]
    else:
        premises = [
            domain["fact"].format(name=name, value=value)
            for name, value in zip(problem.entities, problem.values, strict=True)
        ]
    query_entities = [problem.entities[i] for i in problem.query]
    choices = (
        resource["join_two"].join(query_entities)
        if len(query_entities) == 2
        else ", ".join(query_entities[:-1]) + resource["join_two"] + query_entities[-1]
    )
    key = "equality" if problem.family == "equality" else problem.direction
    question = domain["questions"][key][variant].format(
        choices=choices, left=problem.entities[0], right=problem.entities[1]
    )
    label = solve(problem)
    answer = (
        resource["yes" if label else "no"]
        if problem.family == "equality"
        else problem.entities[label]
    )
    prompt = " ".join([*premises, question, resource["answer_cue"]])
    return prompt, answer, premises


def generate_pilot(
    resource: dict,
    tokenizer: spm.SentencePieceProcessor,
    count: int = 50,
    *,
    excluded_worlds: set[str] | None = None,
    excluded_prompts: set[str] | None = None,
    wording_variants: tuple[int, ...] = (0, 1),
):
    """Create balanced development problems; never create train/test partitions here."""

    rng = random.Random(resource["seed"])
    rows = []
    worlds: set[str] = set(excluded_worlds or ())
    prompts: set[str] = set(excluded_prompts or ())
    rejected = Counter()
    for family_index, family in enumerate(FAMILIES):
        amount = count // 4 + int(family_index < count % 4)
        size = 3 if family in ("extremum", "transitivity") else 2
        domains = balanced(rng, list(resource["domains"]), amount)
        directions = balanced(rng, ["greater", "smaller"], amount)
        positions = balanced(rng, list(range(2 if family == "transitivity" else size)), amount)
        equalities = balanced(rng, [True, False], amount)
        variants = balanced(rng, list(wording_variants), amount)
        orders = balanced(rng, [False, True], amount)
        for index in range(amount):
            for attempt in range(1000):
                names = tuple(rng.sample(resource["names"], size))
                domain = domains[index]
                low, high = resource["domains"][domain]["range"]
                values = rng.sample(range(low, high + 1), size)
                direction = directions[index]
                position = positions[index]
                if family == "transitivity":
                    # Randomize which mentioned entity is the middle of the chain.
                    chain = rng.sample(range(3), 3)
                    edges = [(chain[0], chain[1]), (chain[1], chain[2])]
                    if orders[index]:
                        edges.reverse()
                    correct = chain[0] if direction == "greater" else chain[2]
                    other = chain[2] if direction == "greater" else chain[0]
                    query = (correct, other) if position == 0 else (other, correct)
                    problem = Problem(
                        family, domain, names, edges=tuple(edges), query=query, direction=direction
                    )
                    verify_two_hop(problem)
                else:
                    if family == "equality" and equalities[index]:
                        values[1] = values[0]
                    elif family != "equality":
                        best = values.index((max if direction == "greater" else min)(values))
                        values[position], values[best] = values[best], values[position]
                    elif not equalities[index]:
                        # Balance the direction of unequal values within the negative class.
                        negative_index = sum(not value for value in equalities[:index])
                        values.sort(reverse=bool(negative_index % 2))
                    problem = Problem(
                        family,
                        domain,
                        names,
                        tuple(values),
                        query=tuple(range(size)),
                        direction=direction,
                    )
                prompt, answer, premises = render(problem, resource, variants[index])
                fingerprint = world_fingerprint(problem)
                normalized_prompt = normalize_tokenizer_input(prompt)
                if fingerprint in worlds or normalized_prompt in prompts:
                    rejected["duplicate"] += 1
                    continue
                # Encode the full serialized training text, not separately summed segments.
                full_text = normalize_tokenizer_input(prompt + " " + answer)
                token_ids = tokenizer.encode(full_text, out_type=int) + [tokenizer.eos_id()]
                if len(token_ids) > 256:
                    rejected["over_context"] += 1
                    continue
                if tokenizer.unk_id() in token_ids:
                    rejected["unknown_token"] += 1
                    continue
                label = solve(problem)
                row = {
                    "id": f"{resource['language']}-dev-{len(rows) + 1:03d}",
                    "problem_group_id": fingerprint,
                    "language": resource["language"],
                    "split": "development_review",
                    "task_family": family,
                    "domain": domain,
                    "problem": asdict(problem),
                    "premises": premises,
                    "prompt": prompt,
                    "answer_text": answer,
                    "answer_entity_id": None if family == "equality" else label,
                    "answer_label": bool(label) if family == "equality" else None,
                    "answer_position": None if family == "equality" else problem.query.index(label),
                    "required_hops": 2 if family == "transitivity" else 1,
                    "template_id": f"{resource['language']}.{family}.{domain}.{variants[index]}",
                    "direction": None if family == "equality" else direction,
                    "serialized_tokens_including_eos": len(token_ids),
                    "seed": resource["seed"],
                    "generator_version": GENERATOR_VERSION,
                    "review_status": "unreviewed",
                }
                rows.append(row)
                worlds.add(fingerprint)
                prompts.add(normalized_prompt)
                break
            else:
                raise ValueError(f"Could not generate a valid {family} example after 1000 attempts")
    return rows, dict(rejected)


def sha256(path: Path) -> str:
    """Hash an input or output artifact for the review manifest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_pilot(language: str, root: Path, output: Path) -> list[dict]:
    """Write reproducible records, a review worksheet, and a compact audit manifest."""

    resource_path = root / language / "configs/phase3_templates.json"
    tokenizer_path = root / language / "tokenizer" / f"{language}.model"
    resource = json.loads(resource_path.read_text())
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    rows, rejected = generate_pilot(resource, tokenizer)
    records = output / f"{language}_development.jsonl"
    records.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    worksheet = output / f"{language}_review.csv"
    review_fields = [
        "id",
        "task_family",
        "domain",
        "prompt",
        "answer_text",
        "language_ok",
        "answer_ok",
        "suggested_wording",
        "reviewer",
        "notes",
    ]
    with worksheet.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in review_fields})
    lengths = sorted(row["serialized_tokens_including_eos"] for row in rows)
    summary = {}
    for family in FAMILIES:
        group = [row for row in rows if row["task_family"] == family]
        summary[family] = {
            "examples": len(group),
            "domains": dict(Counter(row["domain"] for row in group)),
            "directions": dict(Counter(str(row["direction"]) for row in group)),
            "answer_positions": dict(Counter(str(row["answer_position"]) for row in group)),
            "equality_labels": dict(Counter(str(row["answer_label"]) for row in group)),
            "templates": dict(Counter(row["template_id"] for row in group)),
        }
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "language": resource["language"],
        "seed": resource["seed"],
        "split": "development_review",
        "examples": len(rows),
        "human_review_status": "pending",
        "release_status": "pilot_only",
        "tokenizer_sha256": sha256(tokenizer_path),
        "templates_sha256": sha256(resource_path),
        "records_sha256": sha256(records),
        "family_summary": summary,
        "length_including_eos": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
        },
        "rejections": rejected,
        "unique_worlds": len({r["problem_group_id"] for r in rows}),
        "worlds_excluded_from_future_heldout": [r["problem_group_id"] for r in rows],
        "normalization": "Collapse Unicode whitespace before encoding complete prompt + answer; append EOS",
        "limitations": [
            "Development examples only; no train/validation/test release",
            "Two domains; two question variants per domain/family",
            "Male names only to limit grammatical agreement complexity in this pilot",
            "Language naturalness requires human review; symbolic validity is insufficient",
        ],
        "code_sha256": {p.name: sha256(p) for p in (root / "src/lma_phase3").glob("*.py")},
    }
    (output / f"{language}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return rows


def main() -> None:
    """Create independent language pilots; refuse to overwrite prior output."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--languages", nargs="+", default=["hindi", "nepali"])
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty; choose a new directory to preserve reviews")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = {
        language: write_pilot(language, args.repo_root, args.output_dir)
        for language in args.languages
    }
    fingerprints = [
        {row["problem_group_id"] for row in rows} for rows in all_rows.values()
    ]
    for index, values in enumerate(fingerprints):
        if any(values & other for other in fingerprints[index + 1 :]):
            raise ValueError("Shared problem instances across languages")
    lines = [
        "# Phase 3 development examples for language review",
        "",
        "These 100 examples are development material, excluded from future held-out sets.",
        "Answers are solver-verified; wording has not been human reviewed. Correct or flag",
        "unnatural wording and ambiguous questions in the companion CSV worksheets.",
        "Fill language_ok and answer_ok with yes/no, plus suggested_wording and reviewer.",
        "",
    ]
    for language, rows in all_rows.items():
        lines += [f"## {language.title()}", ""]
        for row in rows:
            lines += [
                f"### {row['id']} — {row['task_family']} / {row['domain']}",
                "",
                row["prompt"],
                "",
                f"**Expected answer:** {row['answer_text']}",
                "",
            ]
    (args.output_dir / "review.md").write_text("\n".join(lines))
    print(f"Created 50 development examples per language in {args.output_dir}")


if __name__ == "__main__":
    main()
