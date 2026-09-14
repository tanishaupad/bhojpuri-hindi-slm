"""Build and audit separate reproducible reasoning releases."""

from __future__ import annotations

import argparse
import copy
import gzip
import itertools
import json
from collections import Counter
from pathlib import Path

import sentencepiece as spm

from lma_phase1.tokenizer.common import normalize_tokenizer_input
from lma_phase3.generate import FAMILIES, generate_pilot, render, sha256
from lma_phase3.problems import Problem, world_fingerprint

VERSION = "reasoning-v1"


def independent_answer(problem: Problem) -> str | bool:
    """Verify graph answers by enumerating all rankings, independently of closure."""

    if problem.family == "equality":
        return problem.values[0] == problem.values[1]
    if problem.family == "transitivity":
        answers = set()
        for ranking in itertools.permutations(range(len(problem.entities))):
            if all(ranking[a] > ranking[b] for a, b in problem.edges):
                endpoint = (max if problem.direction == "greater" else min)(
                    problem.query, key=lambda i: ranking[i]
                )
                answers.add(problem.entities[endpoint])
        if len(answers) != 1:
            raise ValueError("No unique answer across all rankings consistent with the premises")
        return answers.pop()
    ranked = sorted(zip(problem.values, problem.entities, strict=True))
    return ranked[-1 if problem.direction == "greater" else 0][1]


def audit_rows(rows: list[dict], resource: dict, tokenizer, expected_count: int) -> dict:
    """Verify serialized labels, quotas, token counts, and within-split uniqueness."""

    if len(rows) != expected_count:
        raise ValueError("Wrong split size")
    family_counts = Counter(r["task_family"] for r in rows)
    if (
        set(family_counts) != set(FAMILIES)
        or max(family_counts.values()) - min(family_counts.values()) > 1
    ):
        raise ValueError("Unbalanced task families")
    for row in rows:
        problem = Problem(**row["problem"])
        answer = independent_answer(problem)
        expected = resource["yes" if answer else "no"] if isinstance(answer, bool) else answer
        if row["answer_text"] != expected:
            raise ValueError("Incorrect answer")
        variant = int(row["template_id"].rsplit(".", 1)[1])
        prompt, rendered_answer, premises = render(problem, resource, variant)
        if (prompt, rendered_answer, premises) != (
            row["prompt"],
            row["answer_text"],
            row["premises"],
        ):
            raise ValueError("Serialized prompt differs from the symbolic problem")
        ids = tokenizer.encode(normalize_tokenizer_input(prompt + " " + expected)) + [
            tokenizer.eos_id()
        ]
        if len(ids) > 256 or tokenizer.unk_id() in ids:
            raise ValueError("Invalid tokenized example")
        if len(ids) != row["serialized_tokens_including_eos"]:
            raise ValueError("Incorrect recorded token length")
        if world_fingerprint(problem) != row["problem_group_id"]:
            raise ValueError("Incorrect world fingerprint")
    for key in ("id", "problem_group_id", "prompt"):
        if len({row[key] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate {key}")
    families = {}
    for family in FAMILIES:
        group = [row for row in rows if row["task_family"] == family]
        factors = (
            ("answer_label", "domain")
            if family == "equality"
            else ("answer_position", "direction", "domain")
        )
        for factor in factors:
            counts = Counter(row[factor] for row in group)
            if max(counts.values()) - min(counts.values()) > 1:
                raise ValueError(f"Unbalanced {family} {factor}")
        families[family] = {
            "examples": len(group),
            **{factor: dict(Counter(str(row[factor]) for row in group)) for factor in factors},
        }
    lengths = [row["serialized_tokens_including_eos"] for row in rows]
    return {
        "examples": len(rows),
        "families": families,
        "tokens_including_eos": sum(lengths),
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "mean_tokens": sum(lengths) / len(lengths),
        "template_counts": dict(Counter(row["template_id"] for row in rows)),
        "independent_answer_check": "passed: numerical sort / exhaustive ranking enumeration",
    }


def write_gzip(path: Path, rows: list[dict]) -> None:
    """Write deterministic compressed JSONL with no timestamp or host filename."""

    with (
        path.open("wb") as output,
        gzip.GzipFile(fileobj=output, mode="wb", mtime=0, filename="") as handle,
    ):
        for row in rows:
            handle.write((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode())


def build_release(
    root: Path,
    output: Path,
    evidence: Path,
    languages: tuple[str, ...] = ("hindi", "bhojpuri"),
    pilot_dir: Path | None = None,
) -> dict:
    """Allocate worlds to one split, reserving names/wording and excluding pilot worlds."""

    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output directory; releases are never overwritten")
    configs = {
        lang: json.loads((root / lang / "configs/phase3_release.json").read_text())
        for lang in languages
    }
    names_by_language = [
        set(cfg["core_names"] + cfg["challenge_names"]) for cfg in configs.values()
    ]
    for index, names in enumerate(names_by_language):
        if any(names & other for other in names_by_language[index + 1 :]):
            raise ValueError("Language name pools must be independent")
    pilot = pilot_dir or root / "report/samples/generated/phase3/pilot_v1"
    excluded_worlds = set()
    excluded_prompts = set()
    for lang in configs:
        pilot_records = pilot / f"{lang}_development.jsonl"
        if pilot_records.exists():
            for line in pilot_records.read_text().splitlines():
                row = json.loads(line)
                excluded_worlds.add(row["problem_group_id"])
                excluded_prompts.add(normalize_tokenizer_input(row["prompt"]))
    pilot_worlds = set(excluded_worlds)
    output.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    all_summaries = {}
    for lang, config in configs.items():
        if set(config["core_names"]) & set(config["challenge_names"]):
            raise ValueError("Challenge names overlap with core names")
        if set(config["core_variants"]) & set(config["challenge_variants"]):
            raise ValueError("Challenge wording overlaps with core wording")
        base = json.loads((root / lang / "configs/phase3_templates.json").read_text())
        tokenizer_path = root / lang / "tokenizer" / f"{lang}.model"
        tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        directory = output / lang
        directory.mkdir()
        summaries = {}
        for index, (split, count) in enumerate(config["split_sizes"].items()):
            resource = copy.deepcopy(base)
            resource["names"] = config[
                "challenge_names" if split == "challenge_names" else "core_names"
            ]
            resource["seed"] = config["seed"] + index * 100003
            variants = config[
                "challenge_variants" if split == "challenge_wording" else "core_variants"
            ]
            # The split's seed, quotas, names and wording are fixed before any candidate is rendered.
            rows, rejected = generate_pilot(
                resource,
                tokenizer,
                count,
                excluded_worlds=excluded_worlds,
                excluded_prompts=excluded_prompts,
                wording_variants=tuple(variants),
            )
            for number, row in enumerate(rows):
                row.update(
                    id=f"{resource['language']}-{split}-{number:06d}",
                    split=split,
                    generator_version=VERSION,
                    review_status="approved_template_wording",
                )
            summary = audit_rows(rows, resource, tokenizer, count)
            worlds = {r["problem_group_id"] for r in rows}
            prompts = {normalize_tokenizer_input(r["prompt"]) for r in rows}
            if worlds & excluded_worlds or prompts & excluded_prompts:
                raise ValueError("Cross-split/language or pilot leakage")
            excluded_worlds.update(worlds)
            excluded_prompts.update(prompts)
            path = directory / f"{split}.jsonl.gz"
            write_gzip(path, rows)
            summary.update(
                sha256=sha256(path),
                size_bytes=path.stat().st_size,
                rejected=rejected,
                seed=resource["seed"],
                file=path.name,
                wording_variants=variants,
            )
            summaries[split] = summary
            print(
                lang,
                split,
                count,
                "verified examples, max tokens",
                summary["max_tokens"],
                flush=True,
            )
        manifest = {
            "version": VERSION,
            "language": base["language"],
            "split_summaries": summaries,
            "tokenizer_sha256": sha256(tokenizer_path),
            "templates_sha256": sha256(root / lang / "configs/phase3_templates.json"),
            "release_config_sha256": sha256(root / lang / "configs/phase3_release.json"),
            "review_record_sha256": (
                sha256(pilot / "wording_review.json")
                if (pilot / "wording_review.json").exists()
                else None
            ),
            "pilot_records_sha256": {
                name: sha256(pilot / f"{name}_development.jsonl")
                for name in configs
                if (pilot / f"{name}_development.jsonl").exists()
            },
            "excluded_pilot_worlds": len(pilot_worlds),
            "code_sha256": {
                p.name: sha256(p) for p in sorted((root / "src/lma_phase3").glob("*.py"))
            },
            "leakage_checks": {
                "duplicate_named_worlds": 0,
                "duplicate_normalized_prompts": 0,
                "pilot_world_overlap": 0,
                "core_challenge_name_overlap": 0,
                "core_challenge_wording_overlap": 0,
            },
            "limitations": [
                "Primary splits share task structures by design; this is not unseen-structure evaluation.",
                "Two reviewed wording variants per domain; only variant 0 in core splits.",
                "Ages/heights and masculine names only.",
                "Unknown-token rejection is applied before acceptance and reported in rejections.",
                "Wording approved collectively by user; not every release example was manually inspected.",
            ],
            "serialization": "Whitespace-normalized prompt + space + answer, append EOS; context limit 256",
            "scoring_protocol": {
                "decoding": "greedy",
                "maximum_new_tokens": 16,
                "stop": "EOS or token limit",
                "exact_match": "Unicode NFC, collapse whitespace and strip surrounding whitespace; compare full continuation to canonical answer. No substring matching or answer extraction.",
                "reporting": "Per-task and overall primary test accuracy; challenge scores separately; validation only for selection.",
            },
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
        (evidence / f"{lang}_manifest.json").write_bytes((directory / "manifest.json").read_bytes())
        all_summaries[lang] = manifest
    audit = {
        "version": VERSION,
        "languages": list(configs),
        "examples": sum(
            s["examples"] for m in all_summaries.values() for s in m["split_summaries"].values()
        ),
        "unique_release_worlds": len(excluded_worlds) - len(pilot_worlds),
        "cross_language_world_overlap": 0,
        "cross_split_world_overlap": 0,
        "pilot_world_overlap": 0,
        "all_answer_and_length_checks": "passed",
    }
    (evidence / "release_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main() -> None:
    """Build full datasets under an untracked directory, with compact evidence in report/."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--languages", nargs="+", default=["hindi", "nepali"])
    parser.add_argument("--pilot-dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build_release(
                args.repo_root,
                args.output_dir,
                args.evidence_dir,
                tuple(args.languages),
                args.pilot_dir,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
