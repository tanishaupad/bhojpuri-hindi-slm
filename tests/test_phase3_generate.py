import json
from collections import Counter
from pathlib import Path

import sentencepiece as spm

from lma_phase3.generate import generate_pilot
from lma_phase3.problems import Problem, solve, verify_two_hop

ROOT = Path(__file__).resolve().parents[1]


def test_pilots_have_verified_labels_lengths_balance_and_independence():
    fingerprints = []
    for language in ("hindi", "bhojpuri"):
        resource = json.loads((ROOT / language / "configs/phase3_templates.json").read_text())
        tokenizer = spm.SentencePieceProcessor(
            model_file=str(ROOT / language / "tokenizer" / f"{language}.model")
        )
        rows, _rejected = generate_pilot(resource, tokenizer)
        assert rows == generate_pilot(resource, tokenizer)[0]
        assert len(rows) == 50
        assert Counter(row["task_family"] for row in rows) == {
            "direct": 13,
            "extremum": 13,
            "equality": 12,
            "transitivity": 12,
        }
        assert len({row["prompt"] for row in rows}) == 50
        assert len({row["problem_group_id"] for row in rows}) == 50
        assert all(row["split"] == "development_review" for row in rows)
        assert all(row["serialized_tokens_including_eos"] <= 256 for row in rows)
        for row in rows:
            problem = Problem(**row["problem"])
            label = solve(problem)
            if problem.family == "transitivity":
                verify_two_hop(problem)
            if problem.family == "equality":
                assert row["answer_label"] == label
            else:
                assert row["answer_text"] == problem.entities[label]
        for family in ("direct", "extremum", "transitivity"):
            group = [row for row in rows if row["task_family"] == family]
            for factor in ("answer_position", "direction", "domain"):
                counts = Counter(row[factor] for row in group).values()
                assert max(counts) - min(counts) <= 1
        assert sum(row["answer_label"] is True for row in rows) == 6
        assert sum(row["answer_label"] is False for row in rows) == 6
        fingerprints.append({row["problem_group_id"] for row in rows})
    assert not fingerprints[0] & fingerprints[1]
