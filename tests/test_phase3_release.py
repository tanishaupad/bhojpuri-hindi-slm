import gzip
import json
import shutil
from pathlib import Path

import pytest
import sentencepiece as spm

from lma_phase3.generate import generate_pilot
from lma_phase3.problems import Problem
from lma_phase3.release import audit_rows, build_release, independent_answer, write_gzip

ROOT = Path(__file__).resolve().parents[1]


def test_independent_ranking_verifier_rejects_underdetermined_facts():
    problem = Problem("transitivity", "age", ("A", "B", "C"), edges=((0, 1), (0, 2)), query=(1, 2))
    with pytest.raises(ValueError, match="unique answer"):
        independent_answer(problem)


def test_auditor_detects_corrupt_label_and_prompt():
    resource = json.loads((ROOT / "hindi/configs/phase3_templates.json").read_text())
    tokenizer = spm.SentencePieceProcessor(model_file=str(ROOT / "hindi/tokenizer/hindi.model"))
    rows, _ = generate_pilot(resource, tokenizer)
    answer = rows[0]["answer_text"]
    rows[0]["answer_text"] = "incorrect"
    with pytest.raises(ValueError, match="Incorrect answer"):
        audit_rows(rows, resource, tokenizer, 50)
    rows[0]["answer_text"] = answer
    rows[0]["prompt"] = "The facts have changed"
    with pytest.raises(ValueError, match="Serialized prompt"):
        audit_rows(rows, resource, tokenizer, 50)


def test_small_release_has_isolated_splits_and_deterministic_bytes(tmp_path):
    root = tmp_path / "repo"
    for lang in ("hindi", "bhojpuri"):
        shutil.copytree(ROOT / lang / "configs", root / lang / "configs")
        shutil.copytree(ROOT / lang / "tokenizer", root / lang / "tokenizer")
        p = root / lang / "configs/phase3_release.json"
        config = json.loads(p.read_text())
        config["split_sizes"] = {key: 40 for key in config["split_sizes"]}
        p.write_text(json.dumps(config))
    shutil.copytree(
        ROOT / "report/samples/generated/phase3/pilot_v1",
        root / "report/samples/generated/phase3/pilot_v1",
    )
    shutil.copytree(ROOT / "src/lma_phase3", root / "src/lma_phase3")
    first = tmp_path / "first"
    second = tmp_path / "second"
    audit = build_release(root, first, tmp_path / "evidence1")
    assert audit["examples"] == 400
    assert audit["unique_release_worlds"] == 400
    build_release(root, second, tmp_path / "evidence2")
    for p in first.rglob("*.jsonl.gz"):
        assert p.read_bytes() == (second / p.relative_to(first)).read_bytes()
    with pytest.raises(ValueError, match="never overwritten"):
        build_release(root, first, tmp_path / "evidence3")
    cfg = json.loads((root / "hindi/configs/phase3_release.json").read_text())
    for split in cfg["split_sizes"]:
        with gzip.open(first / "hindi" / f"{split}.jsonl.gz", "rt") as f:
            rows = [json.loads(line) for line in f]
        names = {name for row in rows for name in row["problem"]["entities"]}
        expected = cfg["challenge_names" if split == "challenge_names" else "core_names"]
        assert names <= set(expected)
        assert {int(r["template_id"].rsplit(".", 1)[1]) for r in rows} == {
            1 if split == "challenge_wording" else 0
        }


def test_gzip_does_not_include_output_filename_or_timestamp(tmp_path):
    rows = [{"x": "हिन्दी"}]
    write_gzip(tmp_path / "one.gz", rows)
    write_gzip(tmp_path / "two.gz", rows)
    assert (tmp_path / "one.gz").read_bytes() == (tmp_path / "two.gz").read_bytes()
