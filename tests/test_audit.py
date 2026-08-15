from __future__ import annotations

import argparse
import csv

from lma_phase1.audit import create_sample, sample_score
from lma_phase1.io import write_jsonl
from lma_phase1.schema import Document


def test_sample_score_is_deterministic_and_seeded():
    assert sample_score("seed", "doc") == sample_score("seed", "doc")
    assert sample_score("seed", "doc") != sample_score("other", "doc")


def test_audit_sample_can_filter_one_source(tmp_path):
    input_path = tmp_path / "input.jsonl.gz"
    output_path = tmp_path / "audit.csv"
    documents = [
        Document(
            document_id=f"doc-{index}",
            language="bho",
            text="ई परीक्षण खातिर भोजपुरी वाक्य हवे।",
            source_name=source_name,
            collection_type="manual",
            collection_method="test",
            license="test",
            collected_at="2026-08-15",
        )
        for index, source_name in enumerate(("wanted", "other"), start=1)
    ]
    write_jsonl(input_path, documents)
    args = argparse.Namespace(
        input=str(input_path),
        output=str(output_path),
        source_name=["wanted"],
        size=100,
        per_source=0,
        sample_characters=500,
        seed="seed",
    )

    assert create_sample(args) == 1
    with output_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["source_name"] == "wanted"
