from __future__ import annotations

import argparse
import csv

from lma_phase1.io import iter_jsonl, write_jsonl
from lma_phase1.minhash_deduplicate import bottom_k_sketch
from lma_phase1.schema import Document
from lma_phase1.text import canonical_text
from lma_phase2.decontaminate import bottom_k_from_canonical, run


def make_document(document_id: str, text: str) -> Document:
    return Document(
        document_id=document_id,
        language="bho",
        text=text,
        source_name="test",
        collection_type="manual",
        collection_method="test",
        license="test",
        collected_at="2026-08-22",
        source_url=f"https://example.test/{document_id}",
    )


def arguments(tmp_path, train, validation, test, clean_output=None):
    return argparse.Namespace(
        train=str(train),
        validation=str(validation),
        test=str(test),
        database=str(tmp_path / "audit.sqlite"),
        audit_output=str(tmp_path / "matches.csv"),
        report=str(tmp_path / "report.json"),
        clean_train_output=str(clean_output) if clean_output else None,
        check=None,
        similarity_threshold=0.95,
        sketch_size=64,
        index_size=16,
        minimum_shared_index_hashes=2,
        shingle_size=5,
        minimum_shingles=20,
        maximum_shingles=2048,
        maximum_candidates=200,
        minimum_paragraph_characters=50,
        minimum_paragraph_words=10,
        maximum_train_documents=None,
        commit_interval=2,
        workers=1,
        feature_batch_size=2,
    )


def test_optimized_sketch_matches_phase_1_definition() -> None:
    text = "ई एगो परीक्षण वाक्य हवे जेकर शब्दन से एकदम समान मिनहैश बनल चाहीं। " * 20

    assert bottom_k_from_canonical(canonical_text(text), 64, 5, 2048) == bottom_k_sketch(
        text, 64, 5, 2048
    )


def test_audit_detects_exact_near_and_paragraph_leakage_and_writes_clean_split(
    tmp_path,
) -> None:
    validation_text = " ".join(f"सुरक्षितशब्द{index}" for index in range(80))
    test_text = " ".join(f"परीक्षणशब्द{index}" for index in range(100))
    shared_paragraph = " ".join(f"साझाशब्द{index}" for index in range(30))
    validation = tmp_path / "validation.jsonl.gz"
    test = tmp_path / "test.jsonl.gz"
    train = tmp_path / "train.jsonl.gz"
    clean = tmp_path / "train.clean.jsonl.gz"
    write_jsonl(
        validation,
        [
            make_document("protected-exact", validation_text),
            make_document("protected-paragraph", shared_paragraph),
        ],
    )
    write_jsonl(test, [make_document("protected-near", test_text)])
    write_jsonl(
        train,
        [
            make_document("train-exact", validation_text),
            make_document("train-near", test_text + " अतिरिक्त एक दो तीन"),
            make_document(
                "train-paragraph",
                shared_paragraph + "\n" + " ".join(f"असंबंधितशब्द{index}" for index in range(120)),
            ),
            make_document("train-clear", " ".join(f"स्वच्छप्रशिक्षणशब्द{index}" for index in range(80))),
        ],
    )

    args = arguments(tmp_path, train, validation, test)
    report = run(args)

    assert report["status"] == "complete"
    assert report["counts_cumulative"]["matches_exact_document"] == 1
    assert report["counts_cumulative"]["matches_near_document"] == 1
    assert report["counts_cumulative"]["matches_exact_paragraph"] == 1
    assert report["counts_cumulative"]["train_documents_flagged"] == 3
    with (tmp_path / "matches.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["match_type"] for row in rows} == {
        "exact_document",
        "near_document",
        "exact_paragraph",
    }
    assert {row["protected_split"] for row in rows} == {"validation", "test"}

    for row in rows:
        row["review_label"] = "remove"
        row["reviewer"] = "test-reviewer"
    with (tmp_path / "matches.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    args.clean_train_output = str(clean)
    cleaned_report = run(args)

    assert cleaned_report["clean_train_documents_written"] == 1
    assert [document.document_id for document in iter_jsonl(clean)] == ["train-clear"]
    with (tmp_path / "matches.csv").open(encoding="utf-8", newline="") as handle:
        preserved_rows = list(csv.DictReader(handle))
    assert {row["review_label"] for row in preserved_rows} == {"remove"}
    assert {row["reviewer"] for row in preserved_rows} == {"test-reviewer"}


def test_partial_audit_resumes_without_duplicate_matches(tmp_path) -> None:
    protected_text = " ".join(f"रक्षितशब्द{index}" for index in range(80))
    validation = tmp_path / "validation.jsonl.gz"
    test = tmp_path / "test.jsonl.gz"
    train = tmp_path / "train.jsonl.gz"
    write_jsonl(validation, [make_document("protected", protected_text)])
    write_jsonl(test, [make_document("test", " ".join(f"जाँचशब्द{i}" for i in range(80)))])
    write_jsonl(
        train,
        [
            make_document("duplicate", protected_text),
            make_document("clear", " ".join(f"अलगशब्द{i}" for i in range(80))),
        ],
    )
    args = arguments(tmp_path, train, validation, test)
    args.maximum_train_documents = 1

    partial = run(args)
    assert partial["status"] == "partial"
    assert partial["counts_cumulative"]["train_documents_processed"] == 1

    args.maximum_train_documents = None
    complete = run(args)
    assert complete["status"] == "complete"
    assert complete["counts_cumulative"]["train_documents_processed"] == 2
    assert complete["counts_cumulative"]["matches_exact_document"] == 1
