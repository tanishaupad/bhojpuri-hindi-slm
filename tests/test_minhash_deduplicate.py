from __future__ import annotations

import argparse

from lma_phase1.io import iter_jsonl, write_jsonl
from lma_phase1.minhash_deduplicate import (
    MinHashIndex,
    bottom_k_sketch,
    estimate_jaccard,
    run,
)
from lma_phase1.schema import Document


def make_document(document_id: str, text: str, collection_type: str = "manual") -> Document:
    return Document(
        document_id=document_id,
        language="bho",
        text=text,
        source_name="test",
        collection_type=collection_type,
        collection_method="test",
        license="test",
        collected_at="2026-08-16",
    )


def test_bottom_k_sketch_is_deterministic() -> None:
    text = "ई एगो लमहर भोजपुरी वाक्य हवे जेकर इस्तेमाल मिनहैश जाँच खातिर कइल जात बा। " * 5
    assert bottom_k_sketch(text) == bottom_k_sketch(text)


def test_estimated_similarity_separates_edit_from_unrelated_text() -> None:
    first = "आजु गाँव के बजार में बहुत भीड़ रहल आ सभे लोग सामान खरीदे आइल रहे। " * 5
    edited = first + "साँझ होते सभे अपना घरे लवट गइल।"
    unrelated = "कंप्यूटर विज्ञान में एल्गोरिदम आ डेटा संरचना के पढ़ाई कइल जाला। " * 5
    first_sketch = bottom_k_sketch(first)

    assert estimate_jaccard(first_sketch, bottom_k_sketch(first), 64) == 1.0
    assert estimate_jaccard(first_sketch, bottom_k_sketch(edited), 64) > estimate_jaccard(
        first_sketch, bottom_k_sketch(unrelated), 64
    )


def test_index_finds_near_duplicate_candidate(tmp_path) -> None:
    first_text = " ".join(f"भोजपुरीशब्द{index}" for index in range(100))
    second_text = first_text + " " + " ".join(
        f"नयाशब्द{index}" for index in range(5)
    )
    first = make_document("first", first_text)
    index = MinHashIndex(tmp_path / "minhash.sqlite", index_size=16)
    first_sketch = bottom_k_sketch(first.text)
    index.add(first, first_sketch)

    matched, similarity = index.best_match(
        bottom_k_sketch(second_text),
        sketch_size=64,
        minimum_shared_index_hashes=2,
        maximum_candidates=20,
    )
    index.close()

    assert matched == "first"
    assert similarity >= 0.85


def test_manual_only_index_does_not_compare_downloaded_documents_to_each_other(
    tmp_path,
) -> None:
    text = " ".join(f"हिन्दीशब्द{index}" for index in range(100))
    input_path = tmp_path / "input.jsonl.gz"
    output_path = tmp_path / "output.jsonl.gz"
    write_jsonl(
        input_path,
        [
            make_document("downloaded-1", text, "downloaded"),
            make_document("downloaded-2", text + " अतिरिक्त", "downloaded"),
        ],
    )
    args = argparse.Namespace(
        input=[str(input_path)],
        mode="apply",
        output=str(output_path),
        audit_output=None,
        database=str(tmp_path / "minhash.sqlite"),
        similarity_threshold=0.80,
        sketch_size=64,
        index_size=16,
        minimum_shared_index_hashes=2,
        shingle_size=5,
        minimum_shingles=20,
        maximum_shingles=2048,
        maximum_candidates=200,
        maximum_documents=None,
        write_batch_size=100,
        commit_interval=100,
        index_collection_type=["manual"],
    )

    counters = run(args)

    assert counters["documents_kept"] == 2
    assert counters["documents_kept_not_indexed"] == 2
    assert counters["near_duplicates_removed"] == 0
    assert len(list(iter_jsonl(output_path))) == 2
