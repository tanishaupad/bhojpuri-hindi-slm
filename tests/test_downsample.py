from argparse import Namespace

from lma_phase1.downsample import downloaded_word_budget, run, selection_score
from lma_phase1.io import iter_jsonl, write_jsonl
from lma_phase1.schema import Document


def test_downloaded_word_budget_meets_target_fraction() -> None:
    assert downloaded_word_budget(200, 0.25) == 600


def test_selection_score_is_deterministic_and_seeded() -> None:
    assert selection_score("seed", "doc") == selection_score("seed", "doc")
    assert selection_score("seed", "doc") != selection_score("other", "doc")


def test_run_preserves_manual_and_hits_downloaded_budget(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl.gz"
    output_path = tmp_path / "output.jsonl.gz"
    report_path = tmp_path / "report.json"
    documents = []
    for collection_type, count in (("manual", 2), ("downloaded", 10)):
        for index in range(count):
            documents.append(
                Document(
                    document_id=f"{collection_type}_{index}",
                    language="bho",
                    text=" ".join(["शब्द"] * 100),
                    source_name="test",
                    collection_type=collection_type,
                    collection_method="unit_test",
                    license="CC0-1.0",
                    collected_at="2026-08-16",
                )
            )
    write_jsonl(input_path, documents)

    report = run(
        Namespace(
            input=str(input_path),
            output=str(output_path),
            report=str(report_path),
            target_manual_word_fraction=0.25,
            seed="test-seed",
            write_batch_size=3,
        )
    )

    kept = list(iter_jsonl(output_path))
    assert sum(document.collection_type == "manual" for document in kept) == 2
    assert sum(document.collection_type == "downloaded" for document in kept) == 6
    assert report["actual_manual_word_fraction"] == 0.25
