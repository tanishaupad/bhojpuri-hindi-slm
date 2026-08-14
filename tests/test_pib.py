import argparse
import json

from lma_phase1.collectors.pib import article_failure_reason, import_rows
from lma_phase1.io import iter_jsonl


def test_successful_retry_replaces_failed_pib_attempt(tmp_path) -> None:
    first = tmp_path / "first.jsonl"
    retry = tmp_path / "retry.jsonl"
    first.write_text(
        json.dumps({"prid": "123", "text": "", "error": "timeout"}) + "\n",
        encoding="utf-8",
    )
    retry.write_text(
        json.dumps(
            {
                "prid": "123",
                "text": "यह एक पूरी हिन्दी प्रेस विज्ञप्ति है।",
                "source_url": "https://example/123",
                "page_title": "PIB",
                "error": None,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "output.jsonl.gz"
    report_path = tmp_path / "report.json"
    args = argparse.Namespace(
        input=[str(first), str(retry)],
        output=str(output),
        report=str(report_path),
        index_url="https://example/index",
        collected_at="2026-08-14",
    )

    report = import_rows(args)
    documents = list(iter_jsonl(output))

    assert report["counts"]["documents_written"] == 1
    assert report["counts"]["successful_retries"] == 1
    assert documents[0].document_id == "pib_hindi_123"
    assert documents[0].text == "यह एक पूरी हिन्दी प्रेस विज्ञप्ति है।"


def test_access_denied_body_is_not_treated_as_article() -> None:
    text = "Access Denied\nReference #18\nhttps://errors.edgesuite.net/example"

    assert article_failure_reason(text) == "access_denied"
