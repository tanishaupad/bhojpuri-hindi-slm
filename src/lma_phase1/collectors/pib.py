"""Import browser-collected PIB article text into the canonical corpus schema."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lma_phase1.io import atomic_write_json, write_jsonl
from lma_phase1.schema import Document


def article_failure_reason(text: str) -> str | None:
    """Return why browser text is not a PIB article, or ``None`` when usable."""

    stripped = text.strip()
    if not stripped:
        return "empty_text"
    if stripped.casefold().startswith("access denied") or "errors.edgesuite.net" in stripped:
        return "access_denied"
    return None


def load_latest_rows(
    input_paths: list[Path],
) -> tuple[list[str], dict[str, dict[str, Any]], Counter]:
    """Load browser rows, retaining the last successful attempt for each PRID."""

    order: list[str] = []
    rows: dict[str, dict[str, Any]] = {}
    counters: Counter[str] = Counter()
    for input_path in input_paths:
        with input_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {input_path}:{line_number}: {exc}") from exc
                prid = str(row.get("prid", "")).strip()
                if not prid:
                    raise ValueError(f"Missing PRID at {input_path}:{line_number}")
                counters["attempts"] += 1
                if prid not in rows:
                    order.append(prid)
                elif article_failure_reason(str(row.get("text", ""))) is None:
                    counters["successful_retries"] += 1
                if article_failure_reason(str(row.get("text", ""))) is None or prid not in rows:
                    rows[prid] = row
    return order, rows, counters


def import_rows(args: argparse.Namespace) -> dict[str, Any]:
    """Convert browser output into documents and return an import report."""

    input_paths = [Path(path) for path in args.input]
    order, rows, counters = load_latest_rows(input_paths)
    documents = []
    failed_prids = []
    failures_by_reason: Counter[str] = Counter()
    for prid in order:
        row = rows[prid]
        text = str(row.get("text", "")).strip()
        failure_reason = article_failure_reason(text)
        if failure_reason:
            failed_prids.append(prid)
            failures_by_reason[failure_reason] += 1
            continue
        source_url = str(row.get("source_url", "")).strip() or (
            f"https://www.pib.gov.in/PressReleasePage.aspx?PRID={prid}&reg=3&lang=2"
        )
        documents.append(
            Document(
                document_id=f"pib_hindi_{prid}",
                language="hi",
                text=text,
                source_name="pib_hindi",
                source_url=source_url,
                collection_type="manual",
                collection_method="browser_direct_article",
                license="PIB-reproduction-with-attribution",
                collected_at=args.collected_at,
                source_document_id=prid,
                metadata={
                    "prid": prid,
                    "page_title": row.get("page_title", ""),
                    "source_index_url": args.index_url,
                },
            )
        )
    counters["unique_prids"] = len(order)
    counters["documents_written"] = len(documents)
    counters["failed_prids"] = len(failed_prids)
    write_jsonl(Path(args.output), documents)
    report = {
        "inputs": [str(path) for path in input_paths],
        "output": args.output,
        "collected_at": args.collected_at,
        "counts": dict(sorted(counters.items())),
        "failures_by_reason": dict(sorted(failures_by_reason.items())),
        "failed_prids": failed_prids,
    }
    atomic_write_json(Path(args.report), report)
    return report


def build_parser() -> argparse.ArgumentParser:
    """Build the PIB browser-import CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--index-url",
        default="https://www.pib.gov.in/allrelease.aspx?lang=2&menuid=30&reg=3",
    )
    parser.add_argument("--collected-at", default=datetime.now(UTC).date().isoformat())
    return parser


def main() -> None:
    """Run the PIB browser-import CLI."""

    args = build_parser().parse_args()
    for target in (Path(args.output), Path(args.report)):
        if target.exists():
            raise SystemExit(f"Refusing to overwrite existing output: {target}")
    report = import_rows(args)
    print(
        f"wrote={report['counts']['documents_written']} failed={report['counts']['failed_prids']}"
    )


if __name__ == "__main__":
    main()
