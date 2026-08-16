"""Resumable page-level MediaWiki collector for the manual corpus portion."""

from __future__ import annotations

import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
import mwparserfromhell
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from lma_phase1.io import atomic_write_json, iter_jsonl, read_json, write_jsonl
from lma_phase1.schema import Document

DEFAULT_USER_AGENT = (
    "LMA-Phase1-CorpusCollector/0.1 "
    "(https://github.com/CL3-410/individual-project-tanishaupad; academic project)"
)


def build_session(user_agent: str) -> requests.Session:
    """Create an HTTP session with bounded retries for transient failures."""

    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def enumerate_pages(
    session: requests.Session,
    api_url: str,
    namespace: int,
    batch_size: int,
    continuation: dict[str, Any] | None,
    category: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Enumerate one stable batch of pages, optionally from one category."""

    parameters: dict[str, Any] = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
    }
    if category:
        parameters.update(
            {
                "list": "categorymembers",
                "cmtitle": category,
                "cmnamespace": namespace,
                "cmtype": "page",
                "cmlimit": batch_size,
            }
        )
        continuation_key = "cmcontinue"
        result_key = "categorymembers"
    else:
        parameters.update(
            {
                "list": "allpages",
                "apnamespace": namespace,
                "apfilterredir": "nonredirects",
                "aplimit": batch_size,
            }
        )
        continuation_key = "apcontinue"
        result_key = "allpages"
    if continuation and continuation_key in continuation:
        parameters[continuation_key] = continuation[continuation_key]
    response = session.get(api_url, params=parameters, timeout=(10, 90))
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"MediaWiki API error: {payload['error']}")
    pages = payload.get("query", {}).get(result_key, [])
    next_continuation = payload.get("continue")
    if next_continuation:
        next_continuation = {continuation_key: next_continuation[continuation_key]}
    return pages, next_continuation


def fetch_page_chunk(
    session: requests.Session, api_url: str, chunk: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Fetch complete text and provenance for one API-safe page chunk."""

    parameters: dict[str, Any] = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "pageids": "|".join(str(page["pageid"]) for page in chunk),
        "prop": "extracts|info|revisions|pageprops",
        "explaintext": "1",
        "exlimit": "max",
        "exsectionformat": "plain",
        "inprop": "url",
        "rvprop": "ids|timestamp",
        "ppprop": "disambiguation",
    }
    merged = {int(page["pageid"]): dict(page) for page in chunk}
    while True:
        response = session.get(api_url, params=parameters, timeout=(10, 90))
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise RuntimeError(f"MediaWiki API error: {payload['error']}")
        for page in payload.get("query", {}).get("pages", []):
            merged.setdefault(int(page["pageid"]), {}).update(page)
        property_continuation = payload.get("continue")
        if not property_continuation:
            break
        parameters.update(property_continuation)
    detailed_pages = []
    for stub in chunk:
        page = merged[stub["pageid"]]
        if not (page.get("extract") or "").strip():
            page["extract"] = fetch_parsed_text(session, api_url, int(page["pageid"]))
            page["extraction_method"] = "mediawiki_parse_html"
        detailed_pages.append(page)
    return detailed_pages


def wikitext_to_plaintext(source: str) -> str:
    """Convert revision wikitext to readable text without rendering page templates."""

    if not source.strip():
        return ""
    parsed = mwparserfromhell.parse(source)
    return parsed.strip_code(normalize=True, collapse=True).strip()


def fetch_revision_chunk(
    session: requests.Session,
    api_url: str,
    chunk: list[dict[str, Any]],
    render_empty_revisions: bool = False,
) -> list[dict[str, Any]]:
    """Fetch one revision-content chunk and strip its wikitext locally."""

    parameters: dict[str, Any] = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "pageids": "|".join(str(page["pageid"]) for page in chunk),
        "prop": "info|revisions|pageprops",
        "inprop": "url",
        "rvprop": "ids|timestamp|content",
        "rvslots": "main",
        "ppprop": "disambiguation",
    }
    response = session.get(api_url, params=parameters, timeout=(10, 90))
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"MediaWiki API error: {payload['error']}")
    detailed_pages = []
    for page in payload.get("query", {}).get("pages", []):
        revisions = page.get("revisions") or []
        revision = revisions[0] if revisions else {}
        slot = (revision.get("slots") or {}).get("main", {})
        source = str(slot.get("content", revision.get("content", "")))
        page["extract"] = wikitext_to_plaintext(source)
        page["extraction_method"] = "mediawiki_revision_wikitext"
        needs_render = not page["extract"] or bool(
            re.search(r"<(?:pages|section)\b", source, flags=re.IGNORECASE)
        )
        if needs_render and render_empty_revisions:
            page["extract"] = fetch_parsed_text(session, api_url, int(page["pageid"]))
            page["extraction_method"] = "mediawiki_parse_html"
        detailed_pages.append(page)
    by_id = {int(page["pageid"]): page for page in detailed_pages}
    return [by_id[int(stub["pageid"])] for stub in chunk if int(stub["pageid"]) in by_id]


def fetch_page_chunk_in_new_session(
    api_url: str,
    chunk: list[dict[str, Any]],
    user_agent: str,
    content_mode: str = "extracts",
    render_empty_revisions: bool = False,
) -> list[dict[str, Any]]:
    """Fetch one chunk in a worker-owned HTTP session."""

    session = build_session(user_agent)
    try:
        if content_mode == "revision":
            return fetch_revision_chunk(session, api_url, chunk, render_empty_revisions)
        return fetch_page_chunk(session, api_url, chunk)
    finally:
        session.close()


def fetch_page_details(
    session: requests.Session,
    api_url: str,
    pages: list[dict[str, Any]],
    workers: int = 1,
    user_agent: str = DEFAULT_USER_AGENT,
    content_mode: str = "extracts",
    render_empty_revisions: bool = False,
) -> list[dict[str, Any]]:
    """Fetch complete text and provenance for enumerated page IDs.

    Property calls are kept at twenty pages so anonymous API limits do not
    return partial records with missing extracts or URLs.
    """

    chunk_size = 10 if content_mode == "revision" and render_empty_revisions else 50
    if content_mode == "extracts":
        chunk_size = 20
    chunks = [pages[offset : offset + chunk_size] for offset in range(0, len(pages), chunk_size)]
    if workers == 1 or len(chunks) <= 1:
        fetcher = fetch_revision_chunk if content_mode == "revision" else fetch_page_chunk
        return [
            page
            for chunk in chunks
            for page in (
                fetcher(session, api_url, chunk, render_empty_revisions)
                if content_mode == "revision"
                else fetcher(session, api_url, chunk)
            )
        ]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        chunk_results = executor.map(
            lambda chunk: fetch_page_chunk_in_new_session(
                api_url, chunk, user_agent, content_mode, render_empty_revisions
            ),
            chunks,
        )
        return [page for chunk_result in chunk_results for page in chunk_result]


def fetch_parsed_text(session: requests.Session, api_url: str, page_id: int) -> str:
    """Render one page through MediaWiki when the TextExtracts extension is unavailable."""

    parameters: dict[str, Any] = {
        "action": "parse",
        "format": "json",
        "formatversion": "2",
        "pageid": page_id,
        "prop": "text",
        "disabletoc": "1",
    }
    response = session.get(api_url, params=parameters, timeout=(10, 90))
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"MediaWiki parse API error: {payload['error']}")
    html_text = payload.get("parse", {}).get("text", "")
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    for node in soup.select(
        "script, style, table, nav, footer, header, aside, form, noscript, "
        ".mw-editsection, .mw-references-wrap, .noprint, .ws-noexport, sup.reference"
    ):
        node.decompose()
    return soup.get_text("\n", strip=True)


def fetch_batch(
    session: requests.Session,
    api_url: str,
    namespace: int,
    batch_size: int,
    continuation: dict[str, Any] | None,
    category: str | None = None,
    workers: int = 1,
    user_agent: str = DEFAULT_USER_AGENT,
    content_mode: str = "extracts",
    render_empty_revisions: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Enumerate pages and fetch complete details without mixed continuations."""

    page_stubs, next_continuation = enumerate_pages(
        session, api_url, namespace, batch_size, continuation, category
    )
    return (
        fetch_page_details(
            session,
            api_url,
            page_stubs,
            workers,
            user_agent,
            content_mode,
            render_empty_revisions,
        ),
        next_continuation,
    )


def page_to_document(
    page: dict[str, Any],
    language: str,
    source_name: str,
    license_name: str,
    collected_at: str,
) -> Document | None:
    """Convert an API page to the common schema, skipping unusable pages."""

    text = (page.get("extract") or "").strip()
    if not text or "disambiguation" in page.get("pageprops", {}):
        return None
    page_id = str(page["pageid"])
    revisions = page.get("revisions") or []
    revision = revisions[0] if revisions else {}
    return Document(
        document_id=f"{source_name}_{page_id}",
        language=language,
        text=text,
        source_name=source_name,
        source_url=page.get("fullurl"),
        collection_type="manual",
        collection_method="mediawiki_action_api",
        license=license_name,
        collected_at=collected_at,
        source_document_id=page_id,
        metadata={
            "title": page.get("title"),
            "namespace": page.get("ns"),
            "revision_id": revision.get("revid"),
            "revision_timestamp": revision.get("timestamp"),
            "extraction_method": page.get("extraction_method", "mediawiki_extracts"),
        },
    )


def collect(args: argparse.Namespace) -> dict[str, Any]:
    """Collect pages until exhaustion or the optional page limit is reached."""

    output = Path(args.output)
    checkpoint = Path(args.checkpoint)
    category = getattr(args, "category", None)
    existing_ids = (
        {document.document_id for document in iter_jsonl(output)} if output.exists() else set()
    )
    state = read_json(checkpoint, default={}) or {}
    continuation = state.get("continuation")
    if state.get("complete"):
        return state

    session = build_session(args.user_agent)
    collected_at = datetime.now(UTC).date().isoformat()
    written_this_run = 0
    skipped_empty = int(state.get("skipped_empty", 0))
    duplicate_ids = int(state.get("duplicate_ids", 0))

    while True:
        request_batch_size = args.batch_size
        if args.max_pages:
            request_batch_size = min(request_batch_size, args.max_pages - written_this_run)
            if request_batch_size <= 0:
                break
        pages, next_continuation = fetch_batch(
            session=session,
            api_url=args.api_url,
            namespace=args.namespace,
            batch_size=request_batch_size,
            continuation=continuation,
            category=category,
            workers=args.workers,
            user_agent=args.user_agent,
            content_mode=args.content_mode,
            render_empty_revisions=args.render_empty_revisions,
        )
        documents = []
        for page in pages:
            document = page_to_document(
                page=page,
                language=args.language,
                source_name=args.source_name,
                license_name=args.license,
                collected_at=collected_at,
            )
            if document is None:
                skipped_empty += 1
                continue
            if document.document_id in existing_ids:
                duplicate_ids += 1
                continue
            documents.append(document)
            existing_ids.add(document.document_id)

        written_this_run += write_jsonl(output, documents, append=True)
        continuation = next_continuation
        state = {
            "api_url": args.api_url,
            "source_name": args.source_name,
            "category": category,
            "output": str(output),
            "records_written": len(existing_ids),
            "written_this_run": written_this_run,
            "skipped_empty": skipped_empty,
            "duplicate_ids": duplicate_ids,
            "continuation": continuation,
            "complete": continuation is None,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        atomic_write_json(checkpoint, state)

        if continuation is None:
            break
        if args.max_pages and written_this_run >= args.max_pages:
            break
        time.sleep(args.delay)
    return state


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--namespace", type=int, default=0)
    parser.add_argument(
        "--category",
        help="Optional exact category title, for example Category:भोजपुरी",
    )
    parser.add_argument("--batch-size", type=int, default=50, choices=range(1, 501))
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        choices=range(1, 9),
        help="Concurrent detail requests (1-8); use small values on public wikis",
    )
    parser.add_argument("--max-pages", type=int, default=0, help="0 collects until exhaustion")
    parser.add_argument(
        "--content-mode",
        choices=("extracts", "revision"),
        default="extracts",
        help="Use fast batched revision wikitext when TextExtracts requires per-page continuation",
    )
    parser.add_argument(
        "--render-empty-revisions",
        action="store_true",
        help="Render empty or explicitly transcluded revision pages (useful for Wikisource works)",
    )
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    return parser


def main() -> None:
    """Run the MediaWiki collector CLI."""

    args = build_parser().parse_args()
    state = collect(args)
    print(
        f"{state['source_name']}: {state['records_written']} records; complete={state['complete']}"
    )


if __name__ == "__main__":
    main()
