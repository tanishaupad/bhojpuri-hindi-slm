"""Resumable WordPress REST collector for manually gathered monolingual articles."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from lma_phase1.collectors.mediawiki import DEFAULT_USER_AGENT, build_session
from lma_phase1.io import atomic_write_json, iter_jsonl, read_json, write_jsonl
from lma_phase1.schema import Document


def html_text(value: str) -> str:
    soup = BeautifulSoup(value, "html.parser")
    for node in soup.select("script, style, nav, footer, header, aside, form, iframe, svg"):
        node.decompose()
    return soup.get_text("\n", strip=True)


def collect(args: argparse.Namespace) -> dict[str, int | str | bool]:
    output = Path(args.output)
    checkpoint = Path(args.checkpoint)
    state = read_json(checkpoint, default={}) or {}
    if state.get("complete"):
        return state
    page = int(state.get("next_page", 1))
    existing = {d.document_id for d in iter_jsonl(output)} if output.exists() else set()
    session = build_session(args.user_agent)
    collected_at = datetime.now(UTC).date().isoformat()
    written = len(existing)
    while not args.max_pages or page <= args.max_pages:
        try:
            response = session.get(
                args.api_url,
                params={
                    "page": page,
                    "per_page": args.per_page,
                    "orderby": "id",
                    "order": "asc",
                    "_fields": "id,link,date,title,content",
                },
                timeout=(10, 90),
            )
            if response.status_code == 400 and "rest_post_invalid_page_number" in response.text:
                state["complete"] = True
                break
            response.raise_for_status()
        except requests.RequestException:
            atomic_write_json(checkpoint, {**state, "next_page": page, "records_written": written})
            raise
        documents = []
        for row in response.json():
            identifier = f"{args.source_name}_{row['id']}"
            if identifier in existing:
                continue
            text = html_text(str((row.get("content") or {}).get("rendered", "")))
            if len(text) < args.minimum_characters:
                continue
            link = str(row.get("link") or "") or None
            documents.append(
                Document(
                    document_id=identifier,
                    language=args.language,
                    text=text,
                    source_name=args.source_name,
                    source_url=link,
                    collection_type="manual",
                    collection_method="wordpress_rest_api",
                    license=args.license,
                    collected_at=collected_at,
                    source_document_id=str(row["id"]),
                    metadata={
                        "title": html_text(str((row.get("title") or {}).get("rendered", ""))),
                        "published_at": row.get("date"),
                    },
                )
            )
            existing.add(identifier)
        written += write_jsonl(output, documents, append=True)
        page += 1
        state = {
            "api_url": args.api_url,
            "source_name": args.source_name,
            "output": str(output),
            "next_page": page,
            "records_written": written,
            "complete": False,
        }
        atomic_write_json(checkpoint, state)
    else:
        state["complete"] = False
    state.update({"next_page": page, "records_written": written})
    atomic_write_json(checkpoint, state)
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--per-page", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--minimum-characters", type=int, default=200)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    return parser


def main() -> None:
    state = collect(build_parser().parse_args())
    print(f"{state['source_name']}: {state['records_written']} records; complete={state.get('complete')}")


if __name__ == "__main__":
    main()
