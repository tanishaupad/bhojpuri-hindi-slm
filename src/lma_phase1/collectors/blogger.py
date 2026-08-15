"""Resumable extraction from public Blogger JSON feeds."""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from lma_phase1.collectors.mediawiki import DEFAULT_USER_AGENT, build_session
from lma_phase1.io import iter_jsonl, write_jsonl
from lma_phase1.schema import Document


def extract_entry_text(html_text: str) -> str:
    """Convert a Blogger post body to readable plain text."""

    soup = BeautifulSoup(html_text, "html.parser")
    for node in soup.select(
        "script, style, noscript, iframe, svg, form, .separator, .pvc_clear, .pvc_stats, "
        ".twitter-tweet, .instagram-media"
    ):
        node.decompose()
    lines = []
    for raw_line in soup.get_text("\n").splitlines():
        line = re.sub(r"https?://\S+|pic\.twitter\.com/\S+", "", raw_line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def entry_labels(entry: dict[str, Any]) -> list[str]:
    """Return category labels attached to a Blogger entry."""

    return [
        str(category.get("term", "")).strip()
        for category in entry.get("category", [])
        if category.get("term")
    ]


def entry_url(entry: dict[str, Any]) -> str | None:
    """Return the public permalink for a Blogger entry."""

    for link in entry.get("link", []):
        if link.get("rel") == "alternate" and link.get("href"):
            return str(link["href"])
    return None


def entry_identifier(entry: dict[str, Any]) -> str:
    """Return Blogger's stable post identifier."""

    identifier = str(entry.get("id", {}).get("$t", ""))
    return identifier.rsplit(".", maxsplit=1)[-1]


def load_checkpoint(path: Path) -> dict[str, int]:
    """Load durable pagination state."""

    if not path.exists():
        return {"next_start_index": 1}
    return json.loads(path.read_text(encoding="utf-8"))


def save_checkpoint(path: Path, checkpoint: dict[str, int]) -> None:
    """Atomically store pagination state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def collect(args: argparse.Namespace) -> dict[str, int]:
    """Extract posts from a Blogger feed into the common document schema."""

    output = Path(args.output)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    start_index = checkpoint["next_start_index"]
    existing_ids = (
        {document.document_id for document in iter_jsonl(output)} if output.exists() else set()
    )
    excluded_labels = {label.casefold() for label in args.exclude_label}
    session = build_session(args.user_agent)
    collected_at = datetime.now(UTC).date().isoformat()
    stats = {
        "examined": 0,
        "written": 0,
        "excluded_label": 0,
        "too_short": 0,
        "missing_url": 0,
    }

    while not args.max_posts or stats["examined"] < args.max_posts:
        batch_size = args.batch_size
        if args.max_posts:
            batch_size = min(batch_size, args.max_posts - stats["examined"])
        response = session.get(
            args.feed_url,
            params={
                "alt": "json",
                "start-index": start_index,
                "max-results": batch_size,
            },
            timeout=(10, 90),
        )
        response.raise_for_status()
        feed = response.json().get("feed", {})
        entries = feed.get("entry", [])
        if not entries:
            break

        documents: list[Document] = []
        for entry in entries:
            stats["examined"] += 1
            labels = entry_labels(entry)
            if excluded_labels.intersection(label.casefold() for label in labels):
                stats["excluded_label"] += 1
                continue
            url = entry_url(entry)
            if not url:
                stats["missing_url"] += 1
                continue
            source_id = entry_identifier(entry)
            document_id = f"{args.source_name}_{source_id}"
            if document_id in existing_ids:
                continue
            text = extract_entry_text(str(entry.get("content", {}).get("$t", "")))
            if len(text) < args.minimum_characters:
                stats["too_short"] += 1
                continue
            documents.append(
                Document(
                    document_id=document_id,
                    language=args.language,
                    text=text,
                    source_name=args.source_name,
                    source_url=url,
                    collection_type="manual",
                    collection_method="blogger_json_feed",
                    license=args.license,
                    collected_at=collected_at,
                    source_document_id=source_id,
                    metadata={
                        "title": entry.get("title", {}).get("$t"),
                        "published": entry.get("published", {}).get("$t"),
                        "updated": entry.get("updated", {}).get("$t"),
                        "labels": labels,
                    },
                )
            )
            existing_ids.add(document_id)

        stats["written"] += write_jsonl(output, documents, append=True)
        start_index += len(entries)
        total_results = int(feed.get("openSearch$totalResults", {}).get("$t", start_index - 1))
        save_checkpoint(
            checkpoint_path,
            {"next_start_index": start_index, "total_results": total_results},
        )
        if start_index > total_results:
            break
        time.sleep(args.delay)

    return stats


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feed-url", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--exclude-label", action="append", default=[])
    parser.add_argument("--minimum-characters", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-posts", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    return parser


def main() -> None:
    """Run the Blogger feed collector CLI."""

    args = build_parser().parse_args()
    stats = collect(args)
    print(" ".join(f"{key}={value}" for key, value in stats.items()))


if __name__ == "__main__":
    main()
