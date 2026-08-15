"""Collect approved public article pages enumerated by an XML sitemap."""

from __future__ import annotations

import argparse
import random
import re
from datetime import date, datetime
from xml.etree import ElementTree

from lma_phase1.collectors.mediawiki import DEFAULT_USER_AGENT, build_session
from lma_phase1.collectors.web import collect as collect_web


def parse_timestamp(value: str) -> date:
    """Parse a sitemap date or ISO timestamp."""

    normalized = value.strip().replace("Z", "+00:00")
    return datetime.fromisoformat(normalized).date()


def sitemap_entries(xml_text: str) -> tuple[str, list[tuple[str, str | None]]]:
    """Return the root kind and location/last-modified pairs from sitemap XML."""

    root = ElementTree.fromstring(xml_text)
    root_kind = root.tag.rsplit("}", maxsplit=1)[-1]
    entries = []
    for child in root:
        values = {node.tag.rsplit("}", maxsplit=1)[-1]: node.text for node in child}
        location = (values.get("loc") or "").strip()
        if location:
            entries.append((location, values.get("lastmod")))
    return root_kind, entries


def discover_urls(args: argparse.Namespace) -> list[str]:
    """Resolve article URLs from an index while applying date and sitemap filters."""

    session = build_session(args.user_agent)
    response = session.get(args.sitemap_index, timeout=(10, 90))
    response.raise_for_status()
    kind, entries = sitemap_entries(response.text)
    sitemap_pattern = re.compile(args.sitemap_regex)
    sitemap_urls = [
        location
        for location, _ in entries
        if kind == "urlset" or sitemap_pattern.search(location)
    ]
    if kind == "urlset":
        sitemap_urls = [args.sitemap_index]

    cutoff = date.fromisoformat(args.before_date) if args.before_date else None
    discovered = []
    for sitemap_url in sitemap_urls:
        sitemap_response = session.get(sitemap_url, timeout=(10, 90))
        sitemap_response.raise_for_status()
        child_kind, child_entries = sitemap_entries(sitemap_response.text)
        if child_kind != "urlset":
            continue
        for location, last_modified in child_entries:
            if cutoff and last_modified and parse_timestamp(last_modified) >= cutoff:
                continue
            discovered.append(location)
    return sorted(set(discovered))


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sitemap-index", required=True)
    parser.add_argument("--sitemap-regex", default=r"/post-sitemap\d*\.xml$")
    parser.add_argument("--before-date")
    parser.add_argument("--language", required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state-db", required=True)
    parser.add_argument("--content-selector", default=".entry-content, article")
    parser.add_argument("--minimum-characters", type=int, default=200)
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--sample-seed", default="lma-sitemap-pilot")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    return parser


def main() -> None:
    """Run sitemap discovery and the polite HTML collector."""

    args = build_parser().parse_args()
    args.start_url = discover_urls(args)
    if args.sample_size and len(args.start_url) > args.sample_size:
        args.start_url = sorted(
            random.Random(args.sample_seed).sample(args.start_url, args.sample_size)
        )
    args.max_depth = 0
    args.include_regex = None
    args.exclude_regex = None
    print(f"discovered={len(args.start_url)}")
    stats = collect_web(args)
    print(" ".join(f"{key}={value}" for key, value in stats.items()))


if __name__ == "__main__":
    main()
