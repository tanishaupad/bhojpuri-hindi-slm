"""Polite, resumable same-domain crawler for approved manual web sources."""

from __future__ import annotations

import argparse
import hashlib
import re
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from lma_phase1.collectors.mediawiki import DEFAULT_USER_AGENT, build_session
from lma_phase1.io import iter_jsonl, write_jsonl
from lma_phase1.schema import Document


def normalize_url(url: str) -> str:
    """Remove fragments and normalize an HTTP(S) URL for crawl state."""

    url, _ = urldefrag(url)
    parsed = urlparse(url)
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    return parsed._replace(path=path, fragment="").geturl()


def initialize_state(path: Path, start_urls: list[str]) -> sqlite3.Connection:
    """Initialize durable URL frontier state in SQLite."""

    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS frontier ("
        "url TEXT PRIMARY KEY, depth INTEGER NOT NULL, status TEXT NOT NULL, error TEXT)"
    )
    connection.executemany(
        "INSERT OR IGNORE INTO frontier(url, depth, status) VALUES (?, 0, 'pending')",
        [(normalize_url(url),) for url in start_urls],
    )
    connection.commit()
    return connection


def extract_text_and_links(
    html_text: str, base_url: str, content_selector: str
) -> tuple[str, str | None, list[str]]:
    """Extract main prose, title, and links from one HTML page."""

    soup = BeautifulSoup(html_text, "html.parser")
    for node in soup.select(
        "script, style, nav, footer, header, aside, form, noscript, iframe, svg, "
        ".twitter-tweet, .instagram-media, .sharedaddy, .jp-relatedposts, .post-tags"
    ):
        node.decompose()
    content = None
    for selector in (item.strip() for item in content_selector.split(",")):
        if selector and (content := soup.select_one(selector)) is not None:
            break
    if content is None:
        content = soup.body
    text = content.get_text("\n", strip=True) if content else ""
    title = soup.title.get_text(" ", strip=True) if soup.title else None
    links = [normalize_url(urljoin(base_url, anchor.get("href"))) for anchor in soup.find_all("a")]
    return text, title, links


def collect(args: argparse.Namespace) -> dict[str, int]:
    """Crawl allowed same-domain pages and append extracted documents."""

    start_urls = [normalize_url(url) for url in args.start_url]
    allowed_hosts = {urlparse(url).netloc for url in start_urls}
    include = re.compile(args.include_regex) if args.include_regex else None
    exclude = re.compile(args.exclude_regex) if args.exclude_regex else None
    output = Path(args.output)
    existing_ids = (
        {document.document_id for document in iter_jsonl(output)} if output.exists() else set()
    )
    connection = initialize_state(Path(args.state_db), start_urls)
    session = build_session(args.user_agent)
    robots: dict[str, RobotFileParser] = {}
    collected_at = datetime.now(UTC).date().isoformat()
    stats = {"fetched": 0, "written": 0, "blocked": 0, "errors": 0}

    while not args.max_pages or stats["fetched"] < args.max_pages:
        row = connection.execute(
            "SELECT url, depth FROM frontier WHERE status='pending' ORDER BY depth, url LIMIT 1"
        ).fetchone()
        if row is None:
            break
        url, depth = row
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in robots:
            parser = RobotFileParser(urljoin(origin, "/robots.txt"))
            try:
                parser.read()
            except OSError:
                parser = RobotFileParser()
                parser.parse(["User-agent: *", "Allow: /"])
            robots[origin] = parser
        if not robots[origin].can_fetch(args.user_agent, url):
            connection.execute("UPDATE frontier SET status='blocked' WHERE url=?", (url,))
            connection.commit()
            stats["blocked"] += 1
            continue

        try:
            response = session.get(url, timeout=(10, 90))
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                connection.execute("UPDATE frontier SET status='ignored' WHERE url=?", (url,))
                connection.commit()
                continue
            text, title, links = extract_text_and_links(response.text, url, args.content_selector)
            stats["fetched"] += 1
            document_id = f"{args.source_name}_{hashlib.sha256(url.encode()).hexdigest()[:20]}"
            if len(text) >= args.minimum_characters and document_id not in existing_ids:
                document = Document(
                    document_id=document_id,
                    language=args.language,
                    text=text,
                    source_name=args.source_name,
                    source_url=url,
                    collection_type="manual",
                    collection_method="polite_html_crawl",
                    license=args.license,
                    collected_at=collected_at,
                    source_document_id=url,
                    metadata={"title": title, "crawl_depth": depth},
                )
                stats["written"] += write_jsonl(output, [document], append=True)
                existing_ids.add(document_id)

            if depth < args.max_depth:
                candidates = []
                for link in links:
                    link_parsed = urlparse(link)
                    if (
                        link_parsed.scheme not in {"http", "https"}
                        or link_parsed.netloc not in allowed_hosts
                    ):
                        continue
                    if include and not include.search(link):
                        continue
                    if exclude and exclude.search(link):
                        continue
                    candidates.append((link, depth + 1))
                connection.executemany(
                    "INSERT OR IGNORE INTO frontier(url, depth, status) VALUES (?, ?, 'pending')",
                    candidates,
                )
            connection.execute("UPDATE frontier SET status='done' WHERE url=?", (url,))
            connection.commit()
        except (requests.RequestException, ValueError) as exc:
            connection.execute(
                "UPDATE frontier SET status='error', error=? WHERE url=?", (str(exc)[:500], url)
            )
            connection.commit()
            stats["errors"] += 1
        time.sleep(args.delay)
    connection.close()
    return stats


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-url", action="append", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state-db", required=True)
    parser.add_argument("--content-selector", default="article, main")
    parser.add_argument("--include-regex")
    parser.add_argument(
        "--exclude-regex",
        default=r"/(tag|category|author|login|privacy|terms|feed|wp-json)(/|$)|[?&](replytocom|share)=",
    )
    parser.add_argument("--minimum-characters", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    return parser


def main() -> None:
    """Run the approved-domain web collector CLI."""

    args = build_parser().parse_args()
    stats = collect(args)
    print(" ".join(f"{key}={value}" for key, value in stats.items()))


if __name__ == "__main__":
    main()
