import pytest

from lma_phase1.collectors.sitemap import (
    build_parser,
    parse_timestamp,
    partition_urls,
    sitemap_entries,
)


def test_sitemap_entries_handles_namespace() -> None:
    xml = """<?xml version="1.0"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://example.com/a</loc><lastmod>2022-01-02T03:04:05+00:00</lastmod></url>
      <url><loc>https://example.com/b</loc></url>
    </urlset>
    """

    assert sitemap_entries(xml) == (
        "urlset",
        [
            ("https://example.com/a", "2022-01-02T03:04:05+00:00"),
            ("https://example.com/b", None),
        ],
    )


def test_parse_timestamp_accepts_date_and_zulu_timestamp() -> None:
    assert parse_timestamp("2021-05-06").isoformat() == "2021-05-06"
    assert parse_timestamp("2021-05-06T01:02:03Z").isoformat() == "2021-05-06"


def test_partition_urls_is_disjoint_and_complete() -> None:
    urls = [f"https://example.com/{index}" for index in range(10)]
    partitions = [partition_urls(urls, 3, index) for index in range(3)]

    assert sorted(url for partition in partitions for url in partition) == sorted(urls)
    assert not set(partitions[0]).intersection(partitions[1])


def test_partition_urls_rejects_invalid_index() -> None:
    with pytest.raises(ValueError, match="partition index"):
        partition_urls(["https://example.com"], 2, 2)


def test_discovery_delay_and_cache_defaults() -> None:
    args = build_parser().parse_args(
        [
            "--sitemap-index",
            "https://example.test/sitemap.xml",
            "--language",
            "bho",
            "--source-name",
            "example",
            "--license",
            "research",
            "--output",
            "output.jsonl.gz",
            "--state-db",
            "state.sqlite",
        ]
    )

    assert args.discovery_delay == 2.0
    assert args.url_cache is None
