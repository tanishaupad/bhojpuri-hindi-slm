from lma_phase1.collectors.sitemap import parse_timestamp, sitemap_entries


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
