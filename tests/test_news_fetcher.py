"""Fetcher resilience: circuit breaker, cache fallback, never-raise."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx

from ai_trader.news.feeds import FeedSpec
from ai_trader.news.fetcher import (
    COOLDOWN_S,
    ResilientFeedFetcher,
    parse_feed_datetime,
    parse_feed_xml,
)

NOW = datetime(2026, 8, 26, 12, tzinfo=UTC)
FEED = FeedSpec(name="Test Feed", url="https://example.com/rss")
GOOD_RSS = """<rss><channel>
<item><title>$AAPL beats estimates</title><link>http://x/1</link>
<pubDate>Wed, 26 Aug 2026 10:00:00 GMT</pubDate>
<description>Apple earnings beat</description></item>
<item><title>Story with no date</title><link>http://x/2</link></item>
</channel></rss>"""

ATOM = """<feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>Atom headline</title><link href="http://x/3"/>
<published>2026-08-26T09:00:00Z</published></entry>
</feed>"""

GNEWS_RSS = """<rss><channel>
<item><title>Markets rally - Reuters</title><link>http://n/1</link>
<pubDate>Wed, 26 Aug 2026 10:00:00 GMT</pubDate><source url="https://reuters.com">Reuters</source></item>
</channel></rss>"""


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _response(text, status=200):
    return httpx.Response(status, text=text, request=httpx.Request("GET", FEED.url))


def test_parse_rss_and_atom():
    articles, _ = parse_feed_xml(GOOD_RSS, FEED, NOW)
    assert len(articles) == 2
    assert articles[0].tickers == ("AAPL",)
    assert not articles[0].pub_date_missing
    assert articles[1].pub_date_missing

    atom_articles, _ = parse_feed_xml(ATOM, FEED, NOW)
    assert len(atom_articles) == 1
    assert atom_articles[0].link == "http://x/3"
    assert not atom_articles[0].pub_date_missing


def test_aggregator_lifts_origin_publisher():
    feed = FeedSpec(name="Reuters (GNews)", url="u", is_aggregator=True)
    articles, _ = parse_feed_xml(GNEWS_RSS, feed, NOW)
    assert articles[0].origin_publisher == "Reuters"
    assert articles[0].effective_publisher == "Reuters"


def test_future_pubdate_treated_as_undated():
    xml = GOOD_RSS.replace("26 Aug 2026", "26 Aug 2036")
    articles, _ = parse_feed_xml(xml, FEED, NOW)
    assert articles[0].pub_date_missing


def test_bad_date_formats_are_missing_not_crash():
    assert parse_feed_datetime("not a date") is None
    assert parse_feed_datetime("") is None


def test_two_failures_trip_cooldown_then_recover():
    calls = {"n": 0}

    def transport(url):
        calls["n"] += 1
        raise httpx.ConnectError("boom", request=httpx.Request("GET", url))

    clock = FakeClock()
    fetcher = ResilientFeedFetcher(transport=transport, clock=clock)

    for _ in range(2):
        items, health = fetcher.fetch_feed(FEED, now=NOW)
        assert items == [] and not health.ok

    # Third call: breaker open, transport NOT hit.
    items, health = fetcher.fetch_feed(FEED, now=NOW)
    assert health.on_cooldown and calls["n"] == 2

    # After cooldown expires, fetching resumes.
    clock.t += COOLDOWN_S + 1
    fetcher._transport = lambda url: _response(GOOD_RSS)
    items, health = fetcher.fetch_feed(FEED, now=NOW)
    assert health.ok and len(items) == 2


def test_failure_falls_back_to_cached_items():
    responses = [_response(GOOD_RSS)]

    def transport(url):
        if responses:
            return responses.pop()
        raise httpx.ConnectError("down", request=httpx.Request("GET", url))

    clock = FakeClock()
    fetcher = ResilientFeedFetcher(transport=transport, clock=clock)
    items, health = fetcher.fetch_feed(FEED, now=NOW)
    assert health.ok and len(items) == 2

    clock.t += 40 * 60  # past cache TTL, next call hits the (dead) network
    items, health = fetcher.fetch_feed(FEED, now=NOW)
    assert not health.ok
    assert len(items) == 2  # cached items still served


def test_http_error_never_raises():
    fetcher = ResilientFeedFetcher(transport=lambda url: _response("nope", status=503))
    items, health = fetcher.fetch_feed(FEED, now=NOW)
    assert items == [] and not health.ok
