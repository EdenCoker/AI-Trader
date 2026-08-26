"""End-to-end: both acquisition paths → archive → stories → SignalBundle."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from ai_trader.config import AppSettings
from ai_trader.domain.signals import SignalBundle, SignalDirection
from ai_trader.news.archive import NewsArchive
from ai_trader.news.feeds import FeedSpec
from ai_trader.news.fetcher import ResilientFeedFetcher
from ai_trader.news.models import CoverageState
from ai_trader.news.pipeline import NewsIntelligenceEngine
from ai_trader.news.signals import build_news_signals
from ai_trader.news.worldmonitor import WorldMonitorNewsProvider

NOW = datetime(2026, 8, 26, 12, tzinfo=UTC)
PUB_MS = int((NOW - timedelta(hours=3)).timestamp() * 1000)

WM_DIGEST = {
    "categories": {
        "markets": {
            "items": [
                {
                    "source": "Reuters Business",
                    "title": "Acme Corp cuts full-year guidance on weak demand",
                    "link": "https://wm/acme-guidance",
                    "publishedAt": PUB_MS,
                    "importanceScore": 78,
                    "credibilityScore": 90,
                    "tickers": ["ACME"],
                    "snippet": "Acme lowered its outlook...",
                }
            ]
        }
    },
    "coverage": {"state": "complete"},
}

RSS_XML = f"""<rss><channel>
<item><title>Acme cuts its full-year guidance as demand weakens - CNBC</title>
<link>https://cnbc/acme</link>
<pubDate>{(NOW - timedelta(hours=2)).strftime("%a, %d %b %Y %H:%M:%S GMT")}</pubDate>
<description>$ACME shares fall after the guidance cut</description></item>
</channel></rss>"""


def build_engine(tmp_path: Path) -> NewsIntelligenceEngine:
    wm = WorldMonitorNewsProvider(
        settings=AppSettings(),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text=json.dumps(WM_DIGEST))
        ),
    )
    fetcher = ResilientFeedFetcher(
        transport=lambda url: httpx.Response(
            200, text=RSS_XML, request=httpx.Request("GET", url)
        )
    )
    return NewsIntelligenceEngine(
        settings=AppSettings(),
        archive=NewsArchive(tmp_path / "arch.jsonl"),
        fetcher=fetcher,
        worldmonitor=wm,
        feeds=(FeedSpec(name="CNBC Top News", url="https://example.com/rss"),),
        worldmonitor_enabled=True,
    )


def test_full_flow_to_signal_bundle(tmp_path):
    engine = build_engine(tmp_path)
    report = engine.collect(now=NOW)
    assert report.coverage is CoverageState.COMPLETE
    assert report.article_count == 2
    assert report.publisher_count == 2  # reuters + cnbc

    as_of = NOW + timedelta(minutes=10)
    stories = engine.stories(as_of)
    assert len(stories) == 1, [s.canonical_title for s in stories]
    story = stories[0]
    assert story.publisher_count == 2
    assert story.classification.category == "guidance_cut"
    assert story.tickers == ("ACME",)
    # Server scores arrived on the worldmonitor member and blended in.
    assert any(a.server_importance == 78 for a in story.articles)

    signals = build_news_signals(stories, "ACME", as_of)
    assert len(signals) == 1
    signal = signals[0]
    assert signal.direction is SignalDirection.SHORT
    assert signal.effective_date == NOW.date()
    assert 0 < signal.strength <= 1 and 0 < signal.confidence <= 1

    bundle = SignalBundle(ticker="ACME", as_of=as_of.date(), signals=signals)
    assert bundle.direction is SignalDirection.SHORT
    assert bundle.conviction > 0.3
