"""WorldMonitor digest provider: wire-shape mapping, coverage honesty, never-raise."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import httpx

from ai_trader.config import AppSettings
from ai_trader.news.models import CoverageState
from ai_trader.news.worldmonitor import WorldMonitorNewsProvider

NOW = datetime(2026, 8, 26, 12, tzinfo=UTC)
PUB_MS = int((NOW - timedelta(hours=2)).timestamp() * 1000)

DIGEST = {
    "categories": {
        "markets": {
            "items": [
                {
                    "source": "Reuters Business",
                    "title": "Apple beats estimates with record revenue",
                    "link": "https://x/apple",
                    "publishedAt": PUB_MS,
                    "isAlert": False,
                    "importanceScore": 72,
                    "credibilityScore": 88,
                    "corroborationCount": 4,
                    "tickers": ["AAPL"],
                    "snippet": "Apple reported...",
                    "storyMeta": {
                        "firstSeen": PUB_MS,
                        "mentionCount": 3,
                        "sourceCount": 4,
                        "phase": "STORY_PHASE_DEVELOPING",
                    },
                    "threat": {
                        "level": "THREAT_LEVEL_LOW",
                        "category": "economic",
                        "confidence": 0.8,
                        "source": "keyword",
                    },
                },
                {
                    "source": "Future Feed",
                    "title": "Backdated-to-the-future story",
                    "link": "https://x/future",
                    "publishedAt": int((NOW + timedelta(days=1)).timestamp() * 1000),
                    "tickers": [],
                },
            ]
        }
    },
    "feedStatuses": {},
    "generatedAt": NOW.isoformat(),
    "coverage": {"state": "partial", "itemsServed": 2},
}


def provider_with(payload, status=200):
    def handler(request):
        return httpx.Response(status, text=json.dumps(payload))

    return WorldMonitorNewsProvider(
        settings=AppSettings(), transport=httpx.MockTransport(handler)
    )


def test_digest_items_map_to_observed_articles():
    result = provider_with(DIGEST).fetch_digest(now=NOW)
    assert result.coverage is CoverageState.PARTIAL
    assert len(result.articles) == 2
    apple = result.articles[0]
    assert apple.tickers == ("AAPL",)
    assert apple.server_importance == 72
    assert apple.server_credibility == 88
    assert apple.provenance == "worldmonitor"
    assert apple.metadata["story_phase"] == "developing"
    assert not apple.pub_date_missing
    assert apple.first_seen_at == NOW


def test_future_published_at_is_undated():
    result = provider_with(DIGEST).fetch_digest(now=NOW)
    future = result.articles[1]
    assert future.pub_date_missing and future.published_at is None


def test_zero_category_success_is_unavailable():
    result = provider_with({"categories": {}, "coverage": {"state": "complete"}}).fetch_digest(
        now=NOW
    )
    assert result.coverage is CoverageState.UNAVAILABLE
    assert result.articles == []


def test_http_error_degrades_to_unavailable_not_raise():
    result = provider_with({}, status=503).fetch_digest(now=NOW)
    assert result.coverage is CoverageState.UNAVAILABLE


def test_malformed_payload_never_raises():
    payload = {"categories": {"x": {"items": [None, 42, {"title": ""}]}}}
    result = provider_with(payload).fetch_digest(now=NOW)
    assert result.articles == []


def test_news_provider_protocol_filters_by_ticker_and_observation_window():
    provider = provider_with(DIGEST)
    # fetch_news windows on OBSERVATION time (first_seen_at = the live
    # fetch), never the publisher-claimed pubDate — so the window must
    # bracket "now", and a historical window returns nothing.
    now = datetime.now(UTC)
    articles = provider.fetch_news(["AAPL"], now - timedelta(hours=1), now + timedelta(hours=1))
    assert len(articles) == 1
    assert articles[0].tickers == ("AAPL",)
    none = provider.fetch_news(["TSLA"], now - timedelta(hours=1), now + timedelta(hours=1))
    assert list(none) == []
    historical = provider.fetch_news(["AAPL"], now - timedelta(days=9), now - timedelta(days=8))
    assert list(historical) == []


def test_non_dict_json_payload_is_unavailable_not_crash():
    for payload in ([1, 2, 3], "a string", 42, None):
        result = provider_with(payload).fetch_digest(now=NOW)
        assert result.coverage is CoverageState.UNAVAILABLE
        assert result.articles == []


def test_negative_and_nonfinite_scores_are_clamped_or_dropped():
    digest = {
        "categories": {
            "markets": {
                "items": [
                    {
                        "source": "X",
                        "title": "Negative score story",
                        "link": "https://x/neg",
                        "publishedAt": PUB_MS,
                        "importanceScore": -50,
                        "credibilityScore": 1e9,
                    }
                ]
            }
        }
    }
    result = provider_with(digest).fetch_digest(now=NOW)
    article = result.articles[0]
    assert article.server_importance == 0  # clamped, not negative
    assert article.server_credibility == 100  # clamped, not 1e9


def test_digest_item_caps_bound_the_payload():
    items = [
        {"source": "X", "title": f"Story {i}", "link": f"https://x/{i}", "publishedAt": PUB_MS}
        for i in range(200)
    ]
    digest = {"categories": {"markets": {"items": items}}}
    result = provider_with(digest).fetch_digest(now=NOW)
    assert len(result.articles) == 50  # MAX_ITEMS_PER_CATEGORY


def test_overlong_titles_are_truncated():
    digest = {
        "categories": {
            "markets": {
                "items": [
                    {
                        "source": "X",
                        "title": "A" * 5000,
                        "link": "https://x/long",
                        "publishedAt": PUB_MS,
                    }
                ]
            }
        }
    }
    result = provider_with(digest).fetch_digest(now=NOW)
    assert len(result.articles[0].title) == 300
