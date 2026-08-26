"""World Monitor digest provider.

Pulls the public finance news digest from a worldmonitor deployment
(https://github.com/koala73/worldmonitor). The digest endpoint is the
same call the dashboard makes from the browser: items arrive already
clustered server-side with importance/credibility scores, per-item
tickers, story lifecycle metadata, and an honesty ``coverage`` block.

This module talks to the HTTP API only — no worldmonitor (AGPL) code is
vendored. The upstream scores are carried as ``server_importance`` /
``server_credibility`` and blended in :mod:`ai_trader.news.scoring`;
local classification and clustering still run so the pipeline degrades to
identical behavior when this provider is disabled or unreachable.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import httpx

from ai_trader.config import AppSettings, get_settings
from ai_trader.domain.events import NewsArticle
from ai_trader.news.models import CoverageState, ObservedArticle

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.worldmonitor.app"
DIGEST_PATH = "/api/news/v1/list-feed-digest"
REQUEST_TIMEOUT_S = 15.0

_PHASE_MAP = {
    "STORY_PHASE_BREAKING": "breaking",
    "STORY_PHASE_DEVELOPING": "developing",
    "STORY_PHASE_SUSTAINED": "sustained",
    "STORY_PHASE_FADING": "fading",
}


def _parse_published_at(value: Any, now: datetime) -> datetime | None:
    """Digest items carry epoch milliseconds; accept ISO strings too so a
    contract drift degrades to 'undated' rather than a crash."""

    if isinstance(value, int | float) and value > 0:
        try:
            parsed = datetime.fromtimestamp(value / 1000.0, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    elif isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
        except ValueError:
            return None
    else:
        return None
    # Future-dated beyond clock skew → treated as undated, same rule as
    # the RSS path: no source may claim tomorrow's freshness.
    if (parsed - now).total_seconds() > 3600:
        return None
    return parsed


def _score_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return round(value)


class WorldMonitorDigestResult:
    """Digest pull outcome: articles plus the honesty coverage state."""

    def __init__(
        self,
        articles: list[ObservedArticle],
        coverage: CoverageState,
        raw_coverage: dict[str, Any] | None,
    ) -> None:
        self.articles = articles
        self.coverage = coverage
        self.raw_coverage = raw_coverage


class WorldMonitorNewsProvider:
    """Implements the repo ``NewsProvider`` protocol over the digest."""

    def __init__(
        self,
        *,
        settings: AppSettings | None = None,
        base_url: str | None = None,
        variant: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._base_url = (
            base_url
            or getattr(self._settings, "worldmonitor_base_url", None)
            or DEFAULT_BASE_URL
        ).rstrip("/")
        self._variant = variant or getattr(self._settings, "news_variant", None) or "finance"
        self._transport = transport

    def fetch_digest(self, now: datetime | None = None) -> WorldMonitorDigestResult:
        """Pull and normalize the digest. Never raises; an unreachable or
        malformed digest returns zero articles with UNAVAILABLE coverage."""

        now = now or datetime.now(UTC)
        url = f"{self._base_url}{DIGEST_PATH}"
        try:
            with httpx.Client(
                timeout=REQUEST_TIMEOUT_S,
                transport=self._transport,
                headers={"User-Agent": "ai-trader-news/0.1 (+research)"},
            ) as client:
                response = client.get(
                    url, params={"variant": self._variant, "lang": "en"}
                )
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:  # noqa: BLE001 — availability beats purity
            log.warning("worldmonitor digest fetch failed: %s", exc)
            return WorldMonitorDigestResult([], CoverageState.UNAVAILABLE, None)

        return self._normalize(payload, now)

    def _normalize(
        self, payload: dict[str, Any], now: datetime
    ) -> WorldMonitorDigestResult:
        categories = payload.get("categories")
        if not isinstance(categories, dict) or not categories:
            return WorldMonitorDigestResult([], CoverageState.UNAVAILABLE, None)

        raw_coverage = payload.get("coverage")
        coverage = CoverageState.COMPLETE
        if isinstance(raw_coverage, dict):
            try:
                coverage = CoverageState(str(raw_coverage.get("state", "complete")))
            except ValueError:
                coverage = CoverageState.PARTIAL

        articles: list[ObservedArticle] = []
        for category, bucket in categories.items():
            items = bucket.get("items") if isinstance(bucket, dict) else None
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                article = self._normalize_item(item, category, now)
                if article is not None:
                    articles.append(article)

        if not articles:
            # A digest with categories but zero items is an outage wearing
            # a success status; report it as such.
            return WorldMonitorDigestResult([], CoverageState.UNAVAILABLE, raw_coverage)
        return WorldMonitorDigestResult(
            articles, coverage, raw_coverage if isinstance(raw_coverage, dict) else None
        )

    def _normalize_item(
        self, item: dict[str, Any], category: str, now: datetime
    ) -> ObservedArticle | None:
        title = str(item.get("title") or "").strip()
        if not title:
            return None
        link = str(item.get("link") or "").strip()
        source = str(item.get("source") or "World Monitor").strip()
        published_at = _parse_published_at(item.get("publishedAt"), now)

        tickers_raw = item.get("tickers")
        tickers = (
            tuple(str(t).upper() for t in tickers_raw if isinstance(t, str))[:8]
            if isinstance(tickers_raw, list)
            else ()
        )

        story_meta = item.get("storyMeta") if isinstance(item.get("storyMeta"), dict) else {}
        threat = item.get("threat") if isinstance(item.get("threat"), dict) else {}

        digest = hashlib.sha256(f"wm|{source}|{link}|{title}".encode()).hexdigest()[:16]
        return ObservedArticle(
            article_id=digest,
            title=title,
            link=link,
            source_name=source,
            published_at=published_at,
            pub_date_missing=published_at is None,
            first_seen_at=now,
            last_seen_at=now,
            tickers=tickers,
            snippet=str(item.get("snippet") or "")[:500],
            provenance="worldmonitor",
            server_importance=_score_or_none(item.get("importanceScore")),
            server_credibility=_score_or_none(item.get("credibilityScore")),
            metadata={
                "category": category,
                "corroboration_count": item.get("corroborationCount"),
                "is_alert": bool(item.get("isAlert")),
                "threat_level": threat.get("level"),
                "threat_category": threat.get("category"),
                "story_phase": _PHASE_MAP.get(str(story_meta.get("phase", ""))),
                "story_mention_count": story_meta.get("mentionCount"),
                "story_source_count": story_meta.get("sourceCount"),
            },
        )

    # -- NewsProvider protocol -------------------------------------------
    def fetch_news(
        self,
        tickers: Sequence[str],
        start: datetime,
        end: datetime,
    ) -> Sequence[NewsArticle]:
        wanted = {ticker.upper() for ticker in tickers}
        result = self.fetch_digest()
        out: list[NewsArticle] = []
        for article in result.articles:
            if wanted and not wanted.intersection(article.tickers):
                continue
            stamp = article.published_at or article.first_seen_at
            if not (start <= stamp <= end):
                continue
            out.append(article.to_domain())
        return out
