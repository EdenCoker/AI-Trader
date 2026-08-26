"""Domain models for the news intelligence pipeline.

The pipeline distinguishes three time axes and never conflates them:

- ``published_at``: what the publisher claims. Used for recency scoring and
  the freshness floor only. Publishers can backdate this, so it is NEVER an
  effectiveness date.
- ``first_seen_at``: when THIS system first observed the article. This is the
  no-lookahead anchor: ``Signal.effective_date`` derives from it, mirroring
  the repo-wide disclosure-date / filing-date rule.
- ``last_seen_at``: most recent sighting, used for story phase (fading).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ai_trader.domain.events import NewsArticle


class CoverageState(str, Enum):
    """Honesty label for an acquisition pass, mirroring the closed
    vocabulary used by the worldmonitor digest coverage ledger."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    STALE = "stale"
    UNAVAILABLE = "unavailable"

    @property
    def confidence_multiplier(self) -> float:
        if self is CoverageState.COMPLETE:
            return 1.0
        if self is CoverageState.PARTIAL:
            return 0.85
        if self is CoverageState.STALE:
            return 0.6
        return 0.0


class StoryPhase(str, Enum):
    BREAKING = "breaking"
    DEVELOPING = "developing"
    SUSTAINED = "sustained"
    FADING = "fading"


class ObservedArticle(BaseModel):
    """One sighted article with full provenance.

    Wraps the repo-level :class:`NewsArticle` contract with the fields the
    scoring, clustering, and no-lookahead layers need.
    """

    model_config = ConfigDict(frozen=True)

    article_id: str
    title: str
    link: str
    source_name: str
    # Originating publisher when the feed is an aggregator (Google News
    # stamps the real outlet in the RSS <source> element). Empty when the
    # feed itself is the publisher.
    origin_publisher: str = ""
    published_at: datetime | None = None
    pub_date_missing: bool = False
    first_seen_at: datetime
    last_seen_at: datetime
    sighting_count: int = Field(default=1, ge=1)
    tickers: tuple[str, ...] = ()
    snippet: str = ""
    provenance: str = "rss"  # "rss" | "worldmonitor"
    # Upstream scores when provenance == "worldmonitor" (0-100), else None.
    server_importance: int | None = None
    server_credibility: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def effective_publisher(self) -> str:
        return self.origin_publisher or self.source_name

    def effective_published_ms(self) -> float:
        """Recency timestamp for ranking. Undated items return 0 so they
        sort last and fail every positive-duration recency gate — an
        undated item must never claim freshness."""

        if self.pub_date_missing or self.published_at is None:
            return 0.0
        return self.published_at.timestamp() * 1000.0

    def to_domain(self) -> NewsArticle:
        from ai_trader.domain.events import SourceName

        return NewsArticle(
            article_id=self.article_id,
            published_at=self.published_at or self.first_seen_at,
            title=self.title,
            source_name=self.effective_publisher,
            tickers=self.tickers,
            summary=self.snippet or None,
            url=self.link or None,
            source=SourceName.INTERNAL,
        )


class EventClassification(BaseModel):
    """Deterministic finance-event classification for one headline."""

    model_config = ConfigDict(frozen=True)

    category: str
    severity: int = Field(ge=0, le=100)
    polarity: int = Field(ge=-1, le=1)
    confidence: float = Field(ge=0, le=1)
    matched_keywords: tuple[str, ...] = ()
    historical: bool = False


class NewsStory(BaseModel):
    """A clustered story: several sightings of one real-world event."""

    model_config = ConfigDict(frozen=True)

    story_id: str
    canonical_title: str
    articles: tuple[ObservedArticle, ...]
    classification: EventClassification
    importance_score: int = Field(ge=0, le=200)
    credibility_score: int = Field(ge=0, le=100)
    publisher_count: int = Field(ge=1)
    phase: StoryPhase
    first_seen_at: datetime
    last_seen_at: datetime
    tickers: tuple[str, ...] = ()
    coverage: CoverageState = CoverageState.COMPLETE

    @property
    def primary(self) -> ObservedArticle:
        return self.articles[0]

    @property
    def mention_count(self) -> int:
        return sum(article.sighting_count for article in self.articles)


class FeedHealth(BaseModel):
    """Per-feed acquisition outcome for the coverage ledger."""

    model_config = ConfigDict(frozen=True)

    feed_name: str
    ok: bool
    item_count: int = 0
    dropped_undated: int = 0
    failure: str | None = None
    on_cooldown: bool = False
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AcquisitionReport(BaseModel):
    """Outcome of one collect() pass across every acquisition path."""

    model_config = ConfigDict(frozen=True)

    fetched_at: datetime
    coverage: CoverageState
    article_count: int
    publisher_count: int
    feed_healths: tuple[FeedHealth, ...] = ()
    worldmonitor_coverage: str | None = None
    notes: tuple[str, ...] = ()
