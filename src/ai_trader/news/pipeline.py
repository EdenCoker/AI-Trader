"""News intelligence engine: acquire → archive → cluster → score.

``collect()`` is the only method that touches the network; everything
else reads the sighting archive at an explicit ``as_of``, so live runs
and backtest replays execute the same code path with the same
no-lookahead guarantee.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ai_trader.config import AppSettings, get_settings
from ai_trader.news.archive import NewsArchive
from ai_trader.news.classify import classify_headline
from ai_trader.news.feeds import FINANCE_FEEDS, FeedSpec
from ai_trader.news.fetcher import ResilientFeedFetcher
from ai_trader.news.identity import (
    SIMILARITY_THRESHOLD,
    normalize_title,
    similarity,
    story_id_for,
    story_vector,
)
from ai_trader.news.models import (
    AcquisitionReport,
    CoverageState,
    EventClassification,
    NewsStory,
    ObservedArticle,
    StoryPhase,
)
from ai_trader.news.publishers import count_publisher_families
from ai_trader.news.scoring import (
    credibility_score,
    importance_score,
    is_within_freshness_floor,
)
from ai_trader.news.worldmonitor import WorldMonitorNewsProvider

log = logging.getLogger(__name__)

_DEVELOPING_MAX_AGE = timedelta(hours=2)
_FADING_SILENCE = timedelta(hours=24)


def derive_phase(
    *,
    mention_count: int,
    first_seen_at: datetime,
    last_seen_at: datetime,
    as_of: datetime,
) -> StoryPhase:
    if as_of - last_seen_at > _FADING_SILENCE:
        return StoryPhase.FADING
    if mention_count <= 1:
        return StoryPhase.BREAKING
    if mention_count <= 5 and as_of - first_seen_at < _DEVELOPING_MAX_AGE:
        return StoryPhase.DEVELOPING
    return StoryPhase.SUSTAINED


def _tickers_disjoint(a: ObservedArticle, b: ObservedArticle) -> bool:
    """Finance guard: two articles that each name tickers, with no ticker
    in common, are different stories no matter how similar the wording
    ("Apple beats Q3" / "Amazon beats Q3")."""

    return bool(a.tickers) and bool(b.tickers) and not set(a.tickers) & set(b.tickers)


# Secondary merge floor: two articles that share a ticker AND classify to
# the same (non-general) event category are the same story at a much lower
# lexical bar — cross-outlet paraphrases ("cuts guidance on weak demand" /
# "cuts its guidance as demand weakens") routinely land between the two
# thresholds, and ticker+event identity is a prior the title comparator
# doesn't have.
_SECONDARY_THRESHOLD = 0.45


def cluster_articles(articles: list[ObservedArticle]) -> list[list[ObservedArticle]]:
    """Cluster into stories.

    Join rules, in precedence order:
    1. Disjoint non-empty ticker sets are a hard veto — "Apple beats Q3"
       and "Amazon beats Q3" never merge, however similar the wording.
    2. Title similarity >= SIMILARITY_THRESHOLD merges.
    3. Title similarity >= _SECONDARY_THRESHOLD merges when both articles
       share a ticker and the same non-general event category.
    """

    if not articles:
        return []
    vectors = [story_vector(article.title) for article in articles]
    categories = [
        classify_headline(article.title, article.snippet).category for article in articles
    ]
    clusters: list[list[int]] = []
    for i, vector in enumerate(vectors):
        placed = False
        for cluster in clusters:
            joinable = False
            vetoed = False
            for j in cluster:
                if _tickers_disjoint(articles[i], articles[j]):
                    vetoed = True
                    break
                if vector is None or vectors[j] is None:
                    continue
                sim = similarity(vector, vectors[j])
                if sim >= SIMILARITY_THRESHOLD:
                    joinable = True
                elif (
                    sim >= _SECONDARY_THRESHOLD
                    and categories[i] == categories[j]
                    and categories[i] != "general"
                    and set(articles[i].tickers) & set(articles[j].tickers)
                ):
                    joinable = True
            if joinable and not vetoed:
                cluster.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    return [[articles[i] for i in cluster] for cluster in clusters]


def _story_classification(members: list[ObservedArticle]) -> EventClassification:
    best: EventClassification | None = None
    for member in members:
        candidate = classify_headline(member.title, member.snippet)
        if best is None or (candidate.severity, candidate.confidence) > (
            best.severity,
            best.confidence,
        ):
            best = candidate
    assert best is not None
    return best


def _story_tickers(members: list[ObservedArticle]) -> tuple[str, ...]:
    counts: Counter[str] = Counter()
    order: dict[str, int] = {}
    for position, member in enumerate(members):
        for ticker in member.tickers:
            counts[ticker] += 1
            order.setdefault(ticker, position)
    ranked = sorted(counts, key=lambda t: (-counts[t], order[t]))
    return tuple(ranked[:8])


def build_story(
    members: list[ObservedArticle],
    as_of: datetime,
    coverage: CoverageState,
) -> NewsStory:
    # Canonical identity anchors on the earliest OBSERVED member — a
    # late-arriving wording (or a backdated pubDate) cannot steal the id.
    def anchor_key(article: ObservedArticle) -> tuple[datetime, str]:
        return (article.first_seen_at, normalize_title(article.title) or article.article_id)

    anchor = min(members, key=anchor_key)
    normalized = normalize_title(anchor.title)
    story_id = story_id_for(normalized or f"untrackable:{anchor.article_id}")

    publisher_count = count_publisher_families(
        [member.effective_publisher for member in members]
    )
    classification = _story_classification(members)
    importance = max(
        importance_score(classification, member, publisher_count, now=as_of)
        for member in members
    )
    credibility = max(
        credibility_score(member, publisher_count) for member in members
    )
    first_seen = min(member.first_seen_at for member in members)
    last_seen = max(member.last_seen_at for member in members)
    mention_count = sum(member.sighting_count for member in members)

    ordered = sorted(
        members,
        key=lambda m: (-m.effective_published_ms(), m.first_seen_at, m.article_id),
    )
    return NewsStory(
        story_id=story_id,
        canonical_title=anchor.title,
        articles=tuple(ordered),
        classification=classification,
        importance_score=importance,
        credibility_score=credibility,
        publisher_count=publisher_count,
        phase=derive_phase(
            mention_count=mention_count,
            first_seen_at=first_seen,
            last_seen_at=last_seen,
            as_of=as_of,
        ),
        first_seen_at=first_seen,
        last_seen_at=last_seen,
        tickers=_story_tickers(members),
        coverage=coverage,
    )


class NewsIntelligenceEngine:
    def __init__(
        self,
        *,
        settings: AppSettings | None = None,
        archive: NewsArchive | None = None,
        fetcher: ResilientFeedFetcher | None = None,
        worldmonitor: WorldMonitorNewsProvider | None = None,
        feeds: tuple[FeedSpec, ...] = FINANCE_FEEDS,
        worldmonitor_enabled: bool | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        archive_path = Path(
            getattr(self._settings, "news_archive_path", "data/live/news_archive.jsonl")
        )
        self._archive = archive or NewsArchive(archive_path)
        self._fetcher = fetcher or ResilientFeedFetcher()
        self._worldmonitor = worldmonitor
        self._feeds = feeds
        if worldmonitor_enabled is None:
            worldmonitor_enabled = bool(
                getattr(self._settings, "news_worldmonitor_enabled", True)
            )
        self._worldmonitor_enabled = worldmonitor_enabled
        self._max_age_hours = int(getattr(self._settings, "news_max_age_hours", 96))
        self._last_coverage: CoverageState = CoverageState.UNAVAILABLE

    @property
    def archive(self) -> NewsArchive:
        return self._archive

    def collect(self, now: datetime | None = None) -> AcquisitionReport:
        """One acquisition pass across every enabled path. Never raises."""

        now = now or datetime.now(UTC)
        articles: list[ObservedArticle] = []
        notes: list[str] = []
        wm_coverage: str | None = None

        if self._worldmonitor_enabled:
            provider = self._worldmonitor or WorldMonitorNewsProvider(
                settings=self._settings
            )
            result = provider.fetch_digest(now=now)
            wm_coverage = result.coverage.value
            articles.extend(result.articles)
            if result.coverage is CoverageState.UNAVAILABLE:
                notes.append("worldmonitor digest unavailable")

        healths = []
        for feed in self._feeds:
            items, health = self._fetcher.fetch_feed(feed, now=now)
            healths.append(health)
            articles.extend(items)

        self._archive.append(articles, fetched_at=now)

        publisher_count = (
            count_publisher_families([a.effective_publisher for a in articles])
            if articles
            else 0
        )
        feeds_ok = sum(1 for health in healths if health.ok)
        if not articles:
            coverage = CoverageState.UNAVAILABLE
        elif (
            (self._feeds and feeds_ok < len(self._feeds) / 2)
            or (self._worldmonitor_enabled and wm_coverage in ("unavailable", "stale"))
            or wm_coverage == "partial"
        ):
            coverage = CoverageState.PARTIAL
        else:
            coverage = CoverageState.COMPLETE
        self._last_coverage = coverage

        return AcquisitionReport(
            fetched_at=now,
            coverage=coverage,
            article_count=len(articles),
            publisher_count=publisher_count,
            feed_healths=tuple(healths),
            worldmonitor_coverage=wm_coverage,
            notes=tuple(notes),
        )

    def stories(
        self,
        as_of: datetime | None = None,
        *,
        coverage: CoverageState | None = None,
    ) -> list[NewsStory]:
        """Cluster and score everything visible at ``as_of``. Pure read —
        safe for both live use and backtest replay."""

        as_of = as_of or datetime.now(UTC)
        visible = self._archive.load_visible(as_of, max_age_hours=self._max_age_hours)
        visible = [
            article
            for article in visible
            if is_within_freshness_floor(article, now=as_of, max_age_hours=self._max_age_hours)
        ]
        if not visible:
            return []
        effective_coverage = coverage or self._last_coverage
        if effective_coverage is CoverageState.UNAVAILABLE:
            # We ARE serving stories from the archive, so the honest label
            # for unknown/failed acquisition is stale, not unavailable.
            effective_coverage = CoverageState.STALE
        stories = [
            build_story(members, as_of, effective_coverage)
            for members in cluster_articles(visible)
        ]
        stories.sort(
            key=lambda s: (-s.importance_score, -s.primary.effective_published_ms())
        )
        return stories
