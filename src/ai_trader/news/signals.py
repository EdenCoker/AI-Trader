"""Turn scored news stories into ``Signal`` objects for the fusion layer.

This is where the news pipeline meets the trading algorithm, and the
mapping is deliberate:

- ``strength``    ← importance_score / 100 — "how market-moving"
- ``confidence``  ← credibility_score / 100 × coverage multiplier —
                    "how much to trust it", which is what
                    ``SignalBundle.combined_strength`` weights by
- ``direction``   ← event polarity (long / short / neutral)
- ``effective_date`` ← the story's ``first_seen_at`` date — the moment
                    THIS system observed it, never the publisher's
                    claimed pubDate. Same rule as disclosure_date for
                    congressional trades and filing_date for 13F.

The ``SignalBundle`` validator rejects any signal effective after the
bundle's ``as_of``; ``build_news_signals`` additionally refuses to build
from a story first seen after ``as_of`` so the failure is loud at the
source instead of at bundle time.
"""

from __future__ import annotations

from datetime import datetime

from ai_trader.domain.signals import Signal, SignalDirection
from ai_trader.news.models import NewsStory

# Event horizon: how long the information plausibly stays tradeable.
_CATEGORY_HORIZONS: dict[str, int] = {
    "bankruptcy": 60,
    "fraud": 60,
    "going_concern": 60,
    "default": 45,
    "delisting": 30,
    "mna": 45,
    "investigation": 30,
    "lawsuit": 30,
    "short_report": 21,
    "guidance_cut": 21,
    "guidance_raise": 21,
    "fda_approval": 30,
    "fda_rejection": 30,
    "recall": 21,
    "data_breach": 14,
    "ceo_exit": 14,
    "credit_downgrade": 21,
    "activist_stake": 30,
    "earnings_beat": 10,
    "earnings_miss": 10,
    "analyst_upgrade": 7,
    "analyst_downgrade": 7,
    "dividend_cut": 14,
    "buyback": 14,
    "layoffs": 14,
    "contract_win": 14,
    "macro_rates": 5,
    "macro_inflation": 5,
}
_DEFAULT_HORIZON = 14

# Confidence floor below which a story is noise, not signal.
MIN_SIGNAL_SEVERITY = 25


class NewsLookaheadError(ValueError):
    """A story first observed after ``as_of`` was offered as a signal."""


def _direction(polarity: int) -> SignalDirection:
    if polarity > 0:
        return SignalDirection.LONG
    if polarity < 0:
        return SignalDirection.SHORT
    return SignalDirection.NEUTRAL


def build_news_signal(
    story: NewsStory,
    ticker: str,
    as_of: datetime,
) -> Signal:
    """Build one Signal from one story for one ticker."""

    if story.first_seen_at > as_of:
        raise NewsLookaheadError(
            f"story {story.story_id} first seen {story.first_seen_at.isoformat()} "
            f"is after as_of {as_of.isoformat()}"
        )
    ticker = ticker.upper()
    classification = story.classification
    strength = max(0.0, min(1.0, story.importance_score / 100.0))
    confidence = max(
        0.0,
        min(
            1.0,
            (story.credibility_score / 100.0) * story.coverage.confidence_multiplier,
        ),
    )
    headlines = tuple(article.title for article in story.articles[:3])
    reasons = (
        f"News event: {classification.category} "
        f"(severity {classification.severity}, {story.publisher_count} publisher(s))",
        *headlines,
    )
    return Signal(
        name=f"news.{classification.category}",
        ticker=ticker,
        source="news_pipeline",
        direction=_direction(classification.polarity),
        strength=strength,
        confidence=confidence,
        effective_date=story.first_seen_at.date(),
        horizon_days=_CATEGORY_HORIZONS.get(classification.category, _DEFAULT_HORIZON),
        invalidation=(
            "Story fades (no new sightings for 24h) or a contradicting story "
            "with higher credibility emerges."
        ),
        reasons=reasons,
        metadata={
            "story_id": story.story_id,
            "phase": story.phase.value,
            "publisher_count": story.publisher_count,
            "mention_count": story.mention_count,
            "importance_score": story.importance_score,
            "credibility_score": story.credibility_score,
            "coverage": story.coverage.value,
            "polarity_note": (
                "mna polarity is scored for the acquisition TARGET; invert for acquirers"
                if classification.category == "mna"
                else None
            ),
            "first_seen_at": story.first_seen_at.isoformat(),
            "historical": classification.historical,
        },
    )


def build_news_signals(
    stories: list[NewsStory],
    ticker: str,
    as_of: datetime,
    *,
    min_severity: int = MIN_SIGNAL_SEVERITY,
    limit: int = 5,
) -> tuple[Signal, ...]:
    """Signals for every qualifying story that names ``ticker``.

    Stories are already importance-sorted by the engine; the limit keeps a
    news-storm from flooding the bundle (fusion weights by confidence, but
    twenty near-duplicate signals would still drown smart-money inputs).
    """

    ticker = ticker.upper()
    signals: list[Signal] = []
    for story in stories:
        if len(signals) >= limit:
            break
        if ticker not in story.tickers:
            continue
        if story.classification.severity < min_severity:
            continue
        if story.first_seen_at > as_of:
            # Defensive: the archive replay already filters this; a story
            # slipping through means a bug upstream, so fail loud.
            raise NewsLookaheadError(
                f"story {story.story_id} first seen after as_of"
            )
        signals.append(build_news_signal(story, ticker, as_of))
    return tuple(signals)
