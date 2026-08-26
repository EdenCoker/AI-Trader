"""Two orthogonal story scores, deliberately not merged.

``importance_score`` answers "how market-moving is this right now?" — it
mixes event severity, source tier, corroboration, and recency. A severe
story from a state wire can score high here.

``credibility_score`` answers "how much should the system trust this
source on this story?" — tier, propaganda risk, and independent
corroboration only. High propaganda risk is hard-capped regardless of
corroboration.

Conflating the two is how a well-corroborated lie gets sized into a
position; keeping them separate lets the signal layer put importance into
``Signal.strength`` and credibility into ``Signal.confidence``, which the
bundle fusion then weights independently.

Weight structure follows the published worldmonitor scoring design
(severity-dominant importance; risk-dominant credibility with a high-risk
cap); implementation and calibration are this repo's own.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ai_trader.news.models import EventClassification, ObservedArticle
from ai_trader.news.publishers import propaganda_risk, source_tier

IMPORTANCE_WEIGHTS = {
    "severity": 0.55,
    "source_tier": 0.20,
    "corroboration": 0.15,
    "recency": 0.10,
}
CREDIBILITY_WEIGHTS = {
    "source_tier": 0.30,
    "propaganda_risk": 0.50,
    "corroboration": 0.20,
}

_TIER_SCORES = {1: 100, 2: 75, 3: 50, 4: 25}
_RISK_SCORES = {"low": 100, "medium": 50, "unknown": 35, "high": 12}
# A high propaganda-risk source cannot exceed this credibility no matter
# how much corroboration it appears to have.
CREDIBILITY_HIGH_RISK_CAP = 40

CORROBORATION_CAP = 5
CORROBORATION_PER_PUBLISHER = 20
RECENCY_WINDOW_MS = 24 * 60 * 60 * 1000.0

# Hard freshness floor: stories older than this are dropped before
# scoring. Recency already zeroes out beyond 24h; this floor is the
# "don't surface week-old news" guard, not a scoring input.
DEFAULT_MAX_AGE_HOURS = 96


def _corroboration_score(publisher_count: int) -> float:
    bounded = min(max(publisher_count, 0), CORROBORATION_CAP)
    return bounded * CORROBORATION_PER_PUBLISHER


def recency_score(article: ObservedArticle, now: datetime | None = None) -> float:
    """0-100 recency. Undated articles score 0 — an item that cannot
    prove its age never claims freshness."""

    published_ms = article.effective_published_ms()
    if published_ms <= 0:
        return 0.0
    now_ms = (now or datetime.now(UTC)).timestamp() * 1000.0
    age_ms = max(0.0, now_ms - published_ms)
    return max(0.0, 1.0 - age_ms / RECENCY_WINDOW_MS) * 100.0


def importance_score(
    classification: EventClassification,
    article: ObservedArticle,
    publisher_count: int,
    now: datetime | None = None,
) -> int:
    tier = source_tier(article.effective_publisher)
    local = round(
        classification.severity * IMPORTANCE_WEIGHTS["severity"]
        + _TIER_SCORES[tier] * IMPORTANCE_WEIGHTS["source_tier"]
        + _corroboration_score(publisher_count) * IMPORTANCE_WEIGHTS["corroboration"]
        + recency_score(article, now) * IMPORTANCE_WEIGHTS["recency"]
    )
    if article.server_importance is not None:
        # Blend with the upstream (worldmonitor) score when present: the
        # server sees corroboration across hundreds of feeds we don't
        # fetch; our local score sees the finance-event taxonomy it
        # doesn't. The mean keeps either side from dominating. The
        # provider clamps at ingestion; the clamp here is defense in
        # depth — an out-of-range value would fail NewsStory validation
        # and poison every read of the archive window.
        server = max(0, min(200, article.server_importance))
        return max(0, round((local + server) / 2))
    return max(0, local)


def credibility_score(
    article: ObservedArticle,
    publisher_count: int,
) -> int:
    publisher = article.effective_publisher
    tier = source_tier(publisher)
    risk = propaganda_risk(publisher)
    local = round(
        _TIER_SCORES[tier] * CREDIBILITY_WEIGHTS["source_tier"]
        + _RISK_SCORES[risk] * CREDIBILITY_WEIGHTS["propaganda_risk"]
        + _corroboration_score(publisher_count) * CREDIBILITY_WEIGHTS["corroboration"]
    )
    local = max(0, min(100, local))
    if article.server_credibility is not None:
        local = round((local + max(0, min(100, article.server_credibility))) / 2)
    if risk == "high":
        return min(local, CREDIBILITY_HIGH_RISK_CAP)
    return local


def is_within_freshness_floor(
    article: ObservedArticle,
    now: datetime | None = None,
    max_age_hours: int = DEFAULT_MAX_AGE_HOURS,
) -> bool:
    """Freshness gate. Undated articles pass on their first_seen_at age
    instead: we know when WE saw them even when the publisher won't say
    when it happened."""

    now = now or datetime.now(UTC)
    reference = (
        article.published_at
        if article.published_at is not None and not article.pub_date_missing
        else article.first_seen_at
    )
    age_hours = (now - reference).total_seconds() / 3600.0
    return age_hours <= max_age_hours
