"""Scoring invariants: weights, fail-closed risk, high-risk cap, undated discipline."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ai_trader.news.classify import cap_llm_upgrade, classify_headline
from ai_trader.news.models import EventClassification, ObservedArticle
from ai_trader.news.publishers import (
    count_publisher_families,
    propaganda_risk,
    publisher_family,
    source_tier,
)
from ai_trader.news.scoring import (
    CREDIBILITY_HIGH_RISK_CAP,
    CREDIBILITY_WEIGHTS,
    IMPORTANCE_WEIGHTS,
    credibility_score,
    importance_score,
    is_within_freshness_floor,
    recency_score,
)

NOW = datetime(2026, 8, 26, 12, tzinfo=UTC)


def article(source="Reuters Business", published=NOW, missing=False, **kw):
    return ObservedArticle(
        article_id="a1",
        title="t",
        link="http://x/1",
        source_name=source,
        published_at=published,
        pub_date_missing=missing,
        first_seen_at=NOW,
        last_seen_at=NOW,
        **kw,
    )


def test_weights_sum_to_one():
    assert abs(sum(IMPORTANCE_WEIGHTS.values()) - 1.0) < 1e-9
    assert abs(sum(CREDIBILITY_WEIGHTS.values()) - 1.0) < 1e-9


def test_publisher_families_prevent_self_corroboration():
    assert publisher_family("Reuters World") == publisher_family("Reuters Business")
    assert count_publisher_families(["Reuters World", "Reuters US", "Reuters"]) == 1
    assert count_publisher_families(["Reuters World", "CNBC", "AP News"]) == 3


def test_unknown_source_fails_closed():
    assert source_tier("Random Substack") == 4
    assert propaganda_risk("Random Substack") == "unknown"


def test_high_risk_credibility_cap():
    art = article(source="RT")
    assert credibility_score(art, publisher_count=5) <= CREDIBILITY_HIGH_RISK_CAP


def test_undated_article_never_claims_freshness():
    art = article(missing=True, published=None)
    assert art.effective_published_ms() == 0.0
    assert recency_score(art, now=NOW) == 0.0


def test_backdate_beyond_floor_is_dropped_but_undated_passes_on_first_seen():
    old = article(published=NOW - timedelta(hours=200))
    assert not is_within_freshness_floor(old, now=NOW)
    undated = article(missing=True, published=None)
    assert is_within_freshness_floor(undated, now=NOW)


def test_importance_monotone_in_corroboration():
    cls = EventClassification(
        category="guidance_cut", severity=75, polarity=-1, confidence=0.7
    )
    art = article()
    scores = [importance_score(cls, art, n, now=NOW) for n in (1, 2, 3, 5)]
    assert scores == sorted(scores)


def test_server_scores_blend_not_replace():
    cls = EventClassification(category="general", severity=0, polarity=0, confidence=0.3)
    local_only = article()
    blended = article(server_importance=100)
    assert importance_score(cls, local_only, 1, now=NOW) < importance_score(
        cls, blended, 1, now=NOW
    ) < 100


def test_classifier_severity_and_polarity():
    cases = [
        ("Acme files for Chapter 11 bankruptcy protection", "bankruptcy", 100, -1),
        ("Acme faces SEC probe over revenue recognition", "investigation", 75, -1),
        ("Acme cuts guidance for the full year", "guidance_cut", 75, -1),
        ("MegaCorp to acquire Acme for $5 billion", "mna", 75, 1),
        ("Acme beats estimates with record revenue", "earnings_beat", 50, 1),
        ("Quiet day in the markets", "general", 0, 0),
    ]
    for title, category, severity, polarity in cases:
        result = classify_headline(title)
        assert result.category == category, (title, result)
        assert result.severity == severity
        assert result.polarity == polarity


def test_historical_marker_downgrades():
    live = classify_headline("Acme files for bankruptcy")
    retro = classify_headline("Ten years since Acme filed for bankruptcy: a look back")
    assert retro.historical and retro.severity == live.severity - 50


def test_llm_upgrade_cap_blocks_contamination():
    assert cap_llm_upgrade(0, 100) == 50  # info can never jump to critical
    assert cap_llm_upgrade(25, 100) == 75
    assert cap_llm_upgrade(50, 100) == 100
    assert cap_llm_upgrade(75, 25) == 25  # downgrades pass through
