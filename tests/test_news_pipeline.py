"""Pipeline: archive no-lookahead, clustering with ticker guard, phases, signals."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ai_trader.config import AppSettings
from ai_trader.domain.signals import SignalBundle, SignalDirection
from ai_trader.news.archive import NewsArchive
from ai_trader.news.models import CoverageState, ObservedArticle, StoryPhase
from ai_trader.news.pipeline import (
    NewsIntelligenceEngine,
    build_story,
    cluster_articles,
    derive_phase,
)
from ai_trader.news.signals import (
    NewsLookaheadError,
    build_news_signal,
    build_news_signals,
)

T0 = datetime(2026, 8, 26, 9, tzinfo=UTC)


def article(i, title, source="Reuters Business", tickers=(), seen=T0, published=None, **kw):
    published = published or seen
    return ObservedArticle(
        article_id=f"a{i}",
        title=title,
        link=f"http://x/{i}",
        source_name=source,
        published_at=published,
        first_seen_at=seen,
        last_seen_at=kw.pop("last_seen", seen),
        tickers=tuple(tickers),
        **kw,
    )


def engine_with_archive(tmp_path: Path) -> NewsIntelligenceEngine:
    return NewsIntelligenceEngine(
        settings=AppSettings(),
        archive=NewsArchive(tmp_path / "arch.jsonl"),
        worldmonitor_enabled=False,
        feeds=(),
    )


def test_cluster_articles_merges_variants_and_splits_disjoint_tickers():
    articles = [
        article(1, "Acme Corp beats Q3 earnings expectations", tickers=("ACME",)),
        article(2, "Acme Corp tops Q3 earnings expectations - Reuters", tickers=("ACME",)),
        article(3, "Zeta Corp beats Q3 earnings expectations", tickers=("ZETA",)),
    ]
    clusters = cluster_articles(articles)
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2]
    for cluster in clusters:
        tickers = {t for member in cluster for t in member.tickers}
        assert len(tickers) == 1  # never a merged ACME/ZETA cluster


def test_story_corroboration_counts_publishers_not_feeds():
    members = [
        article(1, "Acme cuts guidance for the year", source="Reuters Business"),
        article(2, "Acme cuts full-year guidance", source="Reuters World"),
        article(3, "Acme slashes guidance - CNBC", source="CNBC Top News"),
    ]
    story = build_story(members, T0 + timedelta(hours=1), CoverageState.COMPLETE)
    assert story.publisher_count == 2
    assert story.classification.category == "guidance_cut"


def test_story_id_anchors_on_earliest_observed_member():
    early = article(1, "Acme cuts guidance", seen=T0)
    late = article(2, "Acme slashes its guidance outlook", seen=T0 + timedelta(hours=1))
    story_ab = build_story([early, late], T0 + timedelta(hours=2), CoverageState.COMPLETE)
    story_ba = build_story([late, early], T0 + timedelta(hours=2), CoverageState.COMPLETE)
    assert story_ab.story_id == story_ba.story_id  # order-independent
    solo = build_story([early], T0 + timedelta(hours=2), CoverageState.COMPLETE)
    assert story_ab.story_id == solo.story_id  # late joiner cannot steal the id


def test_phase_derivation():
    kw = dict(first_seen_at=T0, last_seen_at=T0)
    half_hour = T0 + timedelta(minutes=30)
    one_hour = T0 + timedelta(hours=1)
    assert derive_phase(mention_count=1, as_of=half_hour, **kw) is StoryPhase.BREAKING
    assert derive_phase(mention_count=3, as_of=one_hour, **kw) is StoryPhase.DEVELOPING
    assert derive_phase(mention_count=9, as_of=one_hour, **kw) is StoryPhase.SUSTAINED
    silent = T0 + timedelta(hours=30)
    assert derive_phase(mention_count=9, as_of=silent, **kw) is StoryPhase.FADING


def test_engine_replay_is_no_lookahead(tmp_path):
    engine = engine_with_archive(tmp_path)
    early = article(1, "Acme faces SEC probe over disclosures", tickers=("ACME",), seen=T0)
    late = article(
        2,
        "Acme probe widens as DOJ opens investigation",
        tickers=("ACME",),
        seen=T0 + timedelta(hours=4),
        published=T0 - timedelta(hours=1),  # backdated pubDate must not matter
    )
    engine.archive.append([early], fetched_at=T0)
    engine.archive.append([late], fetched_at=T0 + timedelta(hours=4))

    mid = engine.stories(T0 + timedelta(hours=1))
    assert len(mid) == 1
    assert len(mid[0].articles) == 1  # the later sighting is invisible at T0+1h

    after = engine.stories(T0 + timedelta(hours=5))
    assert sum(len(s.articles) for s in after) == 2


def test_signal_effective_date_is_first_seen_not_pubdate(tmp_path):
    engine = engine_with_archive(tmp_path)
    backdated = article(
        1,
        "Acme files for Chapter 11 bankruptcy protection",
        tickers=("ACME",),
        seen=T0,
        published=T0 - timedelta(days=10),  # publisher claims it's old news
    )
    engine.archive.append([backdated], fetched_at=T0)
    as_of = T0 + timedelta(hours=1)
    # published_at 10 days back is outside nothing (96h floor uses first_seen
    # for undated only) — but the pubDate IS parseable and old, so the
    # freshness floor drops it. Re-stamp within the window to test the date rule.
    recent = article(
        2,
        "Beta Corp files for Chapter 11 bankruptcy protection",
        tickers=("BETA",),
        seen=T0,
        published=T0 - timedelta(hours=20),
    )
    engine.archive.append([recent], fetched_at=T0)
    stories = engine.stories(as_of)
    signals = build_news_signals(stories, "BETA", as_of)
    assert len(signals) == 1
    signal = signals[0]
    assert signal.effective_date == T0.date()  # observation date, not pubDate
    assert signal.direction is SignalDirection.SHORT
    assert signal.name == "news.bankruptcy"

    bundle = SignalBundle(ticker="BETA", as_of=as_of.date(), signals=signals)
    assert bundle.direction is SignalDirection.SHORT


def test_lookahead_story_refuses_to_build_signal():
    story_article = article(1, "Acme beats estimates", tickers=("ACME",), seen=T0)
    story = build_story([story_article], T0, CoverageState.COMPLETE)
    with pytest.raises(NewsLookaheadError):
        build_news_signal(story, "ACME", T0 - timedelta(hours=1))


def test_degraded_coverage_caps_confidence():
    member = article(1, "Acme cuts guidance for the year", tickers=("ACME",), seen=T0)
    full = build_story([member], T0, CoverageState.COMPLETE)
    stale = build_story([member], T0, CoverageState.STALE)
    as_of = T0 + timedelta(minutes=5)
    assert (
        build_news_signal(stale, "ACME", as_of).confidence
        < build_news_signal(full, "ACME", as_of).confidence
    )


def test_low_severity_stories_do_not_become_signals(tmp_path):
    engine = engine_with_archive(tmp_path)
    engine.archive.append(
        [article(1, "Quiet Tuesday for Acme Corp watchers", tickers=("ACME",), seen=T0)],
        fetched_at=T0,
    )
    stories = engine.stories(T0 + timedelta(hours=1))
    assert build_news_signals(stories, "ACME", T0 + timedelta(hours=1)) == ()


def test_collect_never_raises_and_reports_unavailable(tmp_path):
    # No worldmonitor, no feeds → an empty but honest report.
    engine = engine_with_archive(tmp_path)
    report = engine.collect(now=T0)
    assert report.coverage is CoverageState.UNAVAILABLE
    assert report.article_count == 0
