"""Curated finance feed registry.

Every URL here was verified working at curation time; dead feeds are the
silent failure mode of RSS pipelines (Reuters killed public RSS years ago
and `feeds.reuters.com` now resolves to nothing — the old training-data
script carried it for years because a bare `except` swallowed the error).

Publishers without a working native feed are reached through Google News
site-search RSS. Those feeds stamp the ORIGINATING outlet per item in the
RSS ``<source>`` element, which the fetcher lifts into
``origin_publisher`` so corroboration counts the real newsroom, not
"Google News".
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class FeedSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    url: str
    # Feed label used when the item itself doesn't name a publisher.
    is_aggregator: bool = False


def _google_news(query: str) -> str:
    from urllib.parse import quote_plus

    return (
        "https://news.google.com/rss/search?q="
        + quote_plus(query)
        + "&hl=en-US&gl=US&ceid=US:en"
    )


FINANCE_FEEDS: tuple[FeedSpec, ...] = (
    # Native feeds — direct from the publisher.
    FeedSpec(name="CNBC Top News", url="https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    FeedSpec(
        name="MarketWatch Top Stories",
        url="https://feeds.content.dowjones.io/public/rss/mw_topstories",
    ),
    FeedSpec(
        name="WSJ Markets",
        url="https://feeds.content.dowjones.io/public/rss/RSSMarketsMain",
    ),
    FeedSpec(
        name="WSJ Business",
        url="https://feeds.content.dowjones.io/public/rss/WSJcomUSBusiness",
    ),
    FeedSpec(name="BBC Business", url="https://feeds.bbci.co.uk/news/business/rss.xml"),
    FeedSpec(name="Guardian Business", url="https://www.theguardian.com/uk/business/rss"),
    FeedSpec(
        name="NYT Business",
        url="https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
    ),
    FeedSpec(name="Yahoo Finance", url="https://finance.yahoo.com/news/rssindex"),
    FeedSpec(
        name="Seeking Alpha Market Currents",
        url="https://seekingalpha.com/market_currents.xml",
    ),
    FeedSpec(
        name="PR Newswire Releases",
        url="https://www.prnewswire.com/rss/news-releases-list.rss",
    ),
    FeedSpec(
        name="GlobeNewswire Public Companies",
        url=(
            "https://www.globenewswire.com/RssFeed/orgclass/1/feedTitle/"
            "GlobeNewswire%20-%20News%20about%20Public%20Companies"
        ),
    ),
    FeedSpec(
        name="Federal Reserve Press",
        url="https://www.federalreserve.gov/feeds/press_all.xml",
    ),
    FeedSpec(name="SEC Press Releases", url="https://www.sec.gov/news/pressreleases.rss"),
    # Google News site-search fallbacks for publishers without public RSS.
    FeedSpec(
        name="Reuters Business (GNews)",
        url=_google_news("site:reuters.com business OR markets when:1d"),
        is_aggregator=True,
    ),
    FeedSpec(
        name="AP Business (GNews)",
        url=_google_news("site:apnews.com business when:1d"),
        is_aggregator=True,
    ),
    FeedSpec(
        name="FT (GNews)",
        url=_google_news("site:ft.com when:1d"),
        is_aggregator=True,
    ),
    FeedSpec(
        name="Bloomberg (GNews)",
        url=_google_news("site:bloomberg.com markets when:1d"),
        is_aggregator=True,
    ),
)


def training_corpus_feed_urls() -> list[str]:
    """Working feed URLs for the training-data ingestion script."""

    return [feed.url for feed in FINANCE_FEEDS]
