"""Resilient RSS/Atom acquisition.

Consistency guarantees, in order of importance:

- ``fetch_feed`` NEVER raises. A feed failure degrades to cached items or
  an empty result with an honest :class:`FeedHealth` record; one flaky
  publisher must not blank the whole pass.
- Circuit breaker: two consecutive failures put a feed on a 5-minute
  cooldown so a dead host doesn't burn the fetch budget every cycle.
- TTL cache: a healthy parse is reused for 30 minutes; publishers that
  rate-limit datacenter IPs see one request per window, not one per call.
- Undated discipline: items without a parseable date are kept but flagged
  ``pub_date_missing`` — they may never claim freshness downstream.
- Browser User-Agent: several finance publishers 403 default library UAs
  from datacenter IPs.
"""

from __future__ import annotations

import email.utils
import hashlib
import logging
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import UTC, datetime

import httpx

from ai_trader.news.feeds import FeedSpec
from ai_trader.news.models import FeedHealth, ObservedArticle
from ai_trader.news.tickers import extract_tickers

log = logging.getLogger(__name__)

FEED_TIMEOUT_S = 8.0
MAX_ITEMS_PER_FEED = 10
MAX_FAILURES_BEFORE_COOLDOWN = 2
COOLDOWN_S = 5 * 60
CACHE_TTL_S = 30 * 60

# Hard cap on a feed body: a legitimate RSS feed is tens of KB; a
# multi-megabyte body is a misconfiguration or an attack on memory
# (response.text + ET.fromstring both materialize it).
MAX_RESPONSE_BYTES = 5 * 1024 * 1024
MAX_TITLE_LEN = 300

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
)
_HEADERS = {
    "User-Agent": BROWSER_UA,
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_DC_NS = "{http://purl.org/dc/elements/1.1/}"


def _clean_text(value: str | None) -> str:
    if not value:
        return ""
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", value)).strip()


def parse_feed_datetime(raw: str) -> datetime | None:
    """RFC 2822 (RSS) or ISO 8601 (Atom / dc:date). None when unparseable
    — the caller records ``pub_date_missing`` instead of inventing a date."""

    raw = raw.strip()
    if not raw:
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(raw)
        if parsed is not None:
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except (TypeError, ValueError):
        pass
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except ValueError:
        return None


def _article_id(source: str, link: str, title: str) -> str:
    digest = hashlib.sha256(f"{source}|{link}|{title}".encode()).hexdigest()
    return digest[:16]


def parse_feed_xml(
    xml_text: str,
    feed: FeedSpec,
    now: datetime,
) -> tuple[list[ObservedArticle], int]:
    """Parse RSS 2.0 or Atom into observed articles.

    Returns (articles, dropped_undated_count). Undated items are NOT
    dropped here — they are flagged; the count reports items whose date
    string existed but failed to parse, for the coverage ledger.
    """

    root = ET.fromstring(xml_text)
    items = root.findall(".//item")
    is_atom = not items
    if is_atom:
        items = root.findall(f".//{_ATOM_NS}entry")

    articles: list[ObservedArticle] = []
    unparseable_dates = 0
    for item in items[:MAX_ITEMS_PER_FEED]:
        if is_atom:
            title = _clean_text(item.findtext(f"{_ATOM_NS}title"))
            link_el = item.find(f"{_ATOM_NS}link[@href]")
            link = link_el.get("href", "") if link_el is not None else ""
            date_raw = (
                item.findtext(f"{_ATOM_NS}published")
                or item.findtext(f"{_ATOM_NS}updated")
                or ""
            )
            snippet = _clean_text(item.findtext(f"{_ATOM_NS}summary"))
            origin = ""
        else:
            title = _clean_text(item.findtext("title"))
            link = (item.findtext("link") or "").strip()
            # The namespaced lookup covers dc:date; a literal "dc:date"
            # findtext can never match under ElementTree (tags are stored
            # namespace-expanded) and raises on the pure-Python parser.
            date_raw = (
                item.findtext("pubDate")
                or item.findtext(f"{_DC_NS}date")
                or ""
            )
            snippet = _clean_text(item.findtext("description"))
            # Google News stamps the originating outlet in <source>; the
            # aggregator label must not absorb the real publisher.
            origin = _clean_text(item.findtext("source")) if feed.is_aggregator else ""

        if not title:
            continue
        title = title[:MAX_TITLE_LEN]

        published_at = parse_feed_datetime(date_raw)
        missing = published_at is None
        if missing and date_raw:
            unparseable_dates += 1
        # A pubDate from the future (beyond small clock skew) is treated
        # as missing: feeds cannot be allowed to claim tomorrow's
        # freshness today.
        if published_at is not None and (published_at - now).total_seconds() > 3600:
            published_at = None
            missing = True

        snippet = snippet[:500]
        articles.append(
            ObservedArticle(
                article_id=_article_id(feed.name, link, title),
                title=title,
                link=link,
                source_name=feed.name,
                origin_publisher=origin,
                published_at=published_at,
                pub_date_missing=missing,
                first_seen_at=now,
                last_seen_at=now,
                tickers=extract_tickers(f"{title} {snippet}"),
                snippet=snippet,
                provenance="rss",
            )
        )
    return articles, unparseable_dates


class ResilientFeedFetcher:
    """Stateful fetcher with per-feed circuit breaker and TTL cache."""

    def __init__(
        self,
        *,
        timeout_s: float = FEED_TIMEOUT_S,
        transport: Callable[[str], httpx.Response] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._timeout_s = timeout_s
        self._transport = transport
        self._clock = clock
        self._failures: dict[str, int] = {}
        self._cooldown_until: dict[str, float] = {}
        self._cache: dict[str, tuple[float, list[ObservedArticle]]] = {}

    def _get(self, url: str) -> httpx.Response:
        if self._transport is not None:
            return self._transport(url)
        return httpx.get(
            url, headers=_HEADERS, timeout=self._timeout_s, follow_redirects=True
        )

    def on_cooldown(self, feed_name: str) -> bool:
        until = self._cooldown_until.get(feed_name, 0.0)
        if self._clock() < until:
            return True
        if until:
            self._cooldown_until.pop(feed_name, None)
            self._failures.pop(feed_name, None)
        return False

    def _record_failure(self, feed_name: str) -> None:
        count = self._failures.get(feed_name, 0) + 1
        self._failures[feed_name] = count
        if count >= MAX_FAILURES_BEFORE_COOLDOWN:
            self._cooldown_until[feed_name] = self._clock() + COOLDOWN_S
            log.warning("feed %s on cooldown after %d failures", feed_name, count)

    def fetch_feed(self, feed: FeedSpec, now: datetime | None = None) -> tuple[
        list[ObservedArticle], FeedHealth
    ]:
        """Fetch one feed. Never raises."""

        now = now or datetime.now(UTC)
        cached = self._cache.get(feed.name)

        if self.on_cooldown(feed.name):
            items = cached[1] if cached else []
            return items, FeedHealth(
                feed_name=feed.name,
                ok=False,
                item_count=len(items),
                failure="cooldown",
                on_cooldown=True,
                from_cache=True,
                fetched_at=now,
            )

        if cached is not None and self._clock() - cached[0] < CACHE_TTL_S:
            # from_cache marks these as NOT new observations: the caller
            # must not archive them as fresh sightings, or story velocity
            # would measure the polling cadence instead of news flow.
            return cached[1], FeedHealth(
                feed_name=feed.name,
                ok=True,
                item_count=len(cached[1]),
                from_cache=True,
                fetched_at=now,
            )

        try:
            response = self._get(feed.url)
            response.raise_for_status()
            body = response.content
            if len(body) > MAX_RESPONSE_BYTES:
                raise ValueError(
                    f"feed body {len(body)} bytes exceeds cap {MAX_RESPONSE_BYTES}"
                )
            articles, unparseable = parse_feed_xml(response.text, feed, now)
        except Exception as exc:  # noqa: BLE001 — availability beats purity here
            self._record_failure(feed.name)
            items = cached[1] if cached else []
            log.warning("feed %s failed: %s", feed.name, exc)
            return items, FeedHealth(
                feed_name=feed.name,
                ok=False,
                item_count=len(items),
                failure=str(exc)[:200],
                from_cache=bool(items),
                fetched_at=now,
            )

        self._failures.pop(feed.name, None)
        if articles:
            self._cache[feed.name] = (self._clock(), articles)
        return articles, FeedHealth(
            feed_name=feed.name,
            ok=True,
            item_count=len(articles),
            dropped_undated=unparseable,
            fetched_at=now,
        )
