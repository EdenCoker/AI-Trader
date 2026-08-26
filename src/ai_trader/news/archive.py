"""Append-only sighting archive — the no-lookahead backbone.

Every collect() pass appends one JSON line per sighted article, stamped
with the fetch time. ``first_seen_at`` for an article is therefore an
OBSERVED fact of this system, not a publisher claim, and it is the only
timestamp allowed to become a ``Signal.effective_date``.

Replay contract for backtesting: ``load_visible(as_of)`` returns only
sightings whose fetch time is <= ``as_of``. Re-running a backtest against
a grown archive returns exactly the articles that were visible at
``as_of`` when they were recorded — the same guarantee the repo's
disclosure-date and filing-date tests enforce for smart-money data.

Repeated sightings of the same article across passes are folded at read
time into one ObservedArticle with earliest first_seen, latest last_seen,
and a sighting_count — which is the story-velocity input for the
psychology layer.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ai_trader.news.models import ObservedArticle

log = logging.getLogger(__name__)


def _sighting_key(article: ObservedArticle) -> str:
    return article.link or f"{article.source_name}|{article.title}"


class NewsArchive:
    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def append(self, articles: list[ObservedArticle], fetched_at: datetime) -> int:
        """Record one sighting per article at ``fetched_at``. Within a
        single pass, duplicate links are collapsed to one sighting."""

        if not articles:
            return 0
        # Normalize to UTC so archive rows compare chronologically no
        # matter what timezone the caller handed in.
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=UTC)
        else:
            fetched_at = fetched_at.astimezone(UTC)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        seen: set[str] = set()
        written = 0
        with self._path.open("a", encoding="utf-8") as handle:
            for article in articles:
                key = _sighting_key(article)
                if key in seen:
                    continue
                seen.add(key)
                row = article.model_dump(mode="json")
                row["_fetched_at"] = fetched_at.isoformat()
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
        return written

    def load_visible(
        self,
        as_of: datetime,
        *,
        max_age_hours: int = 96,
    ) -> list[ObservedArticle]:
        """Articles visible at ``as_of``, folded across sightings.

        - sightings fetched after ``as_of`` are excluded (no lookahead)
        - articles whose newest visible sighting is older than
          ``max_age_hours`` before ``as_of`` are excluded (freshness floor)
        """

        if not self._path.exists():
            return []

        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=UTC)
        floor = as_of - timedelta(hours=max_age_hours)
        folded: dict[str, dict] = {}
        seen_times: dict[str, tuple[datetime, datetime]] = {}
        with self._path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    fetched_at = datetime.fromisoformat(row.pop("_fetched_at"))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    log.warning("skipping malformed archive row")
                    continue
                if fetched_at.tzinfo is None:
                    fetched_at = fetched_at.replace(tzinfo=UTC)
                if fetched_at > as_of:
                    continue
                key = row.get("link") or f"{row.get('source_name')}|{row.get('title')}"
                existing = folded.get(key)
                if existing is None:
                    row["sighting_count"] = 1
                    folded[key] = row
                    seen_times[key] = (fetched_at, fetched_at)
                else:
                    existing["sighting_count"] += 1
                    first, last = seen_times[key]
                    seen_times[key] = (min(first, fetched_at), max(last, fetched_at))
        for key, row in folded.items():
            first, last = seen_times[key]
            row["first_seen_at"] = first.isoformat()
            row["last_seen_at"] = last.isoformat()

        articles: list[ObservedArticle] = []
        for row in folded.values():
            try:
                article = ObservedArticle.model_validate(row)
            except Exception:  # noqa: BLE001 — schema drift skips a row, not the load
                log.warning("skipping invalid archive row")
                continue
            if article.last_seen_at < floor:
                continue
            articles.append(article)
        articles.sort(key=lambda a: (a.first_seen_at, a.article_id))
        return articles

    def prune(self, *, keep_days: int = 30, now: datetime | None = None) -> int:
        """Compact the archive to sightings newer than ``keep_days``.
        Returns the number of rows kept."""

        if not self._path.exists():
            return 0
        now = now or datetime.now(UTC)
        cutoff = now - timedelta(days=keep_days)
        kept: list[str] = []
        with self._path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    fetched_at = datetime.fromisoformat(json.loads(stripped)["_fetched_at"])
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
                if fetched_at >= cutoff:
                    kept.append(stripped)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        tmp.replace(self._path)
        return len(kept)
