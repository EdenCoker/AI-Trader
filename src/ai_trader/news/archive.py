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


# Minimum spacing between two sightings of the same article that count as
# distinct observations. Without this, a 5-minute cron would record the
# same digest item 288 times a day and story "velocity" would measure the
# polling cadence, not news flow.
RESIGHT_MIN_INTERVAL_S = 25 * 60
_LASTSEEN_RETENTION_S = 24 * 60 * 60


class NewsArchive:
    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def _sidecar(self, suffix: str) -> Path:
        return self._path.with_name(self._path.stem + suffix)

    # -- re-sight suppression (persisted, so cron-spawned processes agree) --
    def register_sightings(
        self, articles: list[ObservedArticle], fetched_at: datetime
    ) -> list[ObservedArticle]:
        """Return the articles that count as NEW sightings at ``fetched_at``
        (first ever, or last counted sighting older than
        RESIGHT_MIN_INTERVAL_S), and update the last-seen sidecar."""

        if not articles:
            return []
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=UTC)
        sidecar = self._sidecar(".lastseen.json")
        last_seen: dict[str, str] = {}
        try:
            loaded = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                last_seen = {k: v for k, v in loaded.items() if isinstance(v, str)}
        except (OSError, json.JSONDecodeError):
            pass

        fresh: list[ObservedArticle] = []
        for article in articles:
            key = _sighting_key(article)
            previous = last_seen.get(key)
            if previous is not None:
                try:
                    previous_dt = datetime.fromisoformat(previous)
                    if previous_dt.tzinfo is None:
                        previous_dt = previous_dt.replace(tzinfo=UTC)
                    if (fetched_at - previous_dt).total_seconds() < RESIGHT_MIN_INTERVAL_S:
                        continue
                except ValueError:
                    pass
            last_seen[key] = fetched_at.isoformat()
            fresh.append(article)

        horizon = fetched_at - timedelta(seconds=_LASTSEEN_RETENTION_S)
        pruned: dict[str, str] = {}
        for key, value in last_seen.items():
            try:
                seen_dt = datetime.fromisoformat(value)
                if seen_dt.tzinfo is None:
                    seen_dt = seen_dt.replace(tzinfo=UTC)
            except ValueError:
                continue
            if seen_dt >= horizon:
                pruned[key] = value
        tmp = sidecar.with_suffix(".tmp")
        try:
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(pruned), encoding="utf-8")
            tmp.replace(sidecar)
        except OSError:
            log.warning("could not persist last-seen sidecar")
        return fresh

    # -- coverage ledger (persisted, so replay never reads process state) --
    def record_coverage(self, state: str, at: datetime) -> None:
        if at.tzinfo is None:
            at = at.replace(tzinfo=UTC)
        ledger = self._sidecar(".coverage.jsonl")
        try:
            ledger.parent.mkdir(parents=True, exist_ok=True)
            with ledger.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"at": at.isoformat(), "state": state}) + "\n")
        except OSError:
            log.warning("could not persist coverage ledger entry")

    def coverage_at(self, as_of: datetime) -> tuple[str, datetime] | None:
        """Latest recorded acquisition coverage at or before ``as_of``."""

        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=UTC)
        ledger = self._sidecar(".coverage.jsonl")
        if not ledger.exists():
            return None
        best: tuple[str, datetime] | None = None
        try:
            with ledger.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        at = datetime.fromisoformat(row["at"])
                        if at.tzinfo is None:
                            at = at.replace(tzinfo=UTC)
                        state = str(row["state"])
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                        continue
                    if at <= as_of and (best is None or at > best[1]):
                        best = (state, at)
        except OSError:
            return None
        return best

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
            # Self-heal a torn tail: a process killed mid-append can leave
            # the file without a trailing newline, and appending onto that
            # fragment would corrupt BOTH rows. One leading newline turns
            # the fragment into a single malformed line the readers skip.
            if handle.tell() > 0:
                with self._path.open("rb") as tail:
                    tail.seek(-1, 2)
                    if tail.read(1) != b"\n":
                        handle.write("\n")
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
                    if not isinstance(row, dict):
                        raise TypeError("archive row is not an object")
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
        dropped_malformed = 0
        with self._path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                    if not isinstance(row, dict):
                        raise TypeError("archive row is not an object")
                    fetched_at = datetime.fromisoformat(row["_fetched_at"])
                    if fetched_at.tzinfo is None:
                        fetched_at = fetched_at.replace(tzinfo=UTC)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    dropped_malformed += 1
                    continue
                if fetched_at >= cutoff:
                    kept.append(stripped)
        if dropped_malformed:
            log.warning("prune dropped %d malformed archive rows", dropped_malformed)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        tmp.replace(self._path)
        return len(kept)
