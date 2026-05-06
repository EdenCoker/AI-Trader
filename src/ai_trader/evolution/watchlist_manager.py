from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class TickerCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    score: float = 0.0
    sources: tuple[str, ...] = ()
    reason: str | None = None
    metadata: dict = Field(default_factory=dict)

    @property
    def normalized_ticker(self) -> str:
        return normalize_ticker(self.ticker)


class WatchlistChange(BaseModel):
    model_config = ConfigDict(frozen=True)

    added: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    demoted: tuple[str, ...] = ()
    total_active: int = 0


class WatchlistManager:
    def __init__(
        self,
        *,
        watchlist_path: Path = Path("data/watchlist.txt"),
        exclusion_path: Path = Path("data/watchlist_exclusions.txt"),
        cold_storage_path: Path = Path("data/cold_storage.txt"),
        max_size: int = 500,
    ) -> None:
        self.watchlist_path = watchlist_path
        self.exclusion_path = exclusion_path
        self.cold_storage_path = cold_storage_path
        self.max_size = max_size

    def load(self) -> list[str]:
        return _load_ticker_lines(self.watchlist_path)

    def exclusions(self) -> set[str]:
        return set(_load_ticker_lines(self.exclusion_path))

    def cold_storage(self) -> set[str]:
        return set(_load_ticker_lines(self.cold_storage_path))

    def add_candidates(
        self,
        candidates: list[TickerCandidate],
        *,
        max_adds: int,
    ) -> WatchlistChange:
        current = self.load()
        existing = set(current)
        exclusions = self.exclusions()
        cold = self.cold_storage()
        skipped: list[str] = []
        added: list[str] = []
        ranked = sorted(
            candidates,
            key=lambda candidate: (candidate.score, candidate.normalized_ticker),
            reverse=True,
        )
        for candidate in ranked:
            ticker = candidate.normalized_ticker
            if not ticker:
                continue
            if ticker in existing or ticker in exclusions or ticker in cold:
                skipped.append(ticker)
                continue
            if len(added) >= max_adds or len(current) + len(added) >= self.max_size:
                skipped.append(ticker)
                continue
            added.append(ticker)
            existing.add(ticker)

        if added:
            self._write_watchlist([*current, *added])
        return WatchlistChange(
            added=tuple(added),
            skipped=tuple(skipped),
            total_active=len(current) + len(added),
        )

    def demote_cold(
        self,
        activity_by_ticker: dict[str, int],
        *,
        max_idle_weeks: int = 6,
    ) -> WatchlistChange:
        current = self.load()
        demoted = [
            ticker
            for ticker in current
            if activity_by_ticker.get(ticker, 0) >= max_idle_weeks
        ]
        if not demoted:
            return WatchlistChange(total_active=len(current))

        remaining = [ticker for ticker in current if ticker not in set(demoted)]
        cold = sorted(set(self.cold_storage()).union(demoted))
        self._write_watchlist(remaining)
        self._atomic_write_lines(self.cold_storage_path, cold)
        return WatchlistChange(
            demoted=tuple(demoted),
            total_active=len(remaining),
        )

    def append_audit(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"created_at": datetime.now(UTC).isoformat(), **payload}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _write_watchlist(self, tickers: list[str]) -> None:
        normalized: list[str] = []
        seen: set[str] = set()
        for ticker in tickers:
            value = normalize_ticker(ticker)
            if value and value not in seen:
                seen.add(value)
                normalized.append(value)
        self._atomic_write_lines(self.watchlist_path, normalized[: self.max_size])

    @staticmethod
    def _atomic_write_lines(path: Path, values: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")
        tmp.replace(path)


def normalize_ticker(value: str) -> str:
    ticker = str(value).strip().upper()
    if not ticker or ticker.startswith("#"):
        return ""
    allowed = []
    for char in ticker:
        if char.isalnum() or char in {".", "-"}:
            allowed.append(char)
    return "".join(allowed)


def _load_ticker_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    values: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        ticker = normalize_ticker(line)
        if ticker and ticker not in seen:
            seen.add(ticker)
            values.append(ticker)
    return values
