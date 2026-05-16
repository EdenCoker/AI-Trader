from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd


class PriceCache:
    """Range-aware local cache for per-ticker price DataFrames."""

    def __init__(self, root: Path, *, enabled: bool = True) -> None:
        self.root = root
        self.enabled = enabled

    def get(self, ticker: str, start: date, end: date) -> pd.DataFrame | None:
        if not self.enabled:
            return None
        ticker_dir = self._ticker_dir(ticker)
        if not ticker_dir.exists():
            return None
        candidates: list[tuple[date, date, Path]] = []
        for path in ticker_dir.glob("*.pkl"):
            bounds = self._parse_bounds(path)
            if bounds is None:
                continue
            cached_start, cached_end = bounds
            if cached_start <= start and cached_end >= end:
                candidates.append((cached_start, cached_end, path))
        if not candidates:
            return None

        cached_start, cached_end, path = sorted(
            candidates,
            key=lambda item: ((item[1] - item[0]).days, item[0]),
        )[0]
        frame = pd.read_pickle(path)
        if frame.empty:
            return frame
        frame = _normalize_date_index(frame)
        return frame[(frame.index >= start) & (frame.index <= end)].copy()

    def put(self, ticker: str, start: date, end: date, frame: pd.DataFrame) -> Path | None:
        if not self.enabled or frame is None:
            return None
        ticker_dir = self._ticker_dir(ticker)
        ticker_dir.mkdir(parents=True, exist_ok=True)
        path = ticker_dir / f"{start:%Y%m%d}_{end:%Y%m%d}.pkl"
        _normalize_date_index(frame).to_pickle(path)
        return path

    def _ticker_dir(self, ticker: str) -> Path:
        safe = re.sub(r"[^A-Z0-9_.-]+", "_", ticker.upper())
        return self.root / "polygon_prices" / safe

    @staticmethod
    def _parse_bounds(path: Path) -> tuple[date, date] | None:
        match = re.fullmatch(r"(\d{8})_(\d{8})\.pkl", path.name)
        if match is None:
            return None
        return (
            date.fromisoformat(_compact_to_iso(match.group(1))),
            date.fromisoformat(_compact_to_iso(match.group(2))),
        )


def _compact_to_iso(value: str) -> str:
    return f"{value[:4]}-{value[4:6]}-{value[6:]}"


def _normalize_date_index(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
        out = out.dropna(subset=["date"]).set_index("date")
    elif not out.empty:
        out.index = pd.to_datetime(out.index, errors="coerce").date
    return out.sort_index()

