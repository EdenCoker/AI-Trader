from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import pandas as pd
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_fixed

from ai_trader.config import AppSettings, get_settings
from ai_trader.providers.contracts import ProviderError


class PolygonDataLoader:
    def __init__(
        self,
        *,
        settings: AppSettings | None = None,
        cache_dir: Path | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._cache_dir = cache_dir or self._settings.polygon_cache_dir
        self._client = http_client or httpx.Client(timeout=30)

    def load_ohlcv(self, ticker: str, start: date, end: date, timespan: str = "day") -> pd.DataFrame:
        cache_path = self._cache_path(ticker, start, end, timespan)
        if cache_path.exists():
            return _read_frame(cache_path)
        if self._settings.polygon_api_key is None:
            raise ProviderError("POLYGON_API_KEY is required for PolygonDataLoader cache misses")

        data = self._fetch(ticker=ticker, start=start, end=end, timespan=timespan)
        frame = _polygon_results_to_frame(data)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        _write_frame(frame, cache_path)
        return frame

    def _fetch(self, *, ticker: str, start: date, end: date, timespan: str) -> dict:
        api_key = self._settings.polygon_api_key
        assert api_key is not None
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{ticker.upper()}/range/1/"
            f"{timespan}/{start.isoformat()}/{end.isoformat()}"
        )
        params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key.get_secret_value()}
        for attempt in Retrying(
            stop=stop_after_attempt(5),
            wait=wait_fixed(max(12, int(60 / max(self._settings.polygon_rate_limit_rpm, 1)))),
            retry=retry_if_exception_type(httpx.HTTPError),
            reraise=True,
        ):
            with attempt:
                response = self._client.get(url, params=params)
                response.raise_for_status()
                return response.json()
        raise ProviderError("Polygon request failed")

    def _cache_path(self, ticker: str, start: date, end: date, timespan: str) -> Path:
        return self._cache_dir / ticker.upper() / timespan / f"{start.isoformat()}_{end.isoformat()}.parquet"


def _polygon_results_to_frame(data: dict) -> pd.DataFrame:
    results = data.get("results", [])
    rows = []
    for item in results:
        rows.append(
            {
                "date": pd.to_datetime(item["t"], unit="ms", utc=True).date(),
                "open": float(item["o"]),
                "high": float(item["h"]),
                "low": float(item["l"]),
                "close": float(item["c"]),
                "volume": float(item.get("v", 0)),
                "vwap": float(item.get("vw", item["c"])),
            }
        )
    return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "vwap"])


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    try:
        frame.to_parquet(path, index=False)
    except Exception:
        frame.to_pickle(path)


def _read_frame(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_pickle(path)

