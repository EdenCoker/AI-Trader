from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Sequence

import httpx

from ai_trader.config import AppSettings, get_settings
from ai_trader.domain.events import MarketTick
from ai_trader.providers.contracts import ProviderError


class PolygonProvider:
    def __init__(self, *, settings: AppSettings | None = None, http_client: httpx.Client | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = http_client or httpx.Client(timeout=30)

    def fetch_ticks(self, ticker: str, start: datetime, end: datetime) -> Sequence[MarketTick]:
        if self._settings.polygon_api_key is None:
            raise ProviderError("POLYGON_API_KEY is required")
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{ticker.upper()}/range/1/minute/"
            f"{start.date().isoformat()}/{end.date().isoformat()}"
        )
        response = self._client.get(
            url,
            params={
                "adjusted": "true",
                "sort": "asc",
                "limit": 50000,
                "apiKey": self._settings.polygon_api_key.get_secret_value(),
            },
        )
        response.raise_for_status()
        ticks = []
        for item in response.json().get("results", []):
            timestamp = datetime.fromtimestamp(item["t"] / 1000, tz=timezone.utc)
            if start <= timestamp <= end:
                ticks.append(
                    MarketTick(
                        ticker=ticker.upper(),
                        timestamp=timestamp,
                        price=Decimal(str(item["c"])),
                        size=Decimal(str(item.get("v", 0))),
                    )
                )
        return tuple(ticks)

