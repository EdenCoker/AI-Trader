from __future__ import annotations

from datetime import UTC, datetime, timedelta
from urllib.parse import unquote

import httpx

from ai_trader.config import AppSettings
from ai_trader.providers.fear_greed import LiveFearGreedProvider, classify_fear_greed


def _chart_payload(symbol: str, *, interval: str, now: datetime) -> dict:
    base = {
        "SPY": 430.0,
        "QQQ": 380.0,
        "IWM": 205.0,
        "RSP": 165.0,
        "HYG": 78.0,
        "LQD": 108.0,
        "TLT": 96.0,
        "GLD": 212.0,
        "UUP": 29.0,
        "^VIX": 18.0,
    }[symbol]
    if interval == "1d":
        count = 180
        timestamps = [int((now - timedelta(days=count - idx)).timestamp()) for idx in range(count)]
        closes = [base * (0.92 + idx / count * 0.08) for idx in range(count)]
    else:
        count = 3
        timestamps = [
            int((now - timedelta(minutes=10 - idx * 5)).timestamp())
            for idx in range(count)
        ]
        closes = [base * 0.998, base, base * 1.002]
    return {
        "chart": {
            "result": [
                {
                    "timestamp": timestamps,
                    "indicators": {"quote": [{"close": closes}]},
                }
            ],
            "error": None,
        }
    }


def test_live_fear_greed_snapshot_uses_components_and_persists(tmp_path):
    now = datetime(2026, 5, 13, 14, 30, tzinfo=UTC)

    def handler(request: httpx.Request) -> httpx.Response:
        if "query1.finance.yahoo.com" in request.url.host:
            symbol = unquote(request.url.path.rsplit("/", 1)[-1])
            interval = request.url.params["interval"]
            return httpx.Response(200, json=_chart_payload(symbol, interval=interval, now=now))
        if "cboe.com" in request.url.host:
            html = "<table><tr><td>TOTAL PUT/CALL RATIO</td><td>0.82</td></tr></table>"
            return httpx.Response(200, text=html)
        raise AssertionError(f"unexpected URL: {request.url}")

    settings = AppSettings(
        fear_greed_component_max_age_minutes=10_000,
        fear_greed_min_components=4,
        fear_greed_snapshot_path=tmp_path / "fear_greed.jsonl",
    )
    provider = LiveFearGreedProvider(
        settings=settings,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    snapshot = provider.fetch_snapshot(now=now)

    assert 0 <= snapshot.value <= 100
    assert snapshot.label == classify_fear_greed(snapshot.value)
    assert snapshot.confidence > 0
    assert snapshot.fresh_component_count == len(snapshot.components)
    assert {"market_momentum", "volatility", "put_call_options"}.issubset(
        {component.name for component in snapshot.components}
    )

    provider.append_snapshot(snapshot)
    loaded = provider.load_latest_snapshot(now=now)

    assert loaded is not None
    assert loaded.value == snapshot.value
