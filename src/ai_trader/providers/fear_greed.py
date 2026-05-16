from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import fmean
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_trader.config import AppSettings, get_settings
from ai_trader.providers.contracts import ProviderError

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
CBOE_DAILY_STATS_URL = "https://www.cboe.com/markets/us/options/market-statistics/daily"
ALPHA_VANTAGE_QUERY_URL = "https://www.alphavantage.co/query"


class FearGreedComponent(BaseModel):
    """One bounded 0-100 sentiment component used by the live composite."""

    model_config = ConfigDict(frozen=True)

    name: str
    score: float = Field(ge=0, le=100)
    raw_value: float | None = None
    source: str
    observed_at: datetime
    available_at: datetime
    confidence: float = Field(ge=0, le=1)
    stale: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class FearGreedSnapshot(BaseModel):
    """Live market fear/greed snapshot with explicit freshness/provenance."""

    model_config = ConfigDict(frozen=True)

    value: float = Field(ge=0, le=100)
    label: str
    source: str = "live_composite"
    observed_at: datetime
    available_at: datetime
    confidence: float = Field(ge=0, le=1)
    fresh_component_count: int = Field(ge=0)
    stale_component_count: int = Field(ge=0)
    components: tuple[FearGreedComponent, ...]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_component_counts(self) -> FearGreedSnapshot:
        if self.fresh_component_count + self.stale_component_count != len(self.components):
            raise ValueError("component counts must match components")
        return self

    @classmethod
    def from_components(
        cls,
        components: tuple[FearGreedComponent, ...],
        *,
        available_at: datetime,
        expected_component_count: int = 7,
        metadata: dict[str, Any] | None = None,
    ) -> FearGreedSnapshot:
        if not components:
            raise ProviderError("no fear/greed components could be built")

        confidence_sum = sum(component.confidence for component in components)
        if confidence_sum <= 0:
            value = fmean(component.score for component in components)
        else:
            weighted_score = sum(
                component.score * component.confidence for component in components
            )
            value = weighted_score / confidence_sum

        fresh_count = sum(1 for component in components if not component.stale)
        stale_count = len(components) - fresh_count
        component_confidence = confidence_sum / len(components)
        coverage = min(len(components) / max(expected_component_count, 1), 1.0)
        freshness = fresh_count / len(components)
        confidence = component_confidence * (0.55 + 0.45 * coverage) * (0.75 + 0.25 * freshness)

        return cls(
            value=round(_clamp(value, 0, 100), 2),
            label=classify_fear_greed(value),
            observed_at=max(component.observed_at for component in components),
            available_at=available_at,
            confidence=round(_clamp(confidence, 0, 1), 4),
            fresh_component_count=fresh_count,
            stale_component_count=stale_count,
            components=components,
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class _ChartSeries:
    symbol: str
    closes: tuple[float, ...]
    timestamps: tuple[datetime, ...]

    @property
    def last_close(self) -> float | None:
        return self.closes[-1] if self.closes else None

    @property
    def last_timestamp(self) -> datetime | None:
        return self.timestamps[-1] if self.timestamps else None


@dataclass(frozen=True)
class _MarketSeries:
    symbol: str
    daily_closes: tuple[float, ...]
    current_close: float
    observed_at: datetime
    source: str


class LiveFearGreedProvider:
    """Build a live CNN-style fear/greed composite from durable market feeds.

    The provider intentionally emits component-level confidence and staleness
    instead of pretending that every source is real-time. No-key market data uses
    Yahoo's chart endpoint; optional Alpha Vantage news sentiment is included
    when `ALPHA_VANTAGE_API_KEY` is configured.
    """

    market_symbols = (
        "SPY",
        "QQQ",
        "IWM",
        "RSP",
        "HYG",
        "LQD",
        "TLT",
        "GLD",
        "UUP",
        "^VIX",
    )

    def __init__(
        self,
        *,
        settings: AppSettings | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = http_client or httpx.Client(timeout=20, follow_redirects=True)

    def fetch_snapshot(self, *, now: datetime | None = None) -> FearGreedSnapshot:
        available_at = _ensure_utc(now or datetime.now(tz=UTC))
        market, warnings = self._fetch_market_data()
        components: list[FearGreedComponent] = []

        builder_type = Callable[[dict[str, _MarketSeries], datetime], FearGreedComponent | None]
        builders: tuple[tuple[str, builder_type], ...] = (
            ("market_momentum", self._market_momentum_component),
            ("breadth_proxy", self._breadth_proxy_component),
            ("volatility", self._volatility_component),
            ("credit_risk_appetite", self._credit_risk_component),
            ("safe_haven_demand", self._safe_haven_component),
            ("growth_speculation", self._growth_speculation_component),
        )
        for name, builder in builders:
            try:
                component = builder(market, available_at)
            except Exception as exc:  # pragma: no cover - defensive per-source isolation.
                warnings.append(f"{name}: {exc}")
                continue
            if component is not None:
                components.append(component)

        try:
            options_component = self._put_call_component(available_at)
        except Exception as exc:  # pragma: no cover - defensive per-source isolation.
            warnings.append(f"put_call_options: {exc}")
        else:
            if options_component is not None:
                components.append(options_component)

        try:
            news_component = self._alpha_news_component(available_at)
        except Exception as exc:  # pragma: no cover - defensive per-source isolation.
            warnings.append(f"news_sentiment: {exc}")
        else:
            if news_component is not None:
                components.append(news_component)

        if len(components) < self._settings.fear_greed_min_components:
            raise ProviderError(
                "insufficient fear/greed components: "
                f"{len(components)} < {self._settings.fear_greed_min_components}; "
                f"warnings={warnings}"
            )

        return FearGreedSnapshot.from_components(
            tuple(components),
            available_at=available_at,
            expected_component_count=7,
            metadata={"warnings": warnings} if warnings else {},
        )

    def append_snapshot(
        self,
        snapshot: FearGreedSnapshot,
        *,
        path: Path | None = None,
    ) -> Path:
        target = path or self._settings.fear_greed_snapshot_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(snapshot.model_dump_json())
            fh.write("\n")
        return target

    def load_latest_snapshot(
        self,
        *,
        path: Path | None = None,
        max_age_minutes: int | None = None,
        now: datetime | None = None,
    ) -> FearGreedSnapshot | None:
        target = path or self._settings.fear_greed_snapshot_path
        if not target.exists():
            return None

        available_at = _ensure_utc(now or datetime.now(tz=UTC))
        max_age = timedelta(
            minutes=max_age_minutes or self._settings.fear_greed_component_max_age_minutes
        )
        lines = target.read_text(encoding="utf-8").splitlines()
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                snapshot = FearGreedSnapshot.model_validate_json(line)
            except ValueError:
                continue
            if available_at - _ensure_utc(snapshot.available_at) <= max_age:
                return snapshot
        return None

    def _fetch_market_data(self) -> tuple[dict[str, _MarketSeries], list[str]]:
        market: dict[str, _MarketSeries] = {}
        warnings: list[str] = []
        for symbol in self.market_symbols:
            try:
                daily = self._fetch_yahoo_chart(symbol, chart_range="1y", interval="1d")
                intraday = self._fetch_yahoo_chart(symbol, chart_range="5d", interval="5m")
            except Exception as exc:
                warnings.append(f"{symbol}: {exc}")
                continue

            current_close = intraday.last_close or daily.last_close
            observed_at = intraday.last_timestamp or daily.last_timestamp
            if current_close is None or observed_at is None or not daily.closes:
                warnings.append(f"{symbol}: no usable chart data")
                continue

            market[symbol] = _MarketSeries(
                symbol=symbol,
                daily_closes=daily.closes,
                current_close=current_close,
                observed_at=observed_at,
                source="yahoo_chart",
            )
        return market, warnings

    def _fetch_yahoo_chart(self, symbol: str, *, chart_range: str, interval: str) -> _ChartSeries:
        encoded_symbol = quote(symbol, safe="")
        response = self._client.get(
            YAHOO_CHART_URL.format(symbol=encoded_symbol),
            params={
                "range": chart_range,
                "interval": interval,
                "includePrePost": "false",
            },
            headers={"User-Agent": "AI-Trader research"},
        )
        response.raise_for_status()
        payload = response.json()
        result = (payload.get("chart", {}).get("result") or [None])[0]
        if result is None:
            error = payload.get("chart", {}).get("error") or {}
            raise ProviderError(f"Yahoo chart returned no result for {symbol}: {error}")

        timestamps = result.get("timestamp") or ()
        quote_data = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        closes = quote_data.get("close") or ()
        clean_timestamps: list[datetime] = []
        clean_closes: list[float] = []
        for timestamp, close in zip(timestamps, closes, strict=False):
            if close is None:
                continue
            close_float = float(close)
            if close_float <= 0 or not math.isfinite(close_float):
                continue
            clean_timestamps.append(datetime.fromtimestamp(int(timestamp), tz=UTC))
            clean_closes.append(close_float)
        return _ChartSeries(
            symbol=symbol,
            closes=tuple(clean_closes),
            timestamps=tuple(clean_timestamps),
        )

    def _market_momentum_component(
        self,
        market: dict[str, _MarketSeries],
        available_at: datetime,
    ) -> FearGreedComponent | None:
        spy = market.get("SPY")
        if spy is None or len(spy.daily_closes) < 125:
            return None
        sma_125 = _mean_tail(spy.daily_closes, 125)
        distance = spy.current_close / sma_125 - 1
        score = _linear_score(distance, -0.10, 0.10)
        return self._component(
            name="market_momentum",
            score=score,
            raw_value=distance,
            source=spy.source,
            observed_at=spy.observed_at,
            available_at=available_at,
            confidence=0.90,
            metadata={
                "symbol": "SPY",
                "current_close": round(spy.current_close, 4),
                "sma_125": round(sma_125, 4),
                "distance_pct": round(distance * 100, 4),
            },
        )

    def _breadth_proxy_component(
        self,
        market: dict[str, _MarketSeries],
        available_at: datetime,
    ) -> FearGreedComponent | None:
        scores: list[float] = []
        distances: dict[str, float] = {}
        for symbol in ("SPY", "QQQ", "IWM", "RSP"):
            series = market.get(symbol)
            if series is None or len(series.daily_closes) < 20:
                continue
            sma_20 = _mean_tail(series.daily_closes, 20)
            distance = series.current_close / sma_20 - 1
            scores.append(_linear_score(distance, -0.05, 0.05))
            distances[symbol] = distance
        if not scores:
            return None
        observed_at = max(market[symbol].observed_at for symbol in distances)
        return self._component(
            name="breadth_proxy",
            score=fmean(scores),
            raw_value=fmean(distances.values()) if distances else None,
            source="yahoo_chart",
            observed_at=observed_at,
            available_at=available_at,
            confidence=0.72,
            metadata={
                "symbols": tuple(distances),
                "distance_pct": {
                    symbol: round(distance * 100, 4)
                    for symbol, distance in distances.items()
                },
            },
        )

    def _volatility_component(
        self,
        market: dict[str, _MarketSeries],
        available_at: datetime,
    ) -> FearGreedComponent | None:
        vix = market.get("^VIX")
        if vix is None:
            return None
        score = 100 - _linear_score(vix.current_close, 12, 35)
        return self._component(
            name="volatility",
            score=score,
            raw_value=vix.current_close,
            source=vix.source,
            observed_at=vix.observed_at,
            available_at=available_at,
            confidence=0.88,
            metadata={"symbol": "^VIX", "vix": round(vix.current_close, 4)},
        )

    def _credit_risk_component(
        self,
        market: dict[str, _MarketSeries],
        available_at: datetime,
    ) -> FearGreedComponent | None:
        hyg = market.get("HYG")
        lqd = market.get("LQD")
        if hyg is None or lqd is None:
            return None
        ratio_series = _ratio_series(hyg.daily_closes, lqd.daily_closes)
        if len(ratio_series) < 20:
            return None
        current_ratio = hyg.current_close / lqd.current_close
        ratio_sma_20 = _mean_tail(ratio_series, 20)
        distance = current_ratio / ratio_sma_20 - 1
        return self._component(
            name="credit_risk_appetite",
            score=_linear_score(distance, -0.025, 0.025),
            raw_value=distance,
            source="yahoo_chart",
            observed_at=max(hyg.observed_at, lqd.observed_at),
            available_at=available_at,
            confidence=0.76,
            metadata={
                "ratio": "HYG/LQD",
                "current_ratio": round(current_ratio, 6),
                "sma_20": round(ratio_sma_20, 6),
                "distance_pct": round(distance * 100, 4),
            },
        )

    def _safe_haven_component(
        self,
        market: dict[str, _MarketSeries],
        available_at: datetime,
    ) -> FearGreedComponent | None:
        spy = market.get("SPY")
        if spy is None:
            return None
        spreads: dict[str, float] = {}
        for symbol in ("TLT", "GLD", "UUP"):
            haven = market.get(symbol)
            if haven is None:
                continue
            spy_return = _tail_return(spy.daily_closes, spy.current_close, days=5)
            haven_return = _tail_return(haven.daily_closes, haven.current_close, days=5)
            if spy_return is None or haven_return is None:
                continue
            spreads[f"SPY-{symbol}"] = spy_return - haven_return
        if not spreads:
            return None
        score = fmean(_linear_score(value, -0.04, 0.04) for value in spreads.values())
        observed_at = max(
            market[symbol].observed_at
            for symbol in ("SPY", "TLT", "GLD", "UUP")
            if symbol in market
        )
        return self._component(
            name="safe_haven_demand",
            score=score,
            raw_value=fmean(spreads.values()),
            source="yahoo_chart",
            observed_at=observed_at,
            available_at=available_at,
            confidence=0.72,
            metadata={key: round(value * 100, 4) for key, value in spreads.items()},
        )

    def _growth_speculation_component(
        self,
        market: dict[str, _MarketSeries],
        available_at: datetime,
    ) -> FearGreedComponent | None:
        spy = market.get("SPY")
        if spy is None:
            return None
        spreads: dict[str, float] = {}
        for symbol in ("QQQ", "IWM"):
            risk_asset = market.get(symbol)
            if risk_asset is None:
                continue
            spy_return = _tail_return(spy.daily_closes, spy.current_close, days=5)
            asset_return = _tail_return(risk_asset.daily_closes, risk_asset.current_close, days=5)
            if spy_return is None or asset_return is None:
                continue
            spreads[f"{symbol}-SPY"] = asset_return - spy_return
        if not spreads:
            return None
        score = fmean(_linear_score(value, -0.03, 0.03) for value in spreads.values())
        observed_at = max(
            market[symbol].observed_at
            for symbol in ("SPY", "QQQ", "IWM")
            if symbol in market
        )
        return self._component(
            name="growth_speculation",
            score=score,
            raw_value=fmean(spreads.values()),
            source="yahoo_chart",
            observed_at=observed_at,
            available_at=available_at,
            confidence=0.68,
            metadata={key: round(value * 100, 4) for key, value in spreads.items()},
        )

    def _put_call_component(self, available_at: datetime) -> FearGreedComponent | None:
        response = self._client.get(
            CBOE_DAILY_STATS_URL,
            headers={"User-Agent": "AI-Trader research"},
        )
        response.raise_for_status()
        plain_text = re.sub(r"<[^>]+>", " ", response.text)
        plain_text = re.sub(r"\s+", " ", plain_text)
        match = re.search(
            r"TOTAL\s+PUT/CALL\s+RATIO\s+([0-9]+(?:\.[0-9]+)?)",
            plain_text,
            flags=re.IGNORECASE,
        )
        if match is None:
            return None
        ratio = float(match.group(1))
        score = 100 - _linear_score(ratio, 0.60, 1.20)
        return self._component(
            name="put_call_options",
            score=score,
            raw_value=ratio,
            source="cboe_daily_market_statistics",
            observed_at=available_at,
            available_at=available_at,
            confidence=0.48,
            max_age_minutes=36 * 60,
            metadata={"total_put_call_ratio": ratio},
        )

    def _alpha_news_component(self, available_at: datetime) -> FearGreedComponent | None:
        if self._settings.alpha_vantage_api_key is None:
            return None
        response = self._client.get(
            ALPHA_VANTAGE_QUERY_URL,
            params={
                "function": "NEWS_SENTIMENT",
                "topics": "financial_markets",
                "sort": "LATEST",
                "limit": "50",
                "apikey": self._settings.alpha_vantage_api_key.get_secret_value(),
            },
            headers={"User-Agent": "AI-Trader research"},
        )
        response.raise_for_status()
        payload = response.json()
        feed = payload.get("feed") or []
        scores: list[float] = []
        timestamps: list[datetime] = []
        for item in feed:
            try:
                scores.append(float(item["overall_sentiment_score"]))
            except (KeyError, TypeError, ValueError):
                continue
            published = _parse_alpha_time(item.get("time_published"))
            if published is not None:
                timestamps.append(published)
        if not scores:
            return None
        average_score = fmean(scores)
        observed_at = max(timestamps) if timestamps else available_at
        return self._component(
            name="news_sentiment",
            score=_linear_score(average_score, -0.25, 0.25),
            raw_value=average_score,
            source="alpha_vantage_news_sentiment",
            observed_at=observed_at,
            available_at=available_at,
            confidence=min(0.78, 0.25 + len(scores) / 80),
            max_age_minutes=12 * 60,
            metadata={"article_count": len(scores)},
        )

    def _component(
        self,
        *,
        name: str,
        score: float,
        raw_value: float | None,
        source: str,
        observed_at: datetime,
        available_at: datetime,
        confidence: float,
        max_age_minutes: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FearGreedComponent:
        observed_at = _ensure_utc(observed_at)
        available_at = _ensure_utc(available_at)
        max_age = timedelta(
            minutes=max_age_minutes or self._settings.fear_greed_component_max_age_minutes
        )
        age = max(available_at - observed_at, timedelta())
        stale = age > max_age
        adjusted_confidence = confidence * (0.55 if stale else 1.0)
        return FearGreedComponent(
            name=name,
            score=round(_clamp(score, 0, 100), 2),
            raw_value=(
                round(raw_value, 8)
                if raw_value is not None and math.isfinite(raw_value)
                else None
            ),
            source=source,
            observed_at=observed_at,
            available_at=available_at,
            confidence=round(_clamp(adjusted_confidence, 0, 1), 4),
            stale=stale,
            metadata={
                **(metadata or {}),
                "age_minutes": round(age.total_seconds() / 60, 2),
            },
        )


def classify_fear_greed(value: float) -> str:
    if value < 25:
        return "extreme_fear"
    if value < 45:
        return "fear"
    if value <= 55:
        return "neutral"
    if value <= 75:
        return "greed"
    return "extreme_greed"


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _linear_score(value: float, fear_at: float, greed_at: float) -> float:
    if greed_at == fear_at:
        return 50.0
    return _clamp((value - fear_at) / (greed_at - fear_at) * 100, 0, 100)


def _mean_tail(values: tuple[float, ...], length: int) -> float:
    if len(values) < length:
        raise ValueError(f"need at least {length} values")
    return fmean(values[-length:])


def _ratio_series(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    size = min(len(left), len(right))
    ratios: list[float] = []
    for numerator, denominator in zip(left[-size:], right[-size:], strict=False):
        if denominator > 0:
            ratios.append(numerator / denominator)
    return tuple(ratios)


def _tail_return(
    closes: tuple[float, ...],
    current_close: float,
    *,
    days: int,
) -> float | None:
    if len(closes) <= days or closes[-days - 1] <= 0:
        return None
    return current_close / closes[-days - 1] - 1


def _parse_alpha_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
    except ValueError:
        return None
