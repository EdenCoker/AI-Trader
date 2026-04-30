from __future__ import annotations

from datetime import date, datetime
from typing import Protocol, Sequence

from ai_trader.domain.events import (
    CongressionalTrade,
    MacroObservation,
    MarketTick,
    NewsArticle,
    SocialMention,
    ThirteenFHolding,
)


class ProviderError(RuntimeError):
    """Raised when an external provider cannot satisfy a request."""


class CongressionalTradesProvider(Protocol):
    def fetch_trades(
        self,
        start: date,
        end: date,
        tickers: Sequence[str] | None = None,
    ) -> Sequence[CongressionalTrade]:
        """Fetch STOCK Act disclosures by disclosure date."""


class ThirteenFProvider(Protocol):
    def fetch_holdings(
        self,
        manager_ciks: Sequence[str],
        filed_after: date,
        filed_before: date,
    ) -> Sequence[ThirteenFHolding]:
        """Fetch 13F holdings by filing date."""


class MarketDataProvider(Protocol):
    def fetch_ticks(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
    ) -> Sequence[MarketTick]:
        """Fetch market ticks for a ticker and time range."""


class NewsProvider(Protocol):
    def fetch_news(
        self,
        tickers: Sequence[str],
        start: datetime,
        end: datetime,
    ) -> Sequence[NewsArticle]:
        """Fetch timestamped news items."""


class SentimentProvider(Protocol):
    def fetch_mentions(
        self,
        tickers: Sequence[str],
        start: datetime,
        end: datetime,
    ) -> Sequence[SocialMention]:
        """Fetch timestamped public social mentions."""


class MacroProvider(Protocol):
    def fetch_observations(
        self,
        series_ids: Sequence[str],
        released_after: date,
        released_before: date,
    ) -> Sequence[MacroObservation]:
        """Fetch macro observations by release date."""

