from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Sequence

import httpx

from ai_trader.config import AppSettings, get_settings
from ai_trader.domain.events import Chamber, CongressionalTrade, TransactionType
from ai_trader.providers.contracts import ProviderError


class QuiverProvider:
    URL = "https://api.quiverquant.com/beta/bulk/congresstrading"

    def __init__(self, *, settings: AppSettings | None = None, http_client: httpx.Client | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = http_client or httpx.Client(timeout=30)

    def fetch_trades(
        self,
        start: date,
        end: date,
        tickers: Sequence[str] | None = None,
    ) -> Sequence[CongressionalTrade]:
        if self._settings.quiver_api_key is None:
            raise ProviderError("QUIVER_API_KEY is required")
        response = self._client.get(
            self.URL,
            headers={"Authorization": f"Bearer {self._settings.quiver_api_key.get_secret_value()}"},
        )
        response.raise_for_status()
        allowed = {ticker.upper() for ticker in tickers} if tickers else None
        trades = []
        for item in response.json():
            ticker = str(item.get("Ticker") or item.get("ticker") or "").upper()
            if allowed is not None and ticker not in allowed:
                continue
            disclosure = _parse_date(item.get("FiledAfterDate") or item.get("ReportDate") or item.get("FiledDate"))
            if disclosure is None or disclosure < start or disclosure > end:
                continue
            trades.append(
                CongressionalTrade(
                    member_name=str(item.get("Representative") or item.get("Member") or "Unknown"),
                    chamber=_parse_chamber(item.get("Chamber")),
                    ticker=ticker,
                    issuer=str(item.get("Company") or item.get("Issuer") or ticker),
                    sector=str(item.get("Sector") or "unknown"),
                    transaction_date=_parse_date(item.get("TransactionDate")) or disclosure,
                    disclosure_date=disclosure,
                    transaction_type=_parse_transaction_type(item.get("Transaction")),
                    amount_low=Decimal(str(item.get("RangeLow") or item.get("AmountLow") or 0)),
                    amount_high=Decimal(str(item.get("RangeHigh") or item.get("AmountHigh") or 0)),
                    source_url=item.get("URL"),
                )
            )
        return tuple(trades)


def _parse_date(value) -> date | None:
    if value is None or value == "":
        return None
    return date.fromisoformat(str(value)[:10])


def _parse_chamber(value) -> Chamber:
    text = str(value or "").casefold()
    return Chamber.SENATE if "sen" in text else Chamber.HOUSE


def _parse_transaction_type(value) -> TransactionType:
    text = str(value or "").casefold()
    if "purchase" in text or text == "p":
        return TransactionType.PURCHASE
    if "sale" in text or text == "s":
        return TransactionType.SALE
    if "exchange" in text:
        return TransactionType.EXCHANGE
    return TransactionType.OTHER

