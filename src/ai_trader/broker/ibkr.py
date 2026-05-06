from __future__ import annotations

import math
from dataclasses import dataclass

from ai_trader.config import AppSettings, get_settings
from ai_trader.broker.contracts import (
    BrokerAccountSnapshot,
    BrokerOrder,
    BrokerOrderResult,
    BrokerPosition,
    BrokerQuote,
    OrderType,
)
from ai_trader.broker.errors import BrokerConfigurationError, BrokerConnectionError

try:
    from ib_insync import IB, LimitOrder, MarketOrder, Stock
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "IBKR support requires `ib_insync`. Install it and ensure TWS/IB Gateway is running."
    ) from exc


@dataclass
class IBKRConnection:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    account: str | None = None
    readonly: bool = False


class IBKRBroker:
    """Thin wrapper around `ib_insync` for account/positions/orders."""

    def __init__(
        self,
        *,
        connection: IBKRConnection,
        trading_mode: str = "paper",
        allow_live_trading: bool = False,
        timeout_s: float = 4.0,
    ) -> None:
        self._connection = connection
        self._trading_mode = (trading_mode or "paper").casefold().strip()
        self._allow_live_trading = allow_live_trading
        self._timeout_s = timeout_s
        self._ib = IB()

    @classmethod
    def from_settings(cls, settings: AppSettings | None = None) -> IBKRBroker:
        settings = settings or get_settings()
        connection = IBKRConnection(
            host=settings.ibkr_host,
            port=settings.ibkr_port,
            client_id=settings.ibkr_client_id,
            account=settings.ibkr_account,
            readonly=settings.ibkr_readonly,
        )
        return cls(
            connection=connection,
            trading_mode=settings.trading_mode,
            allow_live_trading=settings.allow_live_trading,
        )

    def __enter__(self) -> IBKRBroker:
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()

    def connect(self) -> None:
        if self._trading_mode == "live" and not self._allow_live_trading:
            raise BrokerConfigurationError(
                "Live trading is disabled. Set AI_TRADER_ALLOW_LIVE_TRADING=true to enable."
            )

        try:
            self._ib.connect(
                host=self._connection.host,
                port=self._connection.port,
                clientId=self._connection.client_id,
                timeout=self._timeout_s,
                readonly=self._connection.readonly,
                account=self._connection.account or "",
            )
        except Exception as exc:
            raise BrokerConnectionError(f"IBKR connect failed: {exc}") from exc

    def disconnect(self) -> None:
        if self._ib.isConnected():
            self._ib.disconnect()

    def account_snapshot(self, currency: str = "USD") -> BrokerAccountSnapshot:
        if not self._ib.isConnected():
            raise BrokerConnectionError("IBKR is not connected")

        values = self._account_summary_values()
        account = self._connection.account or _first_account(values)
        return BrokerAccountSnapshot(
            account=account,
            currency=currency,
            available_funds=_read_account_value(values, "AvailableFunds", currency, account),
            buying_power=_read_account_value(values, "BuyingPower", currency, account),
            net_liquidation=_read_account_value(values, "NetLiquidation", currency, account),
            cash_balance=_read_account_value(values, "TotalCashValue", currency, account),
        )

    def market_price(self, ticker: str, currency: str = "USD") -> BrokerQuote:
        if not self._ib.isConnected():
            raise BrokerConnectionError("IBKR is not connected")

        contract = Stock(ticker.upper(), "SMART", currency)
        try:
            qualified = self._ib.qualifyContracts(contract)
        except Exception as exc:
            raise BrokerConnectionError(f"IBKR contract qualification failed: {exc}") from exc
        if not qualified:
            raise BrokerConnectionError(f"IBKR could not qualify contract for {ticker}")

        contract = qualified[0]
        ib_ticker = None
        try:
            ib_ticker = self._ib.reqMktData(contract, "", False, False)
            self._ib.sleep(1.0)
            price = _read_ticker_price(ib_ticker)
        except Exception as exc:
            message = f"IBKR market data request failed for {ticker}: {exc}"
            raise BrokerConnectionError(message) from exc
        finally:
            if ib_ticker is not None:
                try:
                    self._ib.cancelMktData(contract)
                except Exception:
                    pass

        if price is None:
            price = self._last_historical_close(contract, ticker)
        return BrokerQuote(ticker=ticker.upper(), price=price, currency=currency)

    def positions(self) -> tuple[BrokerPosition, ...]:
        if not self._ib.isConnected():
            raise BrokerConnectionError("IBKR is not connected")
        positions = []
        for position in self._ib.positions():
            contract = position.contract
            positions.append(
                BrokerPosition(
                    account=position.account,
                    ticker=str(contract.symbol),
                    quantity=float(position.position),
                    avg_cost=float(position.avgCost),
                    security_type=str(contract.secType),
                    currency=str(contract.currency),
                )
            )
        return tuple(positions)

    def place_order(self, order: BrokerOrder) -> BrokerOrderResult:
        if not self._ib.isConnected():
            raise BrokerConnectionError("IBKR is not connected")
        if self._connection.readonly:
            raise BrokerConfigurationError("IBKR connection is read-only; cannot place orders.")

        contract = Stock(order.ticker.upper(), order.exchange, order.currency)
        try:
            qualified = self._ib.qualifyContracts(contract)
        except Exception as exc:
            raise BrokerConnectionError(f"IBKR contract qualification failed: {exc}") from exc
        if not qualified:
            raise BrokerConnectionError(f"IBKR could not qualify contract for {order.ticker}")

        ib_order = _to_ib_order(order)
        trade = self._ib.placeOrder(contract, ib_order)
        self._ib.sleep(0.2)
        status = trade.orderStatus.status or "submitted"
        return BrokerOrderResult(
            order_id=getattr(trade.order, "orderId", None),
            status=status,
            filled=(
                float(trade.orderStatus.filled)
                if trade.orderStatus.filled is not None
                else None
            ),
            avg_fill_price=float(trade.orderStatus.avgFillPrice)
            if trade.orderStatus.avgFillPrice is not None
            else None,
        )

    def _account_summary_values(self):
        try:
            return self._ib.accountSummary(account=self._connection.account or "")
        except TypeError:
            return self._ib.accountSummary()
        except Exception as exc:
            raise BrokerConnectionError(f"IBKR account summary failed: {exc}") from exc

    def _last_historical_close(self, contract, ticker: str) -> float:
        try:
            bars = self._ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr="2 D",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )
        except Exception as exc:
            raise BrokerConnectionError(
                f"IBKR historical price fallback failed for {ticker}: {exc}"
            ) from exc
        if not bars:
            raise BrokerConnectionError(f"IBKR did not return a usable price for {ticker}")
        price = _finite_positive(getattr(bars[-1], "close", None))
        if price is None:
            raise BrokerConnectionError(f"IBKR did not return a usable price for {ticker}")
        return price


def _to_ib_order(order: BrokerOrder):
    action = order.side.value
    qty = order.quantity
    if order.order_type is OrderType.MARKET:
        return MarketOrder(action, qty)
    if order.order_type is OrderType.LIMIT:
        if order.limit_price is None:
            raise BrokerConfigurationError("limit_price is required for LIMIT orders")
        return LimitOrder(action, qty, order.limit_price)
    raise BrokerConfigurationError(f"Unsupported order_type: {order.order_type}")


def _first_account(values) -> str | None:
    for item in values:
        account = str(getattr(item, "account", "") or "").strip()
        if account:
            return account
    return None


def _read_account_value(values, tag: str, currency: str, account: str | None) -> float:
    for item in values:
        item_tag = str(getattr(item, "tag", "") or "")
        item_currency = str(getattr(item, "currency", "") or "")
        item_account = str(getattr(item, "account", "") or "")
        if item_tag != tag:
            continue
        if currency and item_currency and item_currency.upper() != currency.upper():
            continue
        if account and item_account and item_account != account:
            continue
        value = _finite_positive(getattr(item, "value", None))
        if value is not None:
            return value
    return 0.0


def _read_ticker_price(ib_ticker) -> float | None:
    try:
        value = ib_ticker.marketPrice()
        price = _finite_positive(value)
        if price is not None:
            return price
    except Exception:
        pass

    bid = _finite_positive(getattr(ib_ticker, "bid", None))
    ask = _finite_positive(getattr(ib_ticker, "ask", None))
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0

    for attr in ("last", "close"):
        price = _finite_positive(getattr(ib_ticker, attr, None))
        if price is not None:
            return price
    return None


def _finite_positive(value) -> float | None:
    try:
        numeric = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None
    if math.isfinite(numeric) and numeric > 0:
        return numeric
    return None
