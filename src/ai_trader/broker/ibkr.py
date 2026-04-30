from __future__ import annotations

from dataclasses import dataclass

from ai_trader.config import AppSettings, get_settings
from ai_trader.broker.contracts import BrokerOrder, BrokerOrderResult, BrokerPosition, OrderSide, OrderType
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
            filled=float(trade.orderStatus.filled) if trade.orderStatus.filled is not None else None,
            avg_fill_price=float(trade.orderStatus.avgFillPrice)
            if trade.orderStatus.avgFillPrice is not None
            else None,
        )


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

