from __future__ import annotations

from enum import Enum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MKT"
    LIMIT = "LMT"


class BrokerOrder(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    side: OrderSide
    quantity: float = Field(gt=0)
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = Field(default=None, gt=0)
    currency: str = "USD"
    exchange: str = "SMART"


class BrokerOrderResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    order_id: int | None = None
    status: str
    filled: float | None = None
    avg_fill_price: float | None = None
    message: str | None = None


class BrokerPosition(BaseModel):
    model_config = ConfigDict(frozen=True)

    account: str
    ticker: str
    quantity: float
    avg_cost: float
    security_type: str
    currency: str


class BrokerAccountSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    account: str | None = None
    currency: str = "USD"
    available_funds: float = Field(default=0.0, ge=0)
    buying_power: float = Field(default=0.0, ge=0)
    net_liquidation: float = Field(default=0.0, ge=0)
    cash_balance: float = Field(default=0.0, ge=0)

    @property
    def spendable_balance(self) -> float:
        for value in (
            self.available_funds,
            self.buying_power,
            self.cash_balance,
            self.net_liquidation,
        ):
            if value > 0:
                return value
        return 0.0


class BrokerQuote(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    price: float = Field(gt=0)
    currency: str = "USD"
    source: str = "broker"


class Broker(Protocol):
    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def account_snapshot(self, currency: str = "USD") -> BrokerAccountSnapshot: ...

    def market_price(self, ticker: str, currency: str = "USD") -> BrokerQuote: ...

    def positions(self) -> tuple[BrokerPosition, ...]: ...

    def place_order(self, order: BrokerOrder) -> BrokerOrderResult: ...
