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


class Broker(Protocol):
    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def positions(self) -> tuple[BrokerPosition, ...]: ...

    def place_order(self, order: BrokerOrder) -> BrokerOrderResult: ...

