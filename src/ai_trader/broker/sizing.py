from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field

from ai_trader.broker.contracts import BrokerAccountSnapshot, BrokerQuote, OrderSide
from ai_trader.intelligence.trade_plan import TradePlan


class BalanceSizingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    cash_fraction: float = Field(default=0.02, gt=0.0, le=1.0)
    allow_fractional_shares: bool = False
    min_quantity: float = Field(default=0.0001, gt=0.0)
    fractional_precision: int = Field(default=4, ge=0, le=8)


class BalanceSizingResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    side: OrderSide
    quantity: float
    reference_price: float
    balance: float
    cash_fraction: float
    scaled_cash_fraction: float
    conviction: float
    size_multiplier: float
    target_notional: float
    order_notional: float


def size_order_from_balance(
    *,
    plan: TradePlan,
    side: OrderSide,
    account: BrokerAccountSnapshot,
    quote: BrokerQuote,
    config: BalanceSizingConfig | None = None,
) -> BalanceSizingResult:
    """Calculate an order quantity from available balance and plan strength."""

    config = config or BalanceSizingConfig()
    balance = account.spendable_balance
    if balance <= 0:
        raise ValueError("available balance is zero; cannot size an order")

    conviction = max(0.0, float(plan.conviction))
    size_multiplier = max(0.0, float(plan.size_multiplier))
    scaled_cash_fraction = min(1.0, config.cash_fraction * conviction * size_multiplier)
    target_notional = balance * scaled_cash_fraction
    if target_notional <= 0:
        raise ValueError("plan conviction and size multiplier resolved to a zero-sized order")

    raw_quantity = target_notional / quote.price
    if config.allow_fractional_shares:
        quantity = round(raw_quantity, config.fractional_precision)
        minimum_quantity = config.min_quantity
    else:
        quantity = float(math.floor(raw_quantity))
        minimum_quantity = max(1.0, config.min_quantity)

    if quantity < minimum_quantity:
        raise ValueError(
            "calculated order quantity is below the minimum tradable size; "
            "increase balance/cash fraction or allow fractional shares"
        )

    order_notional = quantity * quote.price
    return BalanceSizingResult(
        ticker=plan.ticker.upper(),
        side=side,
        quantity=quantity,
        reference_price=quote.price,
        balance=balance,
        cash_fraction=config.cash_fraction,
        scaled_cash_fraction=scaled_cash_fraction,
        conviction=conviction,
        size_multiplier=size_multiplier,
        target_notional=target_notional,
        order_notional=order_notional,
    )
