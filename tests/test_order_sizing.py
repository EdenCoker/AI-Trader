from datetime import date

import pytest

from ai_trader.broker.contracts import BrokerAccountSnapshot, BrokerQuote, OrderSide
from ai_trader.broker.sizing import BalanceSizingConfig, size_order_from_balance
from ai_trader.domain.signals import SignalDirection
from ai_trader.intelligence.trade_plan import TradePlan


def _plan(*, conviction: float = 0.7, size_multiplier: float = 1.4) -> TradePlan:
    return TradePlan(
        ticker="MSFT",
        as_of=date(2026, 4, 30),
        direction=SignalDirection.LONG,
        conviction=conviction,
        size_multiplier=size_multiplier,
        holding_period_days=30,
        exit_trigger="time exit",
    )


def test_size_order_from_available_balance_uses_plan_strength():
    result = size_order_from_balance(
        plan=_plan(),
        side=OrderSide.BUY,
        account=BrokerAccountSnapshot(available_funds=10_000),
        quote=BrokerQuote(ticker="MSFT", price=100),
        config=BalanceSizingConfig(cash_fraction=0.05),
    )

    assert result.scaled_cash_fraction == pytest.approx(0.049)
    assert result.target_notional == pytest.approx(490)
    assert result.quantity == 4
    assert result.order_notional == pytest.approx(400)


def test_size_order_can_return_fractional_quantity():
    result = size_order_from_balance(
        plan=_plan(),
        side=OrderSide.BUY,
        account=BrokerAccountSnapshot(available_funds=10_000),
        quote=BrokerQuote(ticker="MSFT", price=100),
        config=BalanceSizingConfig(cash_fraction=0.05, allow_fractional_shares=True),
    )

    assert result.quantity == pytest.approx(4.9)
    assert result.order_notional == pytest.approx(490)


def test_size_order_rejects_too_small_non_fractional_orders():
    with pytest.raises(ValueError, match="below the minimum"):
        size_order_from_balance(
            plan=_plan(conviction=0.2, size_multiplier=1.0),
            side=OrderSide.BUY,
            account=BrokerAccountSnapshot(available_funds=100),
            quote=BrokerQuote(ticker="MSFT", price=100),
            config=BalanceSizingConfig(cash_fraction=0.02),
        )
