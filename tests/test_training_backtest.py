from datetime import date

import pytest

from ai_trader.domain.signals import Signal, SignalBundle, SignalDirection
from ai_trader.intelligence.trade_plan import TradePlan
from ai_trader.training.backtest import StrategyBacktestConfig, run_strategy_backtest
from ai_trader.training.data import LocalTrainingExample


def _example(
    *,
    ticker: str,
    as_of: date,
    signal_name: str,
    direction: SignalDirection,
    conviction: float,
    pnl: float,
) -> LocalTrainingExample:
    bundle = SignalBundle(
        ticker=ticker,
        as_of=as_of,
        signals=(
            Signal(
                name=signal_name,
                ticker=ticker,
                direction=direction,
                strength=conviction,
                confidence=0.8,
                effective_date=as_of,
            ),
        ),
    )
    plan = TradePlan(
        ticker=ticker,
        as_of=as_of,
        direction=direction,
        conviction=conviction,
        size_multiplier=1.0,
        holding_period_days=30,
        exit_trigger="time exit",
    )
    return LocalTrainingExample(
        signal_bundle=bundle,
        trade_plan=plan,
        pnl_pct=pnl,
        metadata={"ticker": ticker, "fear_greed": 55},
    )


def test_strategy_backtest_selects_positive_signal_rule():
    examples = (
        _example(
            ticker="AAA",
            as_of=date(2024, 1, 5),
            signal_name="lobbying_activity",
            direction=SignalDirection.LONG,
            conviction=0.3,
            pnl=0.06,
        ),
        _example(
            ticker="BBB",
            as_of=date(2024, 2, 5),
            signal_name="lobbying_activity",
            direction=SignalDirection.LONG,
            conviction=0.3,
            pnl=0.04,
        ),
        _example(
            ticker="CCC",
            as_of=date(2024, 1, 5),
            signal_name="patent_filings",
            direction=SignalDirection.LONG,
            conviction=0.3,
            pnl=-0.08,
        ),
    )

    report = run_strategy_backtest(
        examples,
        StrategyBacktestConfig(min_trades=2, min_active_months=2, min_trades_per_month=1),
    )

    assert report.selected_strategy is not None
    assert report.selected_strategy.rule.signal_name == "lobbying_activity"
    assert report.selected_strategy.metrics.mean_return == pytest.approx(0.05)


def test_strategy_backtest_reports_train_and_test_metrics():
    examples = (
        _example(
            ticker="AAA",
            as_of=date(2024, 1, 5),
            signal_name="lobbying_activity",
            direction=SignalDirection.LONG,
            conviction=0.3,
            pnl=0.06,
        ),
        _example(
            ticker="BBB",
            as_of=date(2024, 2, 5),
            signal_name="lobbying_activity",
            direction=SignalDirection.LONG,
            conviction=0.3,
            pnl=0.04,
        ),
        _example(
            ticker="AAA",
            as_of=date(2024, 3, 5),
            signal_name="lobbying_activity",
            direction=SignalDirection.LONG,
            conviction=0.3,
            pnl=-0.02,
        ),
    )

    report = run_strategy_backtest(
        examples,
        StrategyBacktestConfig(
            split_date=date(2024, 3, 1),
            min_trades=2,
            min_active_months=2,
            min_trades_per_month=1,
        ),
    )

    assert report.selected_strategy is not None
    assert report.selected_strategy.train_metrics is not None
    assert report.selected_strategy.test_metrics is not None
    assert report.selected_strategy.train_metrics.trades == 2
    assert report.selected_strategy.test_metrics.trades == 1


def test_strategy_backtest_can_filter_by_date_range():
    examples = (
        _example(
            ticker="AAA",
            as_of=date(2023, 12, 5),
            signal_name="lobbying_activity",
            direction=SignalDirection.LONG,
            conviction=0.3,
            pnl=-0.06,
        ),
        _example(
            ticker="AAA",
            as_of=date(2024, 1, 5),
            signal_name="lobbying_activity",
            direction=SignalDirection.LONG,
            conviction=0.3,
            pnl=0.06,
        ),
        _example(
            ticker="BBB",
            as_of=date(2024, 2, 5),
            signal_name="lobbying_activity",
            direction=SignalDirection.LONG,
            conviction=0.3,
            pnl=0.04,
        ),
    )

    report = run_strategy_backtest(
        examples,
        StrategyBacktestConfig(
            start_date=date(2024, 1, 1),
            min_trades=2,
            min_active_months=2,
            min_trades_per_month=1,
        ),
    )

    assert report.selected_strategy is not None
    assert report.examples_used == 2
    assert report.selected_strategy.metrics.mean_return == pytest.approx(0.05)
