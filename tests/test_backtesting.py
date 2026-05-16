from datetime import date
from decimal import Decimal
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
from pydantic import SecretStr

from ai_trader.backtesting.data_loader import PolygonDataLoader
from ai_trader.backtesting.engine import WalkForwardConfig, WalkForwardEngine
from ai_trader.backtesting.metrics import (
    cagr,
    calmar_ratio,
    cvar,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    stability_score,
    win_rate,
)
from ai_trader.backtesting.monte_carlo import StressMonteCarlo
from ai_trader.backtesting.replay import EventReplay, ReplayEvent
from ai_trader.config import AppSettings
from ai_trader.domain.events import (
    Chamber,
    CongressionalTrade,
    ThirteenFHolding,
    ThirteenFPositionChange,
    TransactionType,
)
from ai_trader.domain.signals import Signal, SignalBundle, SignalDirection


def test_polygon_loader_uses_cache_on_second_call(tmp_path: Path):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "results": [
                    {"t": 1640995200000, "o": 1, "h": 2, "l": 1, "c": 1.5, "v": 100, "vw": 1.4}
                ]
            },
        )

    loader = PolygonDataLoader(
        settings=AppSettings(polygon_api_key=SecretStr("key"), polygon_cache_dir=tmp_path),
        cache_dir=tmp_path,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    first = loader.load_ohlcv("MSFT", date(2022, 1, 1), date(2022, 1, 2))
    second = loader.load_ohlcv("MSFT", date(2022, 1, 1), date(2022, 1, 2))

    assert calls == 1
    assert first.equals(second)


def test_walk_forward_window_boundaries():
    windows = WalkForwardEngine.windows(
        date(2022, 1, 1),
        date(2022, 1, 20),
        WalkForwardConfig(train_window_days=5, test_window_days=3, step_days=4),
    )
    assert windows[0].train_start == date(2022, 1, 1)
    assert windows[0].train_end == date(2022, 1, 5)
    assert windows[0].test_start == date(2022, 1, 6)
    assert windows[0].test_end == date(2022, 1, 8)


def test_sharpe_ratio_matches_manual_calculation():
    returns = np.array([0.01, 0.02, -0.01, 0.03])
    excess = returns - 0.05 / 252
    expected = np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(252)
    assert sharpe_ratio(returns) == pytest.approx(expected)


def test_max_drawdown_zero_for_monotonic_equity():
    assert max_drawdown(np.array([1.0, 1.1, 1.2])) == 0.0


def test_stress_monte_carlo_shape_and_positive_sanity():
    mc = StressMonteCarlo(n_simulations=25, seed=1)
    result = mc.run(np.array([0.01, 0.02, 0.03]))
    assert mc.last_paths is not None
    assert mc.last_paths.shape == (25, 3)
    assert result.prob_ruin == 0.0


def _trade(disclosure: date, ticker: str = "MSFT") -> CongressionalTrade:
    return CongressionalTrade(
        member_name="Jane Senator",
        chamber=Chamber.SENATE,
        ticker=ticker,
        issuer=f"{ticker} Corp.",
        sector="technology",
        transaction_date=date(2022, 1, 1),
        disclosure_date=disclosure,
        transaction_type=TransactionType.PURCHASE,
        amount_low=Decimal("100001"),
        amount_high=Decimal("250000"),
    )


def test_event_replay_filters_by_effective_date_not_transaction_date():
    replay = EventReplay(
        [
            ReplayEvent(
                event_type="congressional_trade",
                congressional_trade=_trade(date(2022, 1, 10)),
            ),
        ]
    )

    assert replay.available_as_of(date(2022, 1, 9), ticker="MSFT") == ()
    assert len(replay.available_as_of(date(2022, 1, 10), ticker="MSFT")) == 1


def test_event_replay_jsonl_roundtrip(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    event = ReplayEvent(
        event_type="congressional_trade",
        congressional_trade=_trade(date(2022, 1, 10)),
    )
    path.write_text(event.model_dump_json() + "\n", encoding="utf-8")

    replay = EventReplay.from_jsonl(path)

    assert replay.events[0].ticker == "MSFT"
    assert replay.events[0].effective_date == date(2022, 1, 10)


def test_event_replay_discovers_tickers_in_date_range():
    replay = EventReplay(
        [
            ReplayEvent(
                event_type="congressional_trade",
                congressional_trade=_trade(date(2022, 1, 10), ticker="MSFT"),
            ),
            ReplayEvent(
                event_type="congressional_trade",
                congressional_trade=_trade(date(2022, 2, 10), ticker="AAPL"),
            ),
            ReplayEvent(
                event_type="congressional_trade",
                congressional_trade=_trade(date(2023, 1, 10), ticker="NVDA"),
            ),
        ]
    )

    assert replay.tickers(start=date(2022, 1, 1), end=date(2022, 12, 31)) == ("AAPL", "MSFT")


def test_walk_forward_engine_uses_replay_events_without_lookahead():
    class FakeLoader:
        def load_ohlcv(self, ticker, start, end, timespan="day"):
            return pd.DataFrame(
                [
                    {
                        "date": date(2022, 1, day),
                        "open": 100 + day,
                        "high": 0,
                        "low": 0,
                        "close": 101 + day,
                        "volume": 1,
                        "vwap": 0,
                    }
                    for day in range(6, 14)
                ]
            )

    replay = EventReplay(
        [
            ReplayEvent(
                event_type="congressional_trade",
                congressional_trade=_trade(date(2022, 1, 8)),
            ),
            ReplayEvent(
                event_type="congressional_trade",
                congressional_trade=_trade(date(2022, 1, 20)),
            ),
        ]
    )
    engine = WalkForwardEngine(data_loader=FakeLoader(), replay=replay)

    result = engine.run(
        None,
        date(2022, 1, 1),
        date(2022, 1, 15),
        WalkForwardConfig(
            train_window_days=5,
            test_window_days=8,
            step_days=8,
            signal_threshold=0.01,
            max_holding_days=2,
            cash_fraction=1.0,
        ),
    )

    assert result.metadata["mode"] == "event_replay"
    assert result.metadata["ticker_source"] == "events"
    assert result.tickers == ("MSFT",)
    assert len(result.trades) == 1
    assert result.trades[0].entry_date == date(2022, 1, 9)


def test_walk_forward_engine_applies_stop_loss_to_event_trades():
    class FakeLoader:
        def load_ohlcv(self, ticker, start, end, timespan="day"):
            return pd.DataFrame(
                [
                    {
                        "date": date(2022, 1, 9),
                        "open": 100,
                        "high": 101,
                        "low": 94,
                        "close": 96,
                        "volume": 1,
                        "vwap": 0,
                    },
                    {
                        "date": date(2022, 1, 10),
                        "open": 96,
                        "high": 97,
                        "low": 95,
                        "close": 96,
                        "volume": 1,
                        "vwap": 0,
                    },
                ]
            )

    replay = EventReplay(
        [
            ReplayEvent(
                event_type="congressional_trade",
                congressional_trade=_trade(date(2022, 1, 8)),
            )
        ]
    )
    result = WalkForwardEngine(data_loader=FakeLoader(), replay=replay).run(
        None,
        date(2022, 1, 1),
        date(2022, 1, 15),
        WalkForwardConfig(
            train_window_days=5,
            test_window_days=8,
            step_days=8,
            signal_threshold=0.01,
            stop_loss_pct=0.05,
        ),
    )

    assert result.trades[0].exit_reason == "stop_loss"
    assert result.trades[0].pnl_pct == pytest.approx(-0.05)


def test_walk_forward_engine_sizes_trades_from_starting_balance():
    class FakeLoader:
        def load_ohlcv(self, ticker, start, end, timespan="day"):
            return pd.DataFrame(
                [
                    {
                        "date": date(2022, 1, 9),
                        "open": 100,
                        "high": 101,
                        "low": 99,
                        "close": 100,
                        "volume": 1,
                        "vwap": 0,
                    },
                    {
                        "date": date(2022, 1, 10),
                        "open": 100,
                        "high": 111,
                        "low": 99,
                        "close": 110,
                        "volume": 1,
                        "vwap": 0,
                    },
                ]
            )

    class FakeScorer:
        def build_bundle(self, ticker, as_of, congressional_trades, institutional_changes):
            return SignalBundle(
                ticker=ticker,
                as_of=as_of,
                signals=(
                    Signal(
                        name="test_signal",
                        ticker=ticker,
                        direction=SignalDirection.LONG,
                        strength=1.0,
                        confidence=1.0,
                        effective_date=as_of,
                        horizon_days=1,
                    ),
                ),
            )

    replay = EventReplay(
        [
            ReplayEvent(
                event_type="congressional_trade",
                congressional_trade=_trade(date(2022, 1, 8)),
            )
        ]
    )
    result = WalkForwardEngine(
        data_loader=FakeLoader(),
        replay=replay,
        scorer=FakeScorer(),
    ).run(
        None,
        date(2022, 1, 1),
        date(2022, 1, 15),
        WalkForwardConfig(
            train_window_days=5,
            test_window_days=8,
            step_days=8,
            signal_threshold=0.01,
            starting_balance=1_000,
            cash_fraction=0.5,
        ),
    )

    trade = result.trades[0]
    assert trade.quantity == 5
    assert trade.notional == pytest.approx(500)
    assert trade.pnl_amount == pytest.approx(50)
    assert trade.account_return == pytest.approx(0.05)
    assert trade.balance_before == pytest.approx(1_000)
    assert trade.balance_after == pytest.approx(1_050)
    assert result.metadata["ending_balance"] == pytest.approx(1_050)


def test_event_replay_splits_13f_changes():
    holding = ThirteenFHolding(
        manager_name="Berkshire Hathaway",
        cik="1067983",
        filing_date=date(2022, 2, 15),
        report_period=date(2021, 12, 31),
        ticker="MSFT",
        issuer="Microsoft Corp.",
        market_value_usd=Decimal("1000000"),
        shares=Decimal("1000"),
    )
    change = ThirteenFPositionChange(current=holding, previous_shares=Decimal("500"))
    event = ReplayEvent(event_type="13f_change", thirteen_f_change=change)

    congressional, institutional = EventReplay.split((event,))

    assert congressional == ()
    assert institutional == (change,)


# ---------------------------------------------------------------------------
# Comprehensive metrics unit tests
# These cover every public function in backtesting.metrics that previously
# had no dedicated test: cagr, calmar_ratio, sortino_ratio, cvar, win_rate,
# profit_factor, and stability_score.
# ---------------------------------------------------------------------------


class TestCagr:
    def test_zero_for_empty_input(self):
        assert cagr(np.array([])) == 0.0

    def test_zero_for_single_element(self):
        assert cagr(np.array([0.05])) == 0.0

    def test_flat_returns_give_zero_cagr(self):
        # Daily return of 0 -> no growth -> 0 % annualised.
        assert cagr(np.zeros(252)) == pytest.approx(0.0)

    def test_known_constant_daily_return(self):
        # 252 trading days of +0.01/day -> terminal value = 1.01^252
        # CAGR for exactly 1 year = 1.01^252 - 1
        daily = 0.01
        returns = np.full(252, daily)
        expected = (1 + daily) ** 252 - 1
        assert cagr(returns) == pytest.approx(expected, rel=1e-6)

    def test_positive_cagr_for_rising_equity(self):
        # Any consistently positive daily return must give positive CAGR.
        returns = np.full(504, 0.001)  # 2 years
        assert cagr(returns) > 0

    def test_negative_cagr_for_losing_equity(self):
        returns = np.full(252, -0.002)
        assert cagr(returns) < 0


class TestCalmarRatio:
    def test_zero_for_empty_input(self):
        assert calmar_ratio(np.array([])) == 0.0

    def test_zero_when_no_drawdown(self):
        # Monotonically rising equity has 0 drawdown -> calmar undefined -> 0.
        returns = np.full(252, 0.001)
        assert calmar_ratio(returns) == 0.0

    def test_positive_for_profitable_series_with_drawdown(self):
        # Mix of gains and one dip.
        returns = np.array([0.01, 0.02, -0.05, 0.03, 0.04] * 50)
        assert calmar_ratio(returns) > 0


class TestSortinoRatio:
    def test_zero_for_empty_input(self):
        assert sortino_ratio(np.array([])) == 0.0

    def test_zero_for_single_element(self):
        assert sortino_ratio(np.array([0.01])) == 0.0

    def test_zero_when_no_downside_returns(self):
        # All returns positive -> downside sample < 2 -> return 0.
        returns = np.array([0.01, 0.02, 0.03, 0.04])
        assert sortino_ratio(returns) == 0.0

    def test_higher_than_sharpe_for_positively_skewed_returns(self):
        # Sortino ignores upside volatility so it should be >= Sharpe when
        # there is significant upside.
        rng = np.random.default_rng(0)
        returns = rng.normal(0.001, 0.01, 500)
        # Add some large positive returns to increase upside vol.
        returns[:20] = 0.10
        assert sortino_ratio(returns) >= sharpe_ratio(returns)

    def test_positive_for_mean_positive_returns(self):
        returns = np.array([0.02, -0.01, 0.03, -0.005, 0.015, -0.008] * 30)
        assert sortino_ratio(returns) > 0


class TestCvar:
    def test_zero_for_empty_input(self):
        assert cvar(np.array([])) == 0.0

    def test_cvar_is_mean_of_worst_tail(self):
        # Simple deterministic series: worst 5 % of 100 values are -0.10.
        returns = np.concatenate([np.full(5, -0.10), np.full(95, 0.01)])
        result = cvar(returns, alpha=0.05)
        assert result == pytest.approx(-0.10)

    def test_cvar_is_more_negative_than_var(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 1000)
        var_cutoff = float(np.quantile(returns, 0.05))
        cvar_val = cvar(returns, alpha=0.05)
        assert cvar_val <= var_cutoff

    def test_all_same_returns(self):
        returns = np.full(100, -0.02)
        assert cvar(returns, alpha=0.05) == pytest.approx(-0.02)


class TestWinRate:
    def test_zero_for_empty_input(self):
        assert win_rate(np.array([])) == 0.0

    def test_all_winners(self):
        assert win_rate(np.array([0.01, 0.02, 0.03])) == pytest.approx(1.0)

    def test_all_losers(self):
        assert win_rate(np.array([-0.01, -0.02, -0.03])) == pytest.approx(0.0)

    def test_half_wins(self):
        assert win_rate(np.array([0.01, -0.01, 0.02, -0.02])) == pytest.approx(0.5)

    def test_zero_pnl_not_counted_as_win(self):
        # Flat trades (pnl == 0) are not strictly positive.
        assert win_rate(np.array([0.0, 0.0, 0.01])) == pytest.approx(1 / 3)


class TestProfitFactor:
    def test_infinity_when_no_losses(self):
        import math
        assert math.isinf(profit_factor(np.array([0.01, 0.02, 0.03])))

    def test_zero_when_no_gains_and_no_losses(self):
        assert profit_factor(np.array([0.0, 0.0])) == 0.0

    def test_known_ratio(self):
        # Gains: 0.10 + 0.20 = 0.30; Losses: |-0.10| + |-0.05| = 0.15 -> PF = 2.0
        series = np.array([0.10, 0.20, -0.10, -0.05])
        assert profit_factor(series) == pytest.approx(2.0)

    def test_less_than_one_for_net_loser(self):
        series = np.array([-0.10, -0.20, 0.05])
        assert profit_factor(series) < 1.0


class TestStabilityScore:
    def test_zero_for_empty_input(self):
        assert stability_score(np.array([])) == 0.0

    def test_zero_when_mean_sharpe_is_zero(self):
        assert stability_score(np.array([0.5, -0.5])) == 0.0

    def test_one_for_constant_positive_sharpe(self):
        # Identical Sharpe values -> std = 0 -> stability = 1.
        assert stability_score(np.array([1.0, 1.0, 1.0])) == pytest.approx(1.0)

    def test_between_zero_and_one(self):
        scores = stability_score(np.array([0.5, 1.0, 0.8, 1.2, 0.6]))
        assert 0.0 <= scores <= 1.0


class TestMonteCarloVectorized:
    """Verify the vectorized _summarize produces the same outputs as the
    (slower) per-path loop it replaced, for a small deterministic example."""

    def test_shape_and_basic_sanity(self):
        mc = StressMonteCarlo(n_simulations=50, seed=7)
        result = mc.run(np.array([0.01, 0.02, -0.005, 0.015]))
        assert mc.last_paths is not None
        assert mc.last_paths.shape == (50, 4)
        assert result.n_simulations == 50
        # Sharpe percentiles must be ordered.
        assert result.sharpe_p5 <= result.sharpe_p50 <= result.sharpe_p95
        # Drawdown p95 must be in [0, 1].
        assert 0.0 <= result.max_drawdown_p95 <= 1.0

    def test_stress_run_increases_tail_risk(self):
        """Stress run overweights bad paths, so CVaR should be <= normal run."""
        pnl = np.array([0.02, -0.05, 0.01, -0.03, 0.015] * 20)
        mc = StressMonteCarlo(n_simulations=200, seed=99)
        normal = mc.run(pnl)
        stress = mc.run_stress(pnl, stress_weight=5.0)
        # With bad paths overweighted, median terminal return should be lower.
        assert stress.terminal_return_p50 <= normal.terminal_return_p50

    def test_cagr_exposed_on_walk_forward_result(self):
        """WalkForwardResult.cagr must be a finite float (regression test for
        the newly added field)."""
        class FakeLoader:
            def load_ohlcv(self, ticker, start, end, timespan="day"):
                return pd.DataFrame(
                    [
                        {
                            "date": date(2022, 1, 9),
                            "open": 100, "high": 110, "low": 99,
                            "close": 108, "volume": 1, "vwap": 0,
                        },
                        {
                            "date": date(2022, 1, 10),
                            "open": 108, "high": 115, "low": 107,
                            "close": 113, "volume": 1, "vwap": 0,
                        },
                    ]
                )

        from ai_trader.backtesting.engine import WalkForwardConfig, WalkForwardEngine
        from ai_trader.backtesting.replay import EventReplay, ReplayEvent
        from ai_trader.domain.events import (
            Chamber,
            CongressionalTrade,
            TransactionType,
        )
        from decimal import Decimal

        trade_event = CongressionalTrade(
            member_name="Jane Senator",
            chamber=Chamber.SENATE,
            ticker="MSFT",
            issuer="Microsoft Corp.",
            sector="technology",
            transaction_date=date(2022, 1, 1),
            disclosure_date=date(2022, 1, 8),
            transaction_type=TransactionType.PURCHASE,
            amount_low=Decimal("100001"),
            amount_high=Decimal("250000"),
        )
        replay = EventReplay([ReplayEvent(event_type="congressional_trade", congressional_trade=trade_event)])
        engine = WalkForwardEngine(data_loader=FakeLoader(), replay=replay)
        result = engine.run(
            None,
            date(2022, 1, 1),
            date(2022, 1, 15),
            WalkForwardConfig(
                train_window_days=5,
                test_window_days=8,
                step_days=8,
                signal_threshold=0.01,
                cash_fraction=1.0,
            ),
        )
        import math
        assert isinstance(result.cagr, float)
        assert not math.isnan(result.cagr)
