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
from ai_trader.backtesting.metrics import max_drawdown, sharpe_ratio
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


def _trade(disclosure: date) -> CongressionalTrade:
    return CongressionalTrade(
        member_name="Jane Senator",
        chamber=Chamber.SENATE,
        ticker="MSFT",
        issuer="Microsoft Corp.",
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
            ReplayEvent(event_type="congressional_trade", congressional_trade=_trade(date(2022, 1, 10))),
        ]
    )

    assert replay.available_as_of(date(2022, 1, 9), ticker="MSFT") == ()
    assert len(replay.available_as_of(date(2022, 1, 10), ticker="MSFT")) == 1


def test_event_replay_jsonl_roundtrip(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    event = ReplayEvent(event_type="congressional_trade", congressional_trade=_trade(date(2022, 1, 10)))
    path.write_text(event.model_dump_json() + "\n", encoding="utf-8")

    replay = EventReplay.from_jsonl(path)

    assert replay.events[0].ticker == "MSFT"
    assert replay.events[0].effective_date == date(2022, 1, 10)


def test_walk_forward_engine_uses_replay_events_without_lookahead():
    class FakeLoader:
        def load_ohlcv(self, ticker, start, end, timespan="day"):
            return pd.DataFrame(
                [
                    {"date": date(2022, 1, day), "open": 100 + day, "high": 0, "low": 0, "close": 101 + day, "volume": 1, "vwap": 0}
                    for day in range(6, 14)
                ]
            )

    replay = EventReplay(
        [
            ReplayEvent(event_type="congressional_trade", congressional_trade=_trade(date(2022, 1, 8))),
            ReplayEvent(event_type="congressional_trade", congressional_trade=_trade(date(2022, 1, 20))),
        ]
    )
    engine = WalkForwardEngine(data_loader=FakeLoader(), replay=replay)

    result = engine.run(
        ["MSFT"],
        date(2022, 1, 1),
        date(2022, 1, 15),
        WalkForwardConfig(
            train_window_days=5,
            test_window_days=8,
            step_days=8,
            signal_threshold=0.01,
            max_holding_days=2,
        ),
    )

    assert result.metadata["mode"] == "event_replay"
    assert len(result.trades) == 1
    assert result.trades[0].entry_date == date(2022, 1, 9)


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
