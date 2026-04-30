from datetime import date
from pathlib import Path

import httpx
import numpy as np
import pytest
from pydantic import SecretStr

from ai_trader.backtesting.data_loader import PolygonDataLoader
from ai_trader.backtesting.engine import WalkForwardConfig, WalkForwardEngine
from ai_trader.backtesting.metrics import max_drawdown, sharpe_ratio
from ai_trader.backtesting.monte_carlo import StressMonteCarlo
from ai_trader.config import AppSettings


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

