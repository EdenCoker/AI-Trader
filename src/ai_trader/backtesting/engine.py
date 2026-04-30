from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ai_trader.backtesting.data_loader import PolygonDataLoader
from ai_trader.backtesting.metrics import max_drawdown, sharpe_ratio, stability_score


@dataclass(frozen=True)
class WalkForwardConfig:
    train_window_days: int = 252
    test_window_days: int = 63
    step_days: int = 21
    anchored: bool = False
    narrative_enabled: bool = False
    rag_enabled: bool = False


class TradeRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    pnl_pct: float


class WalkForwardWindow(BaseModel):
    model_config = ConfigDict(frozen=True)

    train_start: date
    train_end: date
    test_start: date
    test_end: date
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    trades: tuple[TradeRecord, ...] = ()


class WalkForwardResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    tickers: tuple[str, ...]
    start: date
    end: date
    windows: tuple[WalkForwardWindow, ...]
    sharpe: float
    max_drawdown: float
    stability: float
    trades: tuple[TradeRecord, ...] = ()
    metadata: dict = Field(default_factory=dict)


class WalkForwardEngine:
    def __init__(self, *, data_loader: PolygonDataLoader | None = None) -> None:
        self._data_loader = data_loader or PolygonDataLoader()

    def run(
        self,
        tickers: list[str],
        start: date,
        end: date,
        config: WalkForwardConfig,
    ) -> WalkForwardResult:
        windows = []
        all_returns: list[float] = []
        for window in self.windows(start, end, config):
            returns_by_ticker = []
            for ticker in tickers:
                frame = self._data_loader.load_ohlcv(ticker, window.test_start, window.test_end)
                if frame.empty or "close" not in frame:
                    continue
                close = frame["close"].to_numpy(dtype=float)
                if close.size < 2:
                    continue
                returns_by_ticker.append(np.diff(close) / close[:-1])
            if returns_by_ticker:
                min_len = min(len(values) for values in returns_by_ticker)
                stacked = np.vstack([values[:min_len] for values in returns_by_ticker])
                window_returns = np.mean(stacked, axis=0)
            else:
                window_returns = np.asarray([], dtype=float)
            equity = np.cumprod(1 + window_returns) if window_returns.size else np.asarray([1.0])
            all_returns.extend(window_returns.tolist())
            windows.append(
                window.model_copy(
                    update={
                        "sharpe": sharpe_ratio(window_returns),
                        "max_drawdown": max_drawdown(equity),
                    }
                )
            )

        returns = np.asarray(all_returns, dtype=float)
        equity = np.cumprod(1 + returns) if returns.size else np.asarray([1.0])
        window_sharpes = np.asarray([window.sharpe for window in windows], dtype=float)
        return WalkForwardResult(
            tickers=tuple(tickers),
            start=start,
            end=end,
            windows=tuple(windows),
            sharpe=sharpe_ratio(returns),
            max_drawdown=max_drawdown(equity),
            stability=stability_score(window_sharpes),
            trades=(),
            metadata={"mode": "price_replay_baseline"},
        )

    @staticmethod
    def windows(start: date, end: date, config: WalkForwardConfig) -> tuple[WalkForwardWindow, ...]:
        windows: list[WalkForwardWindow] = []
        cursor = start + timedelta(days=config.train_window_days)
        while cursor + timedelta(days=config.test_window_days) <= end:
            train_start = start if config.anchored else cursor - timedelta(days=config.train_window_days)
            train_end = cursor - timedelta(days=1)
            test_start = cursor
            test_end = cursor + timedelta(days=config.test_window_days - 1)
            windows.append(
                WalkForwardWindow(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )
            cursor += timedelta(days=config.step_days)
        return tuple(windows)

