from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ai_trader.backtesting.data_loader import PolygonDataLoader
from ai_trader.backtesting.metrics import max_drawdown, sharpe_ratio, stability_score
from ai_trader.backtesting.replay import EventReplay
from ai_trader.domain.signals import SignalDirection
from ai_trader.smart_money.scoring import SmartMoneyScorer


@dataclass(frozen=True)
class WalkForwardConfig:
    train_window_days: int = 252
    test_window_days: int = 63
    step_days: int = 21
    anchored: bool = False
    narrative_enabled: bool = False
    rag_enabled: bool = False
    events_file: Path | None = None
    signal_threshold: float = 0.10
    max_holding_days: int = 63


class TradeRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    pnl_pct: float
    direction: SignalDirection = SignalDirection.NEUTRAL
    signal_count: int = 0


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
    def __init__(
        self,
        *,
        data_loader: PolygonDataLoader | None = None,
        replay: EventReplay | None = None,
        scorer: SmartMoneyScorer | None = None,
    ) -> None:
        self._data_loader = data_loader or PolygonDataLoader()
        self._replay = replay
        self._scorer = scorer or SmartMoneyScorer()

    def run(
        self,
        tickers: list[str],
        start: date,
        end: date,
        config: WalkForwardConfig,
    ) -> WalkForwardResult:
        replay = self._replay
        if replay is None and config.events_file is not None:
            replay = EventReplay.from_jsonl(config.events_file)

        windows = []
        all_returns: list[float] = []
        all_trades: list[TradeRecord] = []
        for window in self.windows(start, end, config):
            returns_by_ticker = []
            window_trades: list[TradeRecord] = []
            for ticker in tickers:
                frame = self._data_loader.load_ohlcv(ticker, window.test_start, window.test_end)
                price_returns = _price_returns(frame)
                if price_returns.size:
                    returns_by_ticker.append(price_returns)
                if replay is not None:
                    window_trades.extend(
                        self._event_trades_for_window(
                            ticker=ticker,
                            frame=frame,
                            replay=replay,
                            window=window,
                            config=config,
                        )
                    )

            if window_trades:
                window_returns = np.asarray([trade.pnl_pct for trade in window_trades], dtype=float)
            elif returns_by_ticker:
                min_len = min(len(values) for values in returns_by_ticker)
                stacked = np.vstack([values[:min_len] for values in returns_by_ticker])
                window_returns = np.mean(stacked, axis=0)
            else:
                window_returns = np.asarray([], dtype=float)
            equity = np.cumprod(1 + window_returns) if window_returns.size else np.asarray([1.0])
            all_returns.extend(window_returns.tolist())
            all_trades.extend(window_trades)
            windows.append(
                window.model_copy(
                    update={
                        "sharpe": sharpe_ratio(window_returns),
                        "max_drawdown": max_drawdown(equity),
                        "trades": tuple(window_trades),
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
            trades=tuple(all_trades),
            metadata={"mode": "event_replay" if replay is not None else "price_replay_baseline"},
        )

    def _event_trades_for_window(
        self,
        *,
        ticker: str,
        frame: pd.DataFrame,
        replay: EventReplay,
        window: WalkForwardWindow,
        config: WalkForwardConfig,
    ) -> list[TradeRecord]:
        if frame.empty or not {"date", "open", "close"}.issubset(frame.columns):
            return []
        trades: list[TradeRecord] = []
        frame = frame.sort_values("date").reset_index(drop=True)
        event_days = sorted({event.effective_date for event in replay.events_on(window.test_start, ticker=ticker)})
        for event in replay.events:
            if event.ticker == ticker.upper() and window.test_start <= event.effective_date <= window.test_end:
                event_days.append(event.effective_date)
        for event_day in sorted(set(event_days)):
            available_events = replay.available_as_of(event_day, ticker=ticker, start=window.train_start)
            congressional, institutional = EventReplay.split(available_events)
            bundle = self._scorer.build_bundle(
                ticker=ticker,
                as_of=event_day,
                congressional_trades=congressional,
                institutional_changes=institutional,
            )
            if bundle.direction is SignalDirection.NEUTRAL or bundle.conviction < config.signal_threshold:
                continue
            horizon = min(
                config.max_holding_days,
                max((signal.horizon_days for signal in bundle.signals), default=config.max_holding_days),
            )
            trade = _simulate_trade(frame, event_day, bundle.direction, horizon, len(bundle.signals))
            if trade is not None:
                trades.append(trade.model_copy(update={"ticker": ticker.upper()}))
        return trades

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


def _price_returns(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty or "close" not in frame:
        return np.asarray([], dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    if close.size < 2:
        return np.asarray([], dtype=float)
    return np.diff(close) / close[:-1]


def _simulate_trade(
    frame: pd.DataFrame,
    signal_date: date,
    direction: SignalDirection,
    horizon_days: int,
    signal_count: int,
) -> TradeRecord | None:
    rows = frame.reset_index(drop=True)
    future_indices = [idx for idx, value in enumerate(rows["date"]) if _as_date(value) > signal_date]
    if not future_indices:
        return None
    entry_idx = future_indices[0]
    exit_idx = min(entry_idx + max(1, horizon_days), len(rows) - 1)
    if exit_idx <= entry_idx:
        return None

    entry = rows.iloc[entry_idx]
    exit_row = rows.iloc[exit_idx]
    entry_price = float(entry["open"])
    exit_price = float(exit_row["close"])
    raw_return = (exit_price - entry_price) / entry_price
    pnl = raw_return * direction.multiplier
    return TradeRecord(
        ticker="",
        entry_date=_as_date(entry["date"]),
        exit_date=_as_date(exit_row["date"]),
        entry_price=entry_price,
        exit_price=exit_price,
        pnl_pct=pnl,
        direction=direction,
        signal_count=signal_count,
    )


def _as_date(value) -> date:
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()
