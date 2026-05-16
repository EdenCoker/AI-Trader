from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from math import floor
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ai_trader.backtesting.data_loader import PolygonDataLoader
from ai_trader.backtesting.metrics import cagr, max_drawdown, sharpe_ratio, stability_score
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
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    starting_balance: float = 10_000.0
    cash_fraction: float = 0.02
    fractional_shares: bool = False


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
    exit_reason: str = "time_exit"
    conviction: float = 1.0
    size_multiplier: float = 1.0
    quantity: float = 0.0
    notional: float = 0.0
    pnl_amount: float = 0.0
    balance_before: float = 0.0
    balance_after: float = 0.0
    account_return: float = 0.0


class WalkForwardWindow(BaseModel):
    model_config = ConfigDict(frozen=True)

    train_start: date
    train_end: date
    test_start: date
    test_end: date
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    trades: tuple[TradeRecord, ...] = ()
    starting_balance: float = 0.0
    ending_balance: float = 0.0


class WalkForwardResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    tickers: tuple[str, ...]
    start: date
    end: date
    windows: tuple[WalkForwardWindow, ...]
    sharpe: float
    max_drawdown: float
    stability: float
    cagr: float = 0.0
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
        tickers: list[str] | None,
        start: date,
        end: date,
        config: WalkForwardConfig,
    ) -> WalkForwardResult:
        replay = self._replay
        if replay is None and config.events_file is not None:
            replay = EventReplay.from_jsonl(config.events_file)

        ticker_universe = _normalize_tickers(tickers)
        ticker_source = "input"
        if not ticker_universe and replay is not None:
            ticker_universe = list(replay.tickers(start=start, end=end))
            ticker_source = "events"
        if not ticker_universe:
            raise ValueError(
                "No tickers were provided and no replay events were available to infer them"
            )

        windows = []
        all_returns: list[float] = []
        all_trades: list[TradeRecord] = []
        balance = float(config.starting_balance)
        for window in self.windows(start, end, config):
            returns_by_ticker = []
            window_trades: list[TradeRecord] = []
            window_starting_balance = balance
            for ticker in ticker_universe:
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
                window_trades, sized_returns, balance = _apply_balance_sizing(
                    window_trades,
                    balance,
                    config,
                )
                window_returns = np.asarray(sized_returns, dtype=float)
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
                        "starting_balance": window_starting_balance,
                        "ending_balance": balance,
                    }
                )
            )

        returns = np.asarray(all_returns, dtype=float)
        equity = np.cumprod(1 + returns) if returns.size else np.asarray([1.0])
        window_sharpes = np.asarray([window.sharpe for window in windows], dtype=float)
        return WalkForwardResult(
            tickers=tuple(ticker_universe),
            start=start,
            end=end,
            windows=tuple(windows),
            sharpe=sharpe_ratio(returns),
            max_drawdown=max_drawdown(equity),
            stability=stability_score(window_sharpes),
            cagr=cagr(returns),
            trades=tuple(all_trades),
            metadata={
                "mode": "event_replay" if replay is not None else "price_replay_baseline",
                "ticker_source": ticker_source,
                "starting_balance": config.starting_balance,
                "ending_balance": balance,
                "cash_fraction": config.cash_fraction,
                "fractional_shares": config.fractional_shares,
            },
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
        # Collect every unique signal date that falls inside the test window.
        # events_on(test_start) was previously computed separately, but those
        # events are already captured by the loop below (test_start is within
        # [test_start, test_end]), so the pre-pass was redundant.
        event_days: set[date] = {
            event.effective_date
            for event in replay.events
            if event.ticker == ticker.upper()
            and window.test_start <= event.effective_date <= window.test_end
        }
        for event_day in sorted(event_days):
            available_events = replay.available_as_of(
                event_day, ticker=ticker, start=window.train_start
            )
            congressional, institutional = EventReplay.split(available_events)
            bundle = self._scorer.build_bundle(
                ticker=ticker,
                as_of=event_day,
                congressional_trades=congressional,
                institutional_changes=institutional,
            )
            if (
                bundle.direction is SignalDirection.NEUTRAL
                or bundle.conviction < config.signal_threshold
            ):
                continue
            horizon = min(
                config.max_holding_days,
                max(
                    (signal.horizon_days for signal in bundle.signals),
                    default=config.max_holding_days,
                ),
            )
            trade = _simulate_trade(
                frame,
                event_day,
                bundle.direction,
                horizon,
                len(bundle.signals),
                stop_loss_pct=config.stop_loss_pct,
                take_profit_pct=config.take_profit_pct,
            )
            if trade is not None:
                trades.append(
                    trade.model_copy(
                        update={
                            "ticker": ticker.upper(),
                            "conviction": bundle.conviction,
                            "size_multiplier": 1.0,
                        }
                    )
                )
        return trades

    @staticmethod
    def windows(start: date, end: date, config: WalkForwardConfig) -> tuple[WalkForwardWindow, ...]:
        windows: list[WalkForwardWindow] = []
        cursor = start + timedelta(days=config.train_window_days)
        while cursor + timedelta(days=config.test_window_days) <= end:
            train_start = (
                start if config.anchored else cursor - timedelta(days=config.train_window_days)
            )
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


def _normalize_tickers(tickers: Iterable[str] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_ticker in tickers or ():
        ticker = str(raw_ticker).strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        normalized.append(ticker)
    return normalized


def _apply_balance_sizing(
    trades: list[TradeRecord],
    starting_balance: float,
    config: WalkForwardConfig,
) -> tuple[list[TradeRecord], list[float], float]:
    balance = starting_balance
    sized_trades: list[TradeRecord] = []
    account_returns: list[float] = []
    for trade in sorted(trades, key=lambda item: (item.entry_date, item.ticker)):
        sized = _size_trade_from_balance(trade, balance, config)
        if sized is None:
            continue
        sized_trades.append(sized)
        account_returns.append(sized.account_return)
        balance = sized.balance_after
    return sized_trades, account_returns, balance


def _size_trade_from_balance(
    trade: TradeRecord,
    balance: float,
    config: WalkForwardConfig,
) -> TradeRecord | None:
    if balance <= 0 or trade.entry_price <= 0:
        return None

    scaled_fraction = min(
        1.0,
        max(0.0, config.cash_fraction)
        * max(0.0, trade.conviction)
        * max(0.0, trade.size_multiplier),
    )
    target_notional = balance * scaled_fraction
    if target_notional <= 0:
        return None

    raw_quantity = target_notional / trade.entry_price
    quantity = raw_quantity if config.fractional_shares else float(floor(raw_quantity))
    minimum_quantity = 0.0001 if config.fractional_shares else 1.0
    if quantity < minimum_quantity:
        return None

    notional = quantity * trade.entry_price
    pnl_amount = notional * trade.pnl_pct
    balance_after = balance + pnl_amount
    account_return = pnl_amount / balance
    return trade.model_copy(
        update={
            "quantity": quantity,
            "notional": notional,
            "pnl_amount": pnl_amount,
            "balance_before": balance,
            "balance_after": balance_after,
            "account_return": account_return,
        }
    )


def _simulate_trade(
    frame: pd.DataFrame,
    signal_date: date,
    direction: SignalDirection,
    horizon_days: int,
    signal_count: int,
    *,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
) -> TradeRecord | None:
    rows = frame.reset_index(drop=True)
    future_indices = [
        idx for idx, value in enumerate(rows["date"]) if _as_date(value) > signal_date
    ]
    if not future_indices:
        return None
    entry_idx = future_indices[0]
    exit_idx = min(entry_idx + max(1, horizon_days), len(rows) - 1)
    if exit_idx <= entry_idx:
        return None

    entry = rows.iloc[entry_idx]
    entry_price = float(entry["open"])
    exit_price = float(rows.iloc[exit_idx]["close"])
    exit_date = _as_date(rows.iloc[exit_idx]["date"])
    exit_reason = "time_exit"
    for idx in range(entry_idx, exit_idx + 1):
        row = rows.iloc[idx]
        low = float(row["low"]) if "low" in row else float(row["close"])
        high = float(row["high"]) if "high" in row else float(row["close"])
        current_date = _as_date(row["date"])
        risk_exit = _risk_exit(
            entry_price=entry_price,
            direction=direction,
            low=low,
            high=high,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
        if risk_exit is None:
            continue
        exit_price, exit_reason = risk_exit
        exit_date = current_date
        break

    raw_return = (exit_price - entry_price) / entry_price
    pnl = raw_return * direction.multiplier
    return TradeRecord(
        ticker="",
        entry_date=_as_date(entry["date"]),
        exit_date=exit_date,
        entry_price=entry_price,
        exit_price=exit_price,
        pnl_pct=pnl,
        direction=direction,
        signal_count=signal_count,
        exit_reason=exit_reason,
    )


def _risk_exit(
    *,
    entry_price: float,
    direction: SignalDirection,
    low: float,
    high: float,
    stop_loss_pct: float | None,
    take_profit_pct: float | None,
) -> tuple[float, str] | None:
    if direction is SignalDirection.LONG:
        if stop_loss_pct is not None:
            stop_price = entry_price * (1.0 - stop_loss_pct)
            if low <= stop_price:
                return stop_price, "stop_loss"
        if take_profit_pct is not None:
            target_price = entry_price * (1.0 + take_profit_pct)
            if high >= target_price:
                return target_price, "take_profit"
    if direction is SignalDirection.SHORT:
        if stop_loss_pct is not None:
            stop_price = entry_price * (1.0 + stop_loss_pct)
            if high >= stop_price:
                return stop_price, "stop_loss"
        if take_profit_pct is not None:
            target_price = entry_price * (1.0 - take_profit_pct)
            if low <= target_price:
                return target_price, "take_profit"
    return None


def _as_date(value) -> date:
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()
