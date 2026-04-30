from __future__ import annotations

import numpy as np


def sharpe_ratio(
    returns: np.ndarray,
    risk_free_annual: float = 0.05,
    periods_per_year: int = 252,
) -> float:
    returns = _as_array(returns)
    if returns.size < 2:
        return 0.0
    excess = returns - risk_free_annual / periods_per_year
    std = float(np.std(excess, ddof=1))
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: np.ndarray) -> float:
    equity_curve = _as_array(equity_curve)
    if equity_curve.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (running_max - equity_curve) / np.maximum(running_max, 1e-12)
    return float(np.max(drawdowns))


def calmar_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    returns = _as_array(returns)
    if returns.size == 0:
        return 0.0
    equity = np.cumprod(1 + returns)
    drawdown = max_drawdown(equity)
    if drawdown == 0:
        return 0.0
    annual_return = float(equity[-1] ** (periods_per_year / returns.size) - 1)
    return annual_return / drawdown


def sortino_ratio(
    returns: np.ndarray,
    risk_free_annual: float = 0.05,
    periods_per_year: int = 252,
) -> float:
    returns = _as_array(returns)
    if returns.size < 2:
        return 0.0
    excess = returns - risk_free_annual / periods_per_year
    downside = excess[excess < 0]
    if downside.size < 2:
        return 0.0
    downside_std = float(np.std(downside, ddof=1))
    if downside_std == 0:
        return 0.0
    return float(np.mean(excess) / downside_std * np.sqrt(periods_per_year))


def cvar(returns: np.ndarray, alpha: float = 0.05) -> float:
    returns = _as_array(returns)
    if returns.size == 0:
        return 0.0
    cutoff = np.quantile(returns, alpha)
    tail = returns[returns <= cutoff]
    if tail.size == 0:
        return 0.0
    return float(np.mean(tail))


def win_rate(pnl_series: np.ndarray) -> float:
    pnl_series = _as_array(pnl_series)
    if pnl_series.size == 0:
        return 0.0
    return float(np.mean(pnl_series > 0))


def profit_factor(pnl_series: np.ndarray) -> float:
    pnl_series = _as_array(pnl_series)
    gains = float(np.sum(pnl_series[pnl_series > 0]))
    losses = abs(float(np.sum(pnl_series[pnl_series < 0])))
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def stability_score(window_sharpes: np.ndarray) -> float:
    window_sharpes = _as_array(window_sharpes)
    if window_sharpes.size == 0:
        return 0.0
    mean = abs(float(np.mean(window_sharpes)))
    std = float(np.std(window_sharpes))
    if mean == 0:
        return 0.0
    return float(np.clip(1 - (std / mean), 0, 1))


def _as_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)

