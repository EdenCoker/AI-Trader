from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai_trader.backtesting.metrics import cvar


@dataclass(frozen=True)
class MonteCarloResult:
    n_simulations: int
    sharpe_p5: float
    sharpe_p50: float
    sharpe_p95: float
    max_drawdown_p95: float
    cvar_95: float
    prob_ruin: float
    terminal_return_p5: float = 0.0
    terminal_return_p50: float = 0.0
    terminal_return_p95: float = 0.0


class StressMonteCarlo:
    def __init__(self, n_simulations: int = 10_000, seed: int = 42) -> None:
        self.n_simulations = n_simulations
        self.seed = seed
        self.last_paths: np.ndarray | None = None

    def run(self, pnl_series: np.ndarray) -> MonteCarloResult:
        pnl = np.asarray(pnl_series, dtype=float)
        if pnl.size == 0:
            return MonteCarloResult(self.n_simulations, 0, 0, 0, 0, 0, 0)
        rng = np.random.default_rng(self.seed)
        paths = rng.choice(pnl, size=(self.n_simulations, pnl.size), replace=True)
        return self._summarize(paths)

    def run_stress(self, pnl_series: np.ndarray, stress_weight: float = 3.0) -> MonteCarloResult:
        pnl = np.asarray(pnl_series, dtype=float)
        if pnl.size == 0:
            return MonteCarloResult(self.n_simulations, 0, 0, 0, 0, 0, 0)
        cutoff = np.quantile(pnl, 0.10)
        weights = np.ones_like(pnl, dtype=float)
        weights[pnl <= cutoff] *= stress_weight
        weights = weights / np.sum(weights)
        rng = np.random.default_rng(self.seed)
        paths = rng.choice(pnl, size=(self.n_simulations, pnl.size), replace=True, p=weights)
        return self._summarize(paths)

    def _summarize(self, paths: np.ndarray) -> MonteCarloResult:
        self.last_paths = paths
        n_steps = paths.shape[1] if paths.ndim > 1 else 0

        # Vectorized Sharpe: avoid a Python-level loop over 10 000+ paths.
        # Matches sharpe_ratio() exactly: annualised excess return / excess std, ddof=1.
        if n_steps < 2:
            sharpes = np.zeros(len(paths))
        else:
            _rf = 0.05 / 252  # daily risk-free rate (matches metrics.sharpe_ratio defaults)
            excess = paths - _rf
            mean_exc = np.mean(excess, axis=1)
            std_exc = np.std(excess, axis=1, ddof=1)
            sharpes = np.where(std_exc > 0, mean_exc / std_exc * np.sqrt(252), 0.0)

        # Vectorized max-drawdown across all paths simultaneously.
        equity = np.cumprod(1 + paths, axis=1)
        running_max = np.maximum.accumulate(equity, axis=1)
        dd_matrix = (running_max - equity) / np.maximum(running_max, 1e-12)
        drawdowns = np.max(dd_matrix, axis=1)

        terminal_returns = equity[:, -1] - 1
        return MonteCarloResult(
            n_simulations=self.n_simulations,
            sharpe_p5=float(np.percentile(sharpes, 5)),
            sharpe_p50=float(np.percentile(sharpes, 50)),
            sharpe_p95=float(np.percentile(sharpes, 95)),
            max_drawdown_p95=float(np.percentile(drawdowns, 95)),
            cvar_95=cvar(terminal_returns, alpha=0.05),
            prob_ruin=float(np.mean(drawdowns > 0.50)),
            terminal_return_p5=float(np.percentile(terminal_returns, 5)),
            terminal_return_p50=float(np.percentile(terminal_returns, 50)),
            terminal_return_p95=float(np.percentile(terminal_returns, 95)),
        )
