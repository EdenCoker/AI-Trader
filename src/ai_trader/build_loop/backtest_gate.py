from __future__ import annotations

from datetime import date
from typing import Callable

from pydantic import BaseModel, ConfigDict

from ai_trader.backtesting.engine import WalkForwardConfig, WalkForwardEngine, WalkForwardResult
from ai_trader.self_improvement.proposal import Proposal


class GateResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    sharpe_delta: float
    max_drawdown: float
    reason: str


EngineFactory = Callable[[], WalkForwardEngine]


class BacktestGate:
    MIN_SHARPE_DELTA = -0.10
    MAX_DRAWDOWN_LIMIT = 0.30

    def __init__(self, *, engine_factory: EngineFactory | None = None) -> None:
        self._engine_factory = engine_factory or WalkForwardEngine

    def validate(
        self,
        proposal: Proposal,
        baseline: WalkForwardResult,
        tickers: list[str],
        start: date,
        end: date,
    ) -> GateResult:
        result = self._engine_factory().run(tickers, start, end, WalkForwardConfig())
        sharpe_delta = result.sharpe - baseline.sharpe
        if sharpe_delta < self.MIN_SHARPE_DELTA:
            return GateResult(
                passed=False,
                sharpe_delta=sharpe_delta,
                max_drawdown=result.max_drawdown,
                reason="Sharpe degradation exceeds limit",
            )
        if result.max_drawdown > self.MAX_DRAWDOWN_LIMIT:
            return GateResult(
                passed=False,
                sharpe_delta=sharpe_delta,
                max_drawdown=result.max_drawdown,
                reason="Max drawdown exceeds limit",
            )
        return GateResult(
            passed=True,
            sharpe_delta=sharpe_delta,
            max_drawdown=result.max_drawdown,
            reason="passed",
        )

