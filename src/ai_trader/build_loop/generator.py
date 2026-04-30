from __future__ import annotations

from ai_trader.backtesting.engine import WalkForwardResult
from ai_trader.self_improvement.proposal import Proposal


class ChangeGenerator:
    def generate(
        self,
        backtest_result: WalkForwardResult,
        max_proposals: int = 3,
    ) -> list[Proposal]:
        if not backtest_result.windows:
            return []
        # The LLM-backed implementation will use bottom-quartile windows here.
        # Returning no proposals is the conservative default for unattended runs.
        return []

