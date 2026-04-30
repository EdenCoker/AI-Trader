from __future__ import annotations

from datetime import date, datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from ai_trader.backtesting.engine import WalkForwardConfig, WalkForwardEngine
from ai_trader.build_loop.backtest_gate import BacktestGate
from ai_trader.build_loop.critic import SelfCritic
from ai_trader.build_loop.generator import ChangeGenerator
from ai_trader.build_loop.test_runner import TestRunner
from ai_trader.self_improvement.git_store import GitProposalStore


class BuildLoopReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    started_at: datetime = Field(default_factory=datetime.utcnow)
    proposals_generated: int = 0
    proposals_approved_by_critic: int = 0
    proposals_passed_tests: int = 0
    proposals_passed_backtest: int = 0
    prs_opened: int = 0
    errors: tuple[str, ...] = ()


class BuildLoop:
    def __init__(
        self,
        *,
        engine: WalkForwardEngine | None = None,
        generator: ChangeGenerator | None = None,
        critic: SelfCritic | None = None,
        test_runner: TestRunner | None = None,
        backtest_gate: BacktestGate | None = None,
        store: GitProposalStore | None = None,
    ) -> None:
        self._engine = engine or WalkForwardEngine()
        self._generator = generator or ChangeGenerator()
        self._critic = critic or SelfCritic()
        self._test_runner = test_runner or TestRunner()
        self._backtest_gate = backtest_gate or BacktestGate()
        self._store = store or GitProposalStore()

    def run_once(
        self,
        tickers: list[str],
        backtest_start: date,
        backtest_end: date,
        max_proposals: int = 3,
    ) -> BuildLoopReport:
        errors: list[str] = []
        approved = 0
        passed_tests = 0
        passed_backtest = 0
        prs_opened = 0

        baseline = self._engine.run(tickers, backtest_start, backtest_end, WalkForwardConfig())
        proposals = self._generator.generate(baseline, max_proposals=max_proposals)[:max_proposals]

        for proposal in proposals:
            try:
                critique = self._critic.critique(proposal)
                if not critique.approved:
                    continue
                approved += 1

                test_result = self._test_runner.run()
                if not test_result.success:
                    continue
                passed_tests += 1

                gate_result = self._backtest_gate.validate(
                    proposal,
                    baseline,
                    tickers,
                    backtest_start,
                    backtest_end,
                )
                if not gate_result.passed:
                    continue
                passed_backtest += 1

                self._store.submit(proposal)
                prs_opened += 1
            except Exception as exc:
                errors.append(f"{proposal.proposal_id}: {exc}")

        return BuildLoopReport(
            proposals_generated=len(proposals),
            proposals_approved_by_critic=approved,
            proposals_passed_tests=passed_tests,
            proposals_passed_backtest=passed_backtest,
            prs_opened=prs_opened,
            errors=tuple(errors),
        )

