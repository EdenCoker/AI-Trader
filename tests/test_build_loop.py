from datetime import date
from pathlib import Path
import json
import subprocess

import pytest

from ai_trader.backtesting.engine import WalkForwardResult
from ai_trader.build_loop.backtest_gate import BacktestGate
from ai_trader.build_loop.critic import SelfCritic
from ai_trader.build_loop.loop import BuildLoop
from ai_trader.build_loop.test_runner import TestRunner, TestResult
from ai_trader.domain.signals import SignalDirection
from ai_trader.intelligence.trade_plan import TradePlan
from ai_trader.self_improvement.proposal import PromptProposal, TradeOutcome


class StubLLM:
    def __init__(self, output):
        self.output = output
        self.calls = 0

    def complete(self, prompt, *, system=None, model=None, temperature=0.2, max_tokens=None):
        self.calls += 1
        return self.output

    def chat(self, messages, *, model=None, temperature=0.2, max_tokens=None):
        raise NotImplementedError


def _baseline(sharpe: float = 1.0, drawdown: float = 0.1) -> WalkForwardResult:
    return WalkForwardResult(
        tickers=("MSFT",),
        start=date(2022, 1, 1),
        end=date(2022, 12, 31),
        windows=(),
        sharpe=sharpe,
        max_drawdown=drawdown,
        stability=1.0,
    )


def _outcome() -> TradeOutcome:
    return TradeOutcome(
        trade_plan=TradePlan(
            ticker="MSFT",
            as_of=date(2026, 1, 1),
            direction=SignalDirection.LONG,
            conviction=0.6,
            size_multiplier=1.0,
            holding_period_days=10,
            exit_trigger="x",
        ),
        ticker="MSFT",
        actual_entry_price=100,
        actual_exit_price=90,
        actual_exit_date=date(2026, 1, 10),
        exit_reason="stop_hit",
        pnl_pct=-0.1,
    )


def _proposal() -> PromptProposal:
    return PromptProposal(
        source_outcome=_outcome(),
        target_module="intelligence.narrative:SYSTEM_PROMPT",
        current_text="old",
        proposed_text="new",
        rationale="r",
        expected_improvement="e",
    )


def test_self_critic_rejects_safety_patterns_without_llm():
    proposal = PromptProposal.model_construct(
        proposal_id="x",
        source_outcome=_outcome(),
        target_module="config:allow_live_trading",
        current_text="old",
        proposed_text="new",
        rationale="r",
        expected_improvement="e",
    )
    llm = StubLLM('{"approved":true,"confidence":1,"concerns":[],"overfitting_risk":"low"}')

    result = SelfCritic(llm=llm).critique(proposal)

    assert result.approved is False
    assert llm.calls == 0


def test_self_critic_rejects_high_overfitting_risk():
    result = SelfCritic(
        llm=StubLLM(
            '{"approved":true,"confidence":0.9,"concerns":["fragile"],"overfitting_risk":"high"}'
        )
    ).critique(_proposal())

    assert result.approved is False


def test_test_runner_reports_failure(tmp_path: Path):
    def runner(command, cwd):
        report = {
            "summary": {"passed": 1, "failed": 1, "errors": 0},
            "tests": [{"nodeid": "tests/test_x.py::test_bad", "outcome": "failed"}],
        }
        (cwd / ".test_report.json").write_text(json.dumps(report), encoding="utf-8")
        return subprocess.CompletedProcess(command, 1)

    result = TestRunner(runner=runner).run(tmp_path)

    assert result.success is False
    assert result.failed_tests == ("tests/test_x.py::test_bad",)


def test_backtest_gate_rejects_sharpe_degradation():
    class Engine:
        def run(self, tickers, start, end, config):
            return _baseline(sharpe=0.5, drawdown=0.1)

    result = BacktestGate(engine_factory=lambda: Engine()).validate(
        _proposal(),
        _baseline(sharpe=1.0),
        ["MSFT"],
        date(2022, 1, 1),
        date(2022, 12, 31),
    )

    assert result.passed is False


def test_build_loop_skips_backtest_gate_when_tests_fail():
    class Engine:
        def run(self, tickers, start, end, config):
            return _baseline()

    class Generator:
        def generate(self, baseline, max_proposals=3):
            return [_proposal()]

    class Critic:
        def critique(self, proposal):
            return type("Critique", (), {"approved": True})()

    class Tests:
        def run(self):
            return TestResult(success=False, failed=1)

    class Gate:
        called = False

        def validate(self, *args, **kwargs):
            self.called = True
            raise AssertionError("gate should not run")

    gate = Gate()
    report = BuildLoop(
        engine=Engine(),
        generator=Generator(),
        critic=Critic(),
        test_runner=Tests(),
        backtest_gate=gate,
        store=None,
    ).run_once(["MSFT"], date(2022, 1, 1), date(2022, 12, 31))

    assert report.proposals_approved_by_critic == 1
    assert report.proposals_passed_tests == 0
    assert gate.called is False


def test_build_loop_pr_count_for_mixed_gate_results():
    class Engine:
        def run(self, tickers, start, end, config):
            return _baseline()

    class Generator:
        def generate(self, baseline, max_proposals=3):
            return [_proposal(), _proposal()]

    class Critic:
        def critique(self, proposal):
            return type("Critique", (), {"approved": True})()

    class Tests:
        def run(self):
            return TestResult(success=True, passed=10)

    class Gate:
        def __init__(self):
            self.count = 0

        def validate(self, *args, **kwargs):
            self.count += 1
            return type("GateResult", (), {"passed": self.count == 1})()

    class Store:
        def __init__(self):
            self.count = 0

        def submit(self, proposal):
            self.count += 1

    store = Store()
    report = BuildLoop(
        engine=Engine(),
        generator=Generator(),
        critic=Critic(),
        test_runner=Tests(),
        backtest_gate=Gate(),
        store=store,
    ).run_once(["MSFT"], date(2022, 1, 1), date(2022, 12, 31))

    assert report.prs_opened == 1
    assert store.count == 1

