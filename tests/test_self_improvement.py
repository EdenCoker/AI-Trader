from datetime import date
from pathlib import Path
import subprocess

import pytest

from ai_trader.domain.signals import SignalDirection
from ai_trader.intelligence.trade_plan import TradePlan
from ai_trader.self_improvement.git_store import GitProposalStore
from ai_trader.self_improvement.post_trade_review import PostTradeReviewer
from ai_trader.self_improvement.proposal import PromptProposal, TradeOutcome, WeightProposal
from ai_trader.self_improvement.scheduler import NightlyReviewScheduler


class StubLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def complete(self, prompt, *, system=None, model=None, temperature=0.2, max_tokens=None):
        self.calls += 1
        return self.outputs.pop(0)

    def chat(self, messages, *, model=None, temperature=0.2, max_tokens=None):
        raise NotImplementedError


def _outcome(pnl: float = -0.05) -> TradeOutcome:
    plan = TradePlan(
        ticker="MSFT",
        as_of=date(2026, 1, 1),
        direction=SignalDirection.LONG,
        conviction=0.7,
        size_multiplier=1.0,
        holding_period_days=10,
        exit_trigger="close below support",
        thesis=("strong setup",),
    )
    return TradeOutcome(
        trade_plan=plan,
        ticker="MSFT",
        actual_entry_price=100,
        actual_exit_price=95,
        actual_exit_date=date(2026, 1, 10),
        exit_reason="stop_hit",
        pnl_pct=pnl,
    )


def test_post_trade_reviewer_calls_llm_three_times():
    llm = StubLLM(
        [
            '{"reasoning_error":"overconfidence","missed_evidence":["vol"],"severity":0.8}',
            '{"root_cause":"sizing too high","target_kind":"weight","target":"SmartMoneyWeights.senate_bonus","current_value":"0.05"}',
            '{"proposal_kind":"weight","target":"SmartMoneyWeights.senate_bonus","current_value":0.05,"proposed_value":0.045,"delta_pct":-0.1,"rationale":"reduce chamber bonus"}',
        ]
    )

    proposal = PostTradeReviewer(llm=llm).review(_outcome())

    assert llm.calls == 3
    assert isinstance(proposal, WeightProposal)


def test_weight_proposal_rejects_large_delta():
    with pytest.raises(ValueError):
        WeightProposal(
            source_outcome=_outcome(),
            target_field="SmartMoneyWeights.senate_bonus",
            current_value=0.05,
            proposed_value=0.10,
            delta_pct=1.0,
            rationale="too much",
        )


def test_git_proposal_store_invokes_git_and_gh_in_order(tmp_path: Path):
    commands = []

    def runner(command):
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 0)

    proposal = PromptProposal(
        source_outcome=_outcome(),
        target_module="intelligence.narrative:SYSTEM_PROMPT",
        current_text="old",
        proposed_text="new",
        rationale="tighten prompt",
        expected_improvement="better calibration",
    )

    GitProposalStore(repo_root=tmp_path, runner=runner).submit(proposal)

    assert commands[0][:3] == ["git", "checkout", "-b"]
    assert commands[1] == ["git", "add", "-A"]
    assert commands[2][0:2] == ["git", "commit"]
    assert commands[3] == ["git", "push", "origin", "HEAD"]
    assert commands[4][0:3] == ["gh", "pr", "create"]


def test_scheduler_skips_neutral_trades(tmp_path: Path):
    outcomes_file = tmp_path / "outcomes.jsonl"
    outcomes_file.write_text(_outcome(pnl=0.01).model_dump_json() + "\n", encoding="utf-8")

    class Reviewer:
        def review(self, outcome):
            raise AssertionError("neutral trade should be skipped")

    summary = NightlyReviewScheduler(reviewer=Reviewer(), store=None).run(outcomes_file)

    assert summary["skipped"] == 1


def test_safety_target_rejected_before_git_runner(tmp_path: Path):
    called = False

    def runner(command):
        nonlocal called
        called = True
        return subprocess.CompletedProcess(command, 0)

    proposal = PromptProposal.model_construct(
        proposal_id="unsafe",
        created_at=date(2026, 1, 1),
        source_outcome=_outcome(),
        target_module="config:allow_live_trading",
        current_text="x",
        proposed_text="y",
        rationale="unsafe",
        expected_improvement="none",
    )

    with pytest.raises(ValueError):
        GitProposalStore(repo_root=tmp_path, runner=runner).submit(proposal)
    assert called is False

