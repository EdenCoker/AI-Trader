from datetime import date
from decimal import Decimal

import pytest

from ai_trader.domain.events import (
    Chamber,
    CommitteeAssignment,
    CongressionalTrade,
    ThirteenFHolding,
    ThirteenFPositionChange,
    TransactionType,
)
from ai_trader.domain.signals import SignalDirection
from ai_trader.smart_money.scoring import LookAheadBiasError, SmartMoneyScorer


def _trade(sector: str = "Defense") -> CongressionalTrade:
    return CongressionalTrade(
        member_name="Jane Senator",
        chamber=Chamber.SENATE,
        ticker="LMT",
        issuer="Lockheed Martin",
        sector=sector,
        transaction_date=date(2026, 1, 10),
        disclosure_date=date(2026, 2, 15),
        transaction_type=TransactionType.PURCHASE,
        amount_low=Decimal("100001"),
        amount_high=Decimal("250000"),
    )


def test_congressional_scoring_uses_disclosure_date_not_transaction_date():
    scorer = SmartMoneyScorer()

    with pytest.raises(LookAheadBiasError, match="disclosure dates"):
        scorer.score_congressional_trade(_trade(), as_of=date(2026, 1, 30))


def test_committee_sector_match_increases_congressional_score():
    assignment = CommitteeAssignment(
        member_name="Jane Senator",
        chamber=Chamber.SENATE,
        committee_name="Armed Services",
        sectors=("aerospace and defense",),
        start_date=date(2025, 1, 1),
    )
    matched_scorer = SmartMoneyScorer(committee_assignments=(assignment,))
    unmatched_scorer = SmartMoneyScorer()

    matched = matched_scorer.score_congressional_trade(_trade(), as_of=date(2026, 2, 15))
    unmatched = unmatched_scorer.score_congressional_trade(_trade(), as_of=date(2026, 2, 15))

    assert matched.effective_date == date(2026, 2, 15)
    assert matched.direction is SignalDirection.LONG
    assert matched.strength > unmatched.strength + 0.2
    assert matched.metadata["committee_matches"] == ["Armed Services"]


def test_13f_scoring_uses_filing_date_not_report_period():
    holding = ThirteenFHolding(
        manager_name="Berkshire Hathaway",
        cik="1067983",
        filing_date=date(2026, 5, 15),
        report_period=date(2026, 3, 31),
        ticker="AAPL",
        issuer="Apple Inc.",
        market_value_usd=Decimal("1000000000"),
        shares=Decimal("10000000"),
    )
    change = ThirteenFPositionChange(current=holding, previous_shares=Decimal("5000000"))
    scorer = SmartMoneyScorer()

    with pytest.raises(LookAheadBiasError, match="filing dates"):
        scorer.score_13f_change(change, as_of=date(2026, 4, 30))

    signal = scorer.score_13f_change(change, as_of=date(2026, 5, 15))

    assert signal.effective_date == date(2026, 5, 15)
    assert signal.direction is SignalDirection.LONG
    assert signal.metadata["report_period"] == "2026-03-31"

