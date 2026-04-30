from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ai_trader.intelligence.trade_plan import TradePlan


SAFETY_PATTERNS = (
    "allow_live_trading",
    "look_ahead",
    "lookahead",
    "disclosure_date",
    "filing_date",
    "effective_date",
    "LookAheadBiasError",
    "BrokerConfigurationError",
)


class ProposalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"


class TradeOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    trade_plan: TradePlan
    ticker: str
    actual_entry_price: float = Field(gt=0)
    actual_exit_price: float = Field(gt=0)
    actual_exit_date: date
    exit_reason: Literal["stop_hit", "target_hit", "time_exit", "thesis_invalid"]
    pnl_pct: float


class PromptProposal(BaseModel):
    model_config = ConfigDict(frozen=True)

    proposal_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    source_outcome: TradeOutcome
    target_module: str
    current_text: str
    proposed_text: str
    rationale: str
    expected_improvement: str
    status: ProposalStatus = ProposalStatus.PENDING

    @field_validator("target_module", "current_text", "proposed_text")
    @classmethod
    def _reject_safety_targets(cls, value: str) -> str:
        assert_safe_target(value)
        return value


class WeightProposal(BaseModel):
    model_config = ConfigDict(frozen=True)

    proposal_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    source_outcome: TradeOutcome
    target_field: str
    current_value: float
    proposed_value: float
    delta_pct: float
    rationale: str
    status: ProposalStatus = ProposalStatus.PENDING

    @field_validator("delta_pct")
    @classmethod
    def _limit_delta_pct(cls, value: float) -> float:
        if abs(value) > 0.20:
            raise ValueError("WeightProposal delta_pct is capped at +/-20%")
        return value

    @field_validator("target_field", "rationale")
    @classmethod
    def _reject_safety_targets(cls, value: str) -> str:
        assert_safe_target(value)
        return value


Proposal = PromptProposal | WeightProposal


def assert_safe_target(value: str) -> None:
    lowered = value.casefold()
    for pattern in SAFETY_PATTERNS:
        if pattern.casefold() in lowered:
            raise ValueError(f"AI proposals may not target safety-critical field: {pattern}")


def should_review(outcome: TradeOutcome) -> bool:
    return outcome.pnl_pct < -0.03 or outcome.pnl_pct > 0.10

