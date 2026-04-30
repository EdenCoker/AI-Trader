from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from ai_trader.domain.signals import SignalDirection


class TradePlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    as_of: date
    direction: SignalDirection
    conviction: float = Field(ge=0, le=1)
    size_multiplier: float = Field(ge=0, le=2)
    holding_period_days: int = Field(ge=1, le=365)
    exit_trigger: str
    thesis: tuple[str, ...] = ()
    analogies: tuple[str, ...] = ()
    guardrails: tuple[str, ...] = ()
