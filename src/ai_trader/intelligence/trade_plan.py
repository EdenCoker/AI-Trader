from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_trader.domain.signals import SignalDirection

HorizonClass = Literal["short", "medium", "long"]


def horizon_class_for_days(horizon_days: int) -> HorizonClass:
    if horizon_days <= 14:
        return "short"
    if horizon_days <= 45:
        return "medium"
    return "long"


class TradePlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    as_of: date
    direction: SignalDirection
    conviction: float = Field(ge=0, le=1)
    size_multiplier: float = Field(ge=0, le=2)
    holding_period_days: int = Field(ge=1, le=365)
    horizon_class: HorizonClass
    exit_trigger: str
    thesis: tuple[str, ...] = ()
    analogies: tuple[str, ...] = ()
    guardrails: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _derive_horizon_class(cls, data):
        if not isinstance(data, dict):
            return data
        if data.get("horizon_class"):
            return data
        holding_days = data.get("holding_period_days")
        if holding_days is not None:
            data = dict(data)
            data["horizon_class"] = horizon_class_for_days(int(holding_days))
        return data
