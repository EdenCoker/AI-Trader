from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SignalDirection(str, Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"

    @property
    def multiplier(self) -> int:
        if self is SignalDirection.LONG:
            return 1
        if self is SignalDirection.SHORT:
            return -1
        return 0


class Signal(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    ticker: str
    source: str | None = None
    direction: SignalDirection
    strength: float = Field(ge=0, le=1)
    confidence: float = Field(default=0.5, ge=0, le=1)
    effective_date: date
    horizon_days: int = Field(default=30, ge=1)
    invalidation: str | None = None
    reasons: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def signed_strength(self) -> float:
        return self.direction.multiplier * self.strength


class SignalBundle(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    as_of: date
    signals: tuple[Signal, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _reject_future_effective_signals(self) -> SignalBundle:
        future = [signal.name for signal in self.signals if signal.effective_date > self.as_of]
        if future:
            names = ", ".join(future)
            raise ValueError(f"signals are not effective as of {self.as_of}: {names}")
        return self

    @property
    def combined_strength(self) -> float:
        if not self.signals:
            return 0.0

        weighted = sum(signal.signed_strength * signal.confidence for signal in self.signals)
        confidence = sum(signal.confidence for signal in self.signals)
        if confidence == 0:
            return 0.0
        return max(-1.0, min(1.0, weighted / confidence))

    @property
    def conviction(self) -> float:
        return abs(self.combined_strength)

    @property
    def direction(self) -> SignalDirection:
        if self.combined_strength > 0.05:
            return SignalDirection.LONG
        if self.combined_strength < -0.05:
            return SignalDirection.SHORT
        return SignalDirection.NEUTRAL
