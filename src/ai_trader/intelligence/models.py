from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class PsychologyStage(str, Enum):
    DISBELIEF = "Disbelief"
    HOPE = "Hope"
    OPTIMISM = "Optimism"
    BELIEF = "Belief"
    THRILL = "Thrill"
    EUPHORIA = "Euphoria"
    COMPLACENCY = "Complacency"
    ANXIETY = "Anxiety"
    DENIAL = "Denial"
    PANIC = "Panic"
    CAPITULATION = "Capitulation"
    DEPRESSION = "Depression"


class ExpectationCalibration(BaseModel):
    model_config = ConfigDict(frozen=True)

    consensus_view: str
    key_expectations: tuple[str, ...] = ()
    implied_positioning: str
    confidence: float = Field(ge=0, le=1)


class SurpriseAssessment(BaseModel):
    model_config = ConfigDict(frozen=True)

    direction: str
    surprise_score: float = Field(ge=0, le=1)
    priced_in_fraction: float = Field(ge=0, le=1)
    novelty: float = Field(ge=0, le=1)
    what_changed: tuple[str, ...] = ()
    pricing_context: str


class BehaviorPrediction(BaseModel):
    model_config = ConfigDict(frozen=True)

    psychology_stage: PsychologyStage
    immediate_reaction: str
    follow_through_1w: str
    volatility_risk: float = Field(ge=0, le=1)
    contrarian_risk: float = Field(ge=0, le=1)
    watch_for: tuple[str, ...] = ()


class NarrativeIntelligence(BaseModel):
    model_config = ConfigDict(frozen=True)

    calibration: ExpectationCalibration
    surprise: SurpriseAssessment
    behavior: BehaviorPrediction

