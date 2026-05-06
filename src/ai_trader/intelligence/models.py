from __future__ import annotations

from enum import Enum
from typing import Optional

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


class InsiderNewsAlignment(str, Enum):
    """Degree to which insider/congressional trade direction aligns with the news narrative."""

    STRONGLY_ALIGNED = "strongly_aligned"   # insider buys + bullish news; insider sells + bearish news
    ALIGNED = "aligned"                      # mild confirmation
    MIXED = "mixed"                          # no clear correlation; wait for confirmation
    OPPOSED = "opposed"                      # insider direction contradicts news (contrarian risk)
    STRONGLY_OPPOSED = "strongly_opposed"   # strong contradiction — high uncertainty


class InsiderNewsCorrelation(BaseModel):
    """Cross-reference between congressional/lobbying trades and current news narrative.

    Produced by FinalReasoner._correlate_insider_news() and injected into the
    final reasoning prompt so the LLM can weigh insider signals properly.
    """

    model_config = ConfigDict(frozen=True)

    has_insider_signal: bool
    insider_direction: str  # "buy", "sell", or "none"
    news_sentiment: str     # derived from NarrativeIntelligence (positive/negative/neutral/unknown)
    alignment: InsiderNewsAlignment
    conviction_delta: float = Field(ge=-0.5, le=0.5)   # additive adjustment to LLM conviction
    rationale: str
    catalysts: tuple[str, ...] = ()   # specific news items that corroborate or contradict the insider trade

