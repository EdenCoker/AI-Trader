from __future__ import annotations

from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field

from ai_trader.domain.events import SourceName
from ai_trader.domain.signals import Signal, SignalDirection
from ai_trader.intelligence.models import PsychologyStage


ACCUMULATION_NEXT: dict[PsychologyStage, PsychologyStage] = {
    PsychologyStage.DEPRESSION: PsychologyStage.DISBELIEF,
    PsychologyStage.DISBELIEF: PsychologyStage.HOPE,
    PsychologyStage.HOPE: PsychologyStage.OPTIMISM,
    PsychologyStage.OPTIMISM: PsychologyStage.BELIEF,
    PsychologyStage.BELIEF: PsychologyStage.THRILL,
    PsychologyStage.THRILL: PsychologyStage.EUPHORIA,
}

DISTRIBUTION_NEXT: dict[PsychologyStage, PsychologyStage] = {
    PsychologyStage.EUPHORIA: PsychologyStage.COMPLACENCY,
    PsychologyStage.COMPLACENCY: PsychologyStage.ANXIETY,
    PsychologyStage.ANXIETY: PsychologyStage.DENIAL,
    PsychologyStage.DENIAL: PsychologyStage.PANIC,
    PsychologyStage.PANIC: PsychologyStage.CAPITULATION,
    PsychologyStage.CAPITULATION: PsychologyStage.DEPRESSION,
}


class StageTransition(BaseModel):
    model_config = ConfigDict(frozen=True)

    from_stage: PsychologyStage
    to_stage: PsychologyStage
    as_of: datetime
    evidence_count: int
    evidence: dict[str, float] = Field(default_factory=dict)


class ReflexivityStateMachine:
    def __init__(
        self,
        *,
        current_stage: PsychologyStage = PsychologyStage.DISBELIEF,
        stage_entered_at: datetime | None = None,
        evidence_threshold: float = 1.0,
        min_evidence_count: int = 2,
        consecutive_updates: int = 3,
    ) -> None:
        self.current_stage = current_stage
        self.stage_entered_at = stage_entered_at or datetime.utcnow()
        self.evidence_threshold = evidence_threshold
        self.min_evidence_count = min_evidence_count
        self.consecutive_updates = consecutive_updates
        self.history: list[StageTransition] = []
        self._pending_stage: PsychologyStage | None = None
        self._pending_count = 0
        self._last_update: datetime | None = None

    def update(
        self,
        as_of: datetime,
        price_zscore: float,
        sentiment_velocity: float,
        social_zscore: float,
        vol_zscore: float,
    ) -> PsychologyStage:
        if self._last_update is not None and as_of < self._last_update:
            raise ValueError("as_of must be monotonic; refusing look-ahead-prone state updates")
        self._last_update = as_of

        target, count = self._candidate_transition(
            price_zscore=price_zscore,
            sentiment_velocity=sentiment_velocity,
            social_zscore=social_zscore,
            vol_zscore=vol_zscore,
        )
        if target is None or count < self.min_evidence_count:
            self._pending_stage = None
            self._pending_count = 0
            return self.current_stage

        if target == self._pending_stage:
            self._pending_count += 1
        else:
            self._pending_stage = target
            self._pending_count = 1

        if self._pending_count >= self.consecutive_updates:
            transition = StageTransition(
                from_stage=self.current_stage,
                to_stage=target,
                as_of=as_of,
                evidence_count=count,
                evidence={
                    "price_zscore": price_zscore,
                    "sentiment_velocity": sentiment_velocity,
                    "social_zscore": social_zscore,
                    "vol_zscore": vol_zscore,
                },
            )
            self.history.append(transition)
            self.current_stage = target
            self.stage_entered_at = as_of
            self._pending_stage = None
            self._pending_count = 0

        return self.current_stage

    def stage_duration(self, as_of: datetime | None = None) -> timedelta:
        return (as_of or datetime.utcnow()) - self.stage_entered_at

    def build_signal(self, ticker: str, as_of: datetime) -> Signal:
        direction, base_strength = _stage_signal_profile(self.current_stage)
        duration_days = max(0.0, self.stage_duration(as_of).total_seconds() / 86400)
        maturity_discount = min(0.25, duration_days / 240)
        strength = max(0.05, min(1.0, base_strength - maturity_discount))

        return Signal(
            name="psychology.reflexivity_state",
            ticker=ticker.upper(),
            direction=direction,
            strength=strength,
            confidence=min(1.0, 0.45 + len(self.history) * 0.03),
            effective_date=as_of.date(),
            horizon_days=_stage_horizon_days(self.current_stage, duration_days),
            invalidation="Psychology stage changes or social/price evidence reverses for 3 updates.",
            reasons=(f"Reflexivity psychology stage: {self.current_stage.value}",),
            metadata={
                "source": SourceName.INTERNAL.value,
                "as_of": as_of.isoformat(),
                "stage": self.current_stage.value,
                "stage_entered_at": self.stage_entered_at.isoformat(),
                "stage_duration_days": duration_days,
                "transition_count": len(self.history),
            },
        )

    def _candidate_transition(
        self,
        *,
        price_zscore: float,
        sentiment_velocity: float,
        social_zscore: float,
        vol_zscore: float,
    ) -> tuple[PsychologyStage | None, int]:
        if self.current_stage in ACCUMULATION_NEXT:
            count = _positive_evidence_count(
                price_zscore, sentiment_velocity, social_zscore, vol_zscore, self.evidence_threshold
            )
            return ACCUMULATION_NEXT[self.current_stage], count

        count = _distribution_evidence_count(
            price_zscore, sentiment_velocity, social_zscore, vol_zscore, self.evidence_threshold
        )
        return DISTRIBUTION_NEXT.get(self.current_stage), count


def _positive_evidence_count(
    price_zscore: float,
    sentiment_velocity: float,
    social_zscore: float,
    vol_zscore: float,
    threshold: float,
) -> int:
    return sum(
        (
            price_zscore >= threshold,
            sentiment_velocity >= threshold,
            social_zscore >= threshold,
            vol_zscore <= -threshold,
        )
    )


def _distribution_evidence_count(
    price_zscore: float,
    sentiment_velocity: float,
    social_zscore: float,
    vol_zscore: float,
    threshold: float,
) -> int:
    return sum(
        (
            price_zscore <= -threshold,
            sentiment_velocity <= -threshold,
            social_zscore >= threshold,
            vol_zscore >= threshold,
        )
    )


def _stage_signal_profile(stage: PsychologyStage) -> tuple[SignalDirection, float]:
    mapping = {
        PsychologyStage.DISBELIEF: (SignalDirection.LONG, 0.62),
        PsychologyStage.HOPE: (SignalDirection.LONG, 0.70),
        PsychologyStage.OPTIMISM: (SignalDirection.LONG, 0.66),
        PsychologyStage.BELIEF: (SignalDirection.LONG, 0.56),
        PsychologyStage.THRILL: (SignalDirection.LONG, 0.48),
        PsychologyStage.EUPHORIA: (SignalDirection.SHORT, 0.72),
        PsychologyStage.COMPLACENCY: (SignalDirection.SHORT, 0.66),
        PsychologyStage.ANXIETY: (SignalDirection.SHORT, 0.70),
        PsychologyStage.DENIAL: (SignalDirection.SHORT, 0.74),
        PsychologyStage.PANIC: (SignalDirection.SHORT, 0.82),
        PsychologyStage.CAPITULATION: (SignalDirection.SHORT, 0.78),
        PsychologyStage.DEPRESSION: (SignalDirection.LONG, 0.76),
    }
    return mapping[stage]


def _stage_horizon_days(stage: PsychologyStage, duration_days: float) -> int:
    early = {PsychologyStage.DEPRESSION, PsychologyStage.DISBELIEF, PsychologyStage.HOPE}
    middle = {PsychologyStage.OPTIMISM, PsychologyStage.BELIEF, PsychologyStage.ANXIETY, PsychologyStage.DENIAL}
    if stage in early:
        base = 90
    elif stage in middle:
        base = 45
    else:
        base = 21
    return max(7, int(base - min(duration_days, base * 0.5)))

