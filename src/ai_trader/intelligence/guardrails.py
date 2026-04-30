from __future__ import annotations

from ai_trader.domain.signals import SignalBundle, SignalDirection
from ai_trader.intelligence.models import NarrativeIntelligence
from ai_trader.intelligence.trade_plan import TradePlan


class ReasoningGuardrails:
    """Deterministic post-processing for LLM-generated trade plans."""

    def apply(
        self,
        *,
        plan: TradePlan,
        bundle: SignalBundle,
        narrative: NarrativeIntelligence | None = None,
    ) -> TradePlan:
        notes: list[str] = list(plan.guardrails)
        direction = plan.direction
        conviction = plan.conviction
        size_multiplier = plan.size_multiplier
        evidence_strength = _evidence_strength(bundle, narrative)
        has_evidence = bool(bundle.signals) or narrative is not None

        max_conviction = 0.15 if not has_evidence else min(0.95, 0.25 + 0.75 * evidence_strength)
        signal_direction = bundle.direction

        if not has_evidence and direction is not SignalDirection.NEUTRAL:
            direction = SignalDirection.NEUTRAL
            size_multiplier = 0.0
            max_conviction = min(max_conviction, 0.15)
            notes.append("neutralized: no signal or narrative evidence supplied")

        if (
            signal_direction is not SignalDirection.NEUTRAL
            and direction is not SignalDirection.NEUTRAL
            and direction is not signal_direction
            and bundle.conviction >= 0.25
        ):
            direction = SignalDirection.NEUTRAL
            size_multiplier = 0.0
            max_conviction = min(max_conviction, 0.25)
            notes.append("neutralized: plan direction contradicted fused signal bundle")

        if conviction > max_conviction:
            notes.append(f"conviction capped from {conviction:.2f} to {max_conviction:.2f}")
            conviction = max_conviction

        max_size = 0.0 if direction is SignalDirection.NEUTRAL else min(2.0, conviction * 2.0)
        if conviction < 0.20:
            max_size = min(max_size, 0.25)
        if size_multiplier > max_size:
            notes.append(f"size capped from {size_multiplier:.2f} to {max_size:.2f}")
            size_multiplier = max_size

        return plan.model_copy(
            update={
                "ticker": bundle.ticker,
                "as_of": bundle.as_of,
                "direction": direction,
                "conviction": round(float(conviction), 6),
                "size_multiplier": round(float(size_multiplier), 6),
                "guardrails": tuple(notes),
            }
        )


def _evidence_strength(bundle: SignalBundle, narrative: NarrativeIntelligence | None) -> float:
    strength = bundle.conviction
    if narrative is not None:
        surprise_edge = narrative.surprise.surprise_score * (1.0 - narrative.surprise.priced_in_fraction)
        confidence_edge = narrative.calibration.confidence * 0.5
        strength = max(strength, min(1.0, surprise_edge + confidence_edge))
    return max(0.0, min(1.0, strength))

