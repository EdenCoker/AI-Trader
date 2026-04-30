from __future__ import annotations

import numpy as np

from ai_trader.domain.signals import SignalBundle, SignalDirection
from ai_trader.intelligence.models import NarrativeIntelligence
from ai_trader.intelligence.trade_plan import TradePlan


FEATURE_NAMES = (
    "bundle_combined_strength",
    "bundle_conviction",
    "bundle_direction",
    "signal_count",
    "avg_signal_strength",
    "avg_signal_confidence",
    "plan_direction",
    "plan_conviction",
    "plan_size_multiplier",
    "plan_holding_period_norm",
    "narrative_surprise_edge",
    "narrative_volatility_risk",
    "narrative_contrarian_risk",
)


def extract_features(
    *,
    bundle: SignalBundle,
    plan: TradePlan,
    narrative: NarrativeIntelligence | None = None,
) -> np.ndarray:
    signals = bundle.signals
    signal_count = len(signals)
    avg_strength = (
        sum(signal.strength for signal in signals) / signal_count if signal_count else 0.0
    )
    avg_confidence = (
        sum(signal.confidence for signal in signals) / signal_count if signal_count else 0.0
    )

    surprise_edge = 0.0
    volatility_risk = 0.0
    contrarian_risk = 0.0
    if narrative is not None:
        surprise_edge = narrative.surprise.surprise_score * (
            1.0 - narrative.surprise.priced_in_fraction
        )
        volatility_risk = narrative.behavior.volatility_risk
        contrarian_risk = narrative.behavior.contrarian_risk

    return np.asarray(
        [
            bundle.combined_strength,
            bundle.conviction,
            _direction_value(bundle.direction),
            min(signal_count, 20) / 20.0,
            avg_strength,
            avg_confidence,
            _direction_value(plan.direction),
            plan.conviction,
            plan.size_multiplier / 2.0,
            min(plan.holding_period_days, 365) / 365.0,
            surprise_edge,
            volatility_risk,
            contrarian_risk,
        ],
        dtype=float,
    )


def _direction_value(direction: SignalDirection) -> float:
    if direction is SignalDirection.LONG:
        return 1.0
    if direction is SignalDirection.SHORT:
        return -1.0
    return 0.0
