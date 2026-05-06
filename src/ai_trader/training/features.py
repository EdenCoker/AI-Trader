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
    "has_insider_buy",
    "insider_value_usd",
    "eps_surprise_pct",
    "put_call_ratio",
    "yield_spread_2_10",
    "institutional_delta_shares",
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

    signal_features = _signal_features(bundle)

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
            signal_features["has_insider_buy"],
            signal_features["insider_value_usd"],
            signal_features["eps_surprise_pct"],
            signal_features["put_call_ratio"],
            signal_features["yield_spread_2_10"],
            signal_features["institutional_delta_shares"],
        ],
        dtype=float,
    )


def _direction_value(direction: SignalDirection) -> float:
    if direction is SignalDirection.LONG:
        return 1.0
    if direction is SignalDirection.SHORT:
        return -1.0
    return 0.0


def _signal_features(bundle: SignalBundle) -> dict[str, float]:
    features = {
        "has_insider_buy": 0.0,
        "insider_value_usd": 0.0,
        "eps_surprise_pct": 0.0,
        "put_call_ratio": 0.0,
        "yield_spread_2_10": 0.0,
        "institutional_delta_shares": 0.0,
    }
    for signal in bundle.signals:
        metadata = signal.metadata or {}
        if signal.name == "insider_buy":
            features["has_insider_buy"] = 1.0
        if signal.name in {"insider_buy", "insider_sell"}:
            features["insider_value_usd"] = max(
                features["insider_value_usd"],
                _scaled_money(metadata.get("transaction_value_usd")),
            )
        if signal.name in {"earnings_beat", "earnings_miss"}:
            features["eps_surprise_pct"] = _clip(
                _float(metadata.get("eps_surprise_pct")) / 100.0,
                -2.0,
                2.0,
            )
        if signal.name == "options_put_call_contrarian":
            features["put_call_ratio"] = (
                _clip(_float(metadata.get("put_call_ratio")), 0.0, 10.0) / 10.0
            )
        if signal.name == "macro_regime":
            features["yield_spread_2_10"] = _clip(
                _float(metadata.get("yield_spread_2_10")),
                -5.0,
                5.0,
            ) / 5.0
        if signal.name == "institutional_accumulation":
            features["institutional_delta_shares"] = max(
                features["institutional_delta_shares"],
                _clip(_float(metadata.get("institutional_delta_shares")), 0.0, 100_000_000.0)
                / 100_000_000.0,
            )
    return features


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _scaled_money(value) -> float:
    return _clip(_float(value), 0.0, 50_000_000.0) / 50_000_000.0


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
