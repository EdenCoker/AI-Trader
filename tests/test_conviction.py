from datetime import date

import pytest

from ai_trader.domain.signals import Signal, SignalBundle, SignalDirection
from ai_trader.training.conviction import (
    ConvictionMetric,
    agreement_adjusted_conviction,
    conviction_evidence,
)


def _signal(
    name: str,
    direction: SignalDirection,
    strength: float,
    confidence: float = 0.8,
) -> Signal:
    return Signal(
        name=name,
        ticker="MSFT",
        direction=direction,
        strength=strength,
        confidence=confidence,
        effective_date=date(2026, 5, 1),
    )


def test_agreement_adjusted_conviction_rewards_breadth_and_agreement():
    broad_bundle = SignalBundle(
        ticker="MSFT",
        as_of=date(2026, 5, 1),
        signals=(
            _signal("fundamental_growth", SignalDirection.LONG, 0.6),
            _signal("analyst_consensus", SignalDirection.LONG, 0.5),
            _signal("employment_macro", SignalDirection.LONG, 0.4),
        ),
    )
    single_signal_bundle = SignalBundle(
        ticker="MSFT",
        as_of=date(2026, 5, 1),
        signals=(_signal("fundamental_growth", SignalDirection.LONG, 0.6),),
    )

    assert agreement_adjusted_conviction(
        broad_bundle,
        direction=SignalDirection.LONG,
    ) > agreement_adjusted_conviction(single_signal_bundle, direction=SignalDirection.LONG)


def test_agreement_adjusted_conviction_penalizes_opposing_evidence():
    clean_bundle = SignalBundle(
        ticker="MSFT",
        as_of=date(2026, 5, 1),
        signals=(
            _signal("fundamental_growth", SignalDirection.LONG, 0.6),
            _signal("analyst_consensus", SignalDirection.LONG, 0.5),
        ),
    )
    conflicted_bundle = SignalBundle(
        ticker="MSFT",
        as_of=date(2026, 5, 1),
        signals=(
            _signal("fundamental_growth", SignalDirection.LONG, 0.6),
            _signal("analyst_consensus", SignalDirection.LONG, 0.5),
            _signal("macro_regime", SignalDirection.SHORT, 0.4),
        ),
    )

    assert agreement_adjusted_conviction(
        conflicted_bundle,
        direction=SignalDirection.LONG,
    ) < agreement_adjusted_conviction(clean_bundle, direction=SignalDirection.LONG)


def test_conviction_evidence_exposes_failure_drivers():
    bundle = SignalBundle(
        ticker="MSFT",
        as_of=date(2026, 5, 1),
        signals=(
            _signal("fundamental_growth", SignalDirection.LONG, 0.8),
            _signal("macro_regime", SignalDirection.SHORT, 0.4),
        ),
    )

    evidence = conviction_evidence(bundle, direction=SignalDirection.LONG)

    assert evidence.support_count == 1
    assert evidence.opposing_count == 1
    assert evidence.agreement == pytest.approx(1 / 3)


def test_conviction_metric_enum_values_are_stable():
    assert ConvictionMetric.PLAN.value == "plan"
    assert ConvictionMetric.BUNDLE.value == "bundle"
    assert ConvictionMetric.AGREEMENT_ADJUSTED.value == "agreement_adjusted"
