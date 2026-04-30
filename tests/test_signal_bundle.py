from datetime import date

import pytest

from ai_trader.domain.signals import Signal, SignalBundle, SignalDirection


def test_signal_bundle_rejects_future_effective_signal():
    future_signal = Signal(
        name="future",
        ticker="MSFT",
        direction=SignalDirection.LONG,
        strength=0.5,
        confidence=0.5,
        effective_date=date(2026, 2, 1),
    )

    with pytest.raises(ValueError, match="not effective"):
        SignalBundle(ticker="MSFT", as_of=date(2026, 1, 31), signals=(future_signal,))


def test_signal_bundle_combines_signed_strength():
    long_signal = Signal(
        name="long",
        ticker="MSFT",
        direction=SignalDirection.LONG,
        strength=0.8,
        confidence=0.75,
        effective_date=date(2026, 1, 1),
    )
    short_signal = Signal(
        name="short",
        ticker="MSFT",
        direction=SignalDirection.SHORT,
        strength=0.2,
        confidence=0.25,
        effective_date=date(2026, 1, 1),
    )

    bundle = SignalBundle(ticker="MSFT", as_of=date(2026, 1, 2), signals=(long_signal, short_signal))

    assert bundle.direction is SignalDirection.LONG
    assert bundle.conviction == pytest.approx(0.55)

