from __future__ import annotations

from datetime import datetime

from ai_trader.domain.signals import Signal
from ai_trader.psychology.state_machine import ReflexivityStateMachine
from ai_trader.psychology.velocity import SocialVelocityIndicator


def build_psychology_signal(
    *,
    ticker: str,
    as_of: datetime,
    state_machine: ReflexivityStateMachine,
    social_velocity: SocialVelocityIndicator | None = None,
) -> Signal:
    if social_velocity is not None:
        social_velocity.assert_no_lookahead(as_of)
    return state_machine.build_signal(ticker=ticker, as_of=as_of)

