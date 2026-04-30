from datetime import datetime, timedelta

import pytest

from ai_trader.domain.events import SocialMention, SourceName
from ai_trader.domain.signals import SignalDirection
from ai_trader.intelligence.models import PsychologyStage
from ai_trader.psychology.scoring import build_psychology_signal
from ai_trader.psychology.state_machine import ReflexivityStateMachine
from ai_trader.psychology.velocity import SocialVelocityIndicator


def test_state_machine_transitions_after_consecutive_evidence():
    now = datetime(2026, 1, 1, 12, 0, 0)
    machine = ReflexivityStateMachine(
        current_stage=PsychologyStage.DEPRESSION,
        stage_entered_at=now,
        consecutive_updates=3,
    )

    assert machine.update(now + timedelta(minutes=1), 2.0, 2.0, 0.0, 0.0) is PsychologyStage.DEPRESSION
    assert machine.update(now + timedelta(minutes=2), 2.0, 2.0, 0.0, 0.0) is PsychologyStage.DEPRESSION
    assert machine.update(now + timedelta(minutes=3), 2.0, 2.0, 0.0, 0.0) is PsychologyStage.DISBELIEF


def test_state_machine_single_spike_does_not_thrash():
    now = datetime(2026, 1, 1, 12, 0, 0)
    machine = ReflexivityStateMachine(
        current_stage=PsychologyStage.HOPE,
        stage_entered_at=now,
        consecutive_updates=3,
    )

    machine.update(now + timedelta(minutes=1), 2.0, 2.0, 0.0, 0.0)
    machine.update(now + timedelta(minutes=2), 0.0, 0.0, 0.0, 0.0)

    assert machine.current_stage is PsychologyStage.HOPE
    assert machine.history == []


def test_social_velocity_lookahead_guard():
    now = datetime(2026, 1, 1, 12, 0, 0)
    indicator = SocialVelocityIndicator()
    indicator.ingest(
        SocialMention(
            mention_id="future",
            platform=SourceName.REDDIT,
            published_at=now + timedelta(minutes=1),
            author="u",
            text="MSFT",
            engagement_count=10,
        )
    )

    with pytest.raises(ValueError):
        indicator.velocity(as_of=now)


def test_build_signal_direction_for_all_stages():
    now = datetime(2026, 1, 1, 12, 0, 0)
    expected = {
        PsychologyStage.DISBELIEF: SignalDirection.LONG,
        PsychologyStage.HOPE: SignalDirection.LONG,
        PsychologyStage.OPTIMISM: SignalDirection.LONG,
        PsychologyStage.BELIEF: SignalDirection.LONG,
        PsychologyStage.THRILL: SignalDirection.LONG,
        PsychologyStage.EUPHORIA: SignalDirection.SHORT,
        PsychologyStage.COMPLACENCY: SignalDirection.SHORT,
        PsychologyStage.ANXIETY: SignalDirection.SHORT,
        PsychologyStage.DENIAL: SignalDirection.SHORT,
        PsychologyStage.PANIC: SignalDirection.SHORT,
        PsychologyStage.CAPITULATION: SignalDirection.SHORT,
        PsychologyStage.DEPRESSION: SignalDirection.LONG,
    }
    for stage, direction in expected.items():
        machine = ReflexivityStateMachine(current_stage=stage, stage_entered_at=now)
        signal = build_psychology_signal(ticker="MSFT", as_of=now, state_machine=machine)
        assert signal.direction is direction
        assert signal.metadata["source"] == SourceName.INTERNAL.value


def test_velocity_zscore_gracefully_degrades_with_insufficient_history():
    indicator = SocialVelocityIndicator()
    assert indicator.zscore() == 0.0

