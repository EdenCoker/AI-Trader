from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta

from ai_trader.domain.events import SocialMention


class SocialVelocityIndicator:
    def __init__(self) -> None:
        self._mentions: deque[SocialMention] = deque()
        self._velocity_samples: deque[tuple[datetime, float]] = deque()

    def ingest(self, mention: SocialMention) -> None:
        self._mentions.append(mention)
        self._mentions = deque(sorted(self._mentions, key=lambda item: item.published_at))

    def assert_no_lookahead(self, as_of: datetime) -> None:
        future = [mention.mention_id for mention in self._mentions if mention.published_at > as_of]
        if future:
            raise ValueError(f"Social mentions are not public as of {as_of.isoformat()}: {future[:3]}")

    def velocity(self, window_hours: float = 4.0, *, as_of: datetime | None = None) -> float:
        as_of = as_of or self._latest_timestamp()
        if as_of is None:
            return 0.0
        self.assert_no_lookahead(as_of)

        window = timedelta(hours=window_hours)
        recent_start = as_of - window
        baseline_start = as_of - window * 2

        recent = self._engagement_between(recent_start, as_of)
        baseline = self._engagement_between(baseline_start, recent_start)
        value = (recent - baseline) / max(baseline, 1)
        self._record_velocity(as_of, value)
        return value

    def acceleration(self, *, as_of: datetime | None = None) -> float:
        as_of = as_of or self._latest_timestamp()
        if as_of is None:
            return 0.0
        self.velocity(as_of=as_of)
        if len(self._velocity_samples) < 2:
            return 0.0
        (_, previous), (current_ts, current) = self._velocity_samples[-2], self._velocity_samples[-1]
        elapsed_hours = max((current_ts - self._velocity_samples[-2][0]).total_seconds() / 3600, 1e-9)
        return (current - previous) / elapsed_hours

    def zscore(self, lookback_days: int = 30, *, as_of: datetime | None = None) -> float:
        as_of = as_of or self._latest_timestamp()
        if as_of is None:
            return 0.0
        current = self.velocity(as_of=as_of)
        cutoff = as_of - timedelta(days=lookback_days)
        values = [value for timestamp, value in self._velocity_samples if cutoff <= timestamp <= as_of]
        if len(values) < 2:
            return 0.0

        mean, variance = _welford(values)
        if variance <= 0:
            return 0.0
        return (current - mean) / (variance**0.5)

    def _latest_timestamp(self) -> datetime | None:
        if not self._mentions:
            return None
        return self._mentions[-1].published_at

    def _engagement_between(self, start: datetime, end: datetime) -> int:
        return sum(
            mention.engagement_count
            for mention in self._mentions
            if start < mention.published_at <= end
        )

    def _record_velocity(self, as_of: datetime, value: float) -> None:
        if self._velocity_samples and self._velocity_samples[-1][0] == as_of:
            self._velocity_samples[-1] = (as_of, value)
        else:
            self._velocity_samples.append((as_of, value))
        cutoff = as_of - timedelta(days=30)
        while self._velocity_samples and self._velocity_samples[0][0] < cutoff:
            self._velocity_samples.popleft()


def _welford(values: list[float]) -> tuple[float, float]:
    mean = 0.0
    m2 = 0.0
    count = 0
    for value in values:
        count += 1
        delta = value - mean
        mean += delta / count
        m2 += delta * (value - mean)
    if count < 2:
        return mean, 0.0
    return mean, m2 / (count - 1)

