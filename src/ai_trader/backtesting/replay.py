from __future__ import annotations

from datetime import date
from enum import Enum
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_trader.domain.events import CongressionalTrade, ThirteenFPositionChange


class ReplayEventType(str, Enum):
    CONGRESSIONAL_TRADE = "congressional_trade"
    THIRTEEN_F_CHANGE = "13f_change"


class ReplayEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_type: ReplayEventType
    congressional_trade: CongressionalTrade | None = None
    thirteen_f_change: ThirteenFPositionChange | None = None
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_payload(self) -> ReplayEvent:
        if self.event_type is ReplayEventType.CONGRESSIONAL_TRADE and self.congressional_trade is None:
            raise ValueError("congressional_trade payload is required")
        if self.event_type is ReplayEventType.THIRTEEN_F_CHANGE and self.thirteen_f_change is None:
            raise ValueError("thirteen_f_change payload is required")
        return self

    @property
    def ticker(self) -> str:
        if self.congressional_trade is not None:
            return self.congressional_trade.ticker.upper()
        assert self.thirteen_f_change is not None
        return self.thirteen_f_change.current.ticker.upper()

    @property
    def effective_date(self) -> date:
        if self.congressional_trade is not None:
            return self.congressional_trade.disclosure_date
        assert self.thirteen_f_change is not None
        return self.thirteen_f_change.current.filing_date


class EventReplay:
    def __init__(self, events: Iterable[ReplayEvent] = ()) -> None:
        self._events = tuple(sorted(events, key=lambda event: event.effective_date))

    @property
    def events(self) -> tuple[ReplayEvent, ...]:
        return self._events

    @classmethod
    def from_jsonl(cls, path: Path) -> EventReplay:
        events: list[ReplayEvent] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                events.append(ReplayEvent.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"Invalid replay event at {path}:{line_number}: {exc}") from exc
        return cls(events)

    def available_as_of(
        self,
        as_of: date,
        *,
        ticker: str | None = None,
        start: date | None = None,
    ) -> tuple[ReplayEvent, ...]:
        ticker_upper = ticker.upper() if ticker else None
        results = []
        for event in self._events:
            if event.effective_date > as_of:
                continue
            if start is not None and event.effective_date < start:
                continue
            if ticker_upper is not None and event.ticker != ticker_upper:
                continue
            results.append(event)
        return tuple(results)

    def events_on(
        self,
        day: date,
        *,
        ticker: str | None = None,
    ) -> tuple[ReplayEvent, ...]:
        ticker_upper = ticker.upper() if ticker else None
        return tuple(
            event
            for event in self._events
            if event.effective_date == day and (ticker_upper is None or event.ticker == ticker_upper)
        )

    @staticmethod
    def split(events: Iterable[ReplayEvent]) -> tuple[tuple[CongressionalTrade, ...], tuple[ThirteenFPositionChange, ...]]:
        congressional: list[CongressionalTrade] = []
        institutional: list[ThirteenFPositionChange] = []
        for event in events:
            if event.congressional_trade is not None:
                congressional.append(event.congressional_trade)
            if event.thirteen_f_change is not None:
                institutional.append(event.thirteen_f_change)
        return tuple(congressional), tuple(institutional)

