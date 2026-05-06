from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field


class AgentReport(BaseModel):
    """Structured report emitted by every autonomous evolution agent."""

    model_config = ConfigDict(frozen=True)

    agent: str
    run_id: str
    started_at: datetime
    finished_at: datetime
    status: str
    summary: dict = Field(default_factory=dict)
    errors: tuple[str, ...] = ()
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class ReportBuilder:
    def __init__(self, agent: str, run_id: str) -> None:
        self.agent = agent
        self.run_id = run_id
        self.started_at = datetime.now(UTC)
        self._started_perf = perf_counter()

    def build(
        self,
        *,
        status: str,
        summary: dict | None = None,
        errors: list[str] | tuple[str, ...] | None = None,
    ) -> AgentReport:
        finished_at = datetime.now(UTC)
        return AgentReport(
            agent=self.agent,
            run_id=self.run_id,
            started_at=self.started_at,
            finished_at=finished_at,
            status=status,
            summary=summary or {},
            errors=tuple(errors or ()),
            duration_s=round(perf_counter() - self._started_perf, 3),
        )


@contextmanager
def report_builder(agent: str, run_id: str) -> Iterator[ReportBuilder]:
    yield ReportBuilder(agent=agent, run_id=run_id)
