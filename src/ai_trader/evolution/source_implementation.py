from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ai_trader.evolution.reports import AgentReport, ReportBuilder
from ai_trader.evolution.source_registry import DataSourceRecord, SourceRegistry


class ImplementationTask(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str
    priority: int = Field(ge=1, le=5)
    category: str
    ingestion_adapter: str
    auth_env: str | None = None
    free_tier: bool = True
    profitability_proxy: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    bootstrap_steps: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SourceImplementationAgent:
    """Builds an implementation queue for high-value discovered sources."""

    def __init__(
        self,
        *,
        registry_path: Path = Path("data/source_registry.json"),
        out_path: Path = Path("data/source_proposals/implementation_tasks.json"),
        min_confidence: float = 0.45,
        run_id: str = "manual",
    ) -> None:
        self.registry_path = registry_path
        self.out_path = out_path
        self.min_confidence = min_confidence
        self.run_id = run_id

    def run(self) -> AgentReport:
        builder = ReportBuilder("SourceImplementationAgent", self.run_id)
        try:
            registry = SourceRegistry.load(self.registry_path)
            tasks = self._build_tasks(registry)
            self._write_tasks(tasks)
            return builder.build(
                status="ok",
                summary={
                    "sources": len(registry.sources),
                    "queued_tasks": len(tasks),
                    "min_confidence": self.min_confidence,
                    "out_path": str(self.out_path),
                },
            )
        except Exception as exc:
            return builder.build(status="failed", errors=[str(exc)])

    def _build_tasks(self, registry: SourceRegistry) -> list[ImplementationTask]:
        tasks: list[ImplementationTask] = []
        for source in registry.sources:
            if source.status not in {"pending_approval", "candidate", "active"}:
                continue
            confidence = _implementation_confidence(source)
            if confidence < self.min_confidence:
                continue
            steps = _bootstrap_steps(source)
            task = ImplementationTask(
                source_id=source.id,
                priority=_priority_from_source(source),
                category=source.category,
                ingestion_adapter=source.ingestion_adapter or _default_adapter(source),
                auth_env=source.auth,
                free_tier=source.free_tier,
                profitability_proxy=source.profitability_proxy,
                confidence=confidence,
                bootstrap_steps=steps,
            )
            tasks.append(task)
        tasks.sort(key=lambda item: (item.priority, -item.confidence, item.source_id))
        return tasks

    def _write_tasks(self, tasks: list[ImplementationTask]) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "tasks": [task.model_dump(mode="json") for task in tasks],
        }
        self.out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _implementation_confidence(source: DataSourceRecord) -> float:
    # Candidate probes often lack validation timestamps; use a neutral freshness prior.
    if source.freshness_score is None and source.status in {"candidate", "pending_approval"}:
        freshness = 0.75
    else:
        freshness = source.freshness_score or 0.0
    base = (
        (0.35 * source.coverage_score)
        + (0.25 * freshness)
        + (0.30 * max(source.lift_score, 0.0))
        - (0.10 * source.complexity_score)
    )
    value = (0.65 * max(base, 0.0)) + (0.35 * source.profitability_proxy)
    return round(max(0.0, min(value, 1.0)), 4)


def _priority_from_source(source: DataSourceRecord) -> int:
    if source.profitability_proxy >= 0.8:
        return 1
    if source.profitability_proxy >= 0.65:
        return 2
    if source.profitability_proxy >= 0.5:
        return 3
    if source.profitability_proxy >= 0.35:
        return 4
    return 5


def _default_adapter(source: DataSourceRecord) -> str:
    if source.type == "rss":
        return "rss_events"
    if source.type == "csv":
        return "csv_bulk"
    return "http_json"


def _bootstrap_steps(source: DataSourceRecord) -> tuple[str, ...]:
    adapter = source.ingestion_adapter or _default_adapter(source)
    steps: list[str] = [
        f"add loader hook for {source.id} using adapter={adapter}",
        f"attach features/signals for category={source.category}",
        "run ingest on a 20-ticker smoke set and validate schema",
        "backtest source-only and source+baseline strategies over trailing 2y",
    ]
    if source.auth:
        steps.insert(0, f"configure secret env var: {source.auth}")
    return tuple(steps)
