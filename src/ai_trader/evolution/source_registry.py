from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

SourceStatus = Literal["active", "pending_approval", "suspended", "candidate"]


class DataSourceRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    type: str
    url: HttpUrl | str
    auth: str | None = None
    schema_version: int = 1
    last_validated: date | None = None
    status: SourceStatus = "candidate"
    lift_score: float = 0.0
    coverage_score: float = Field(default=0.0, ge=0.0, le=1.0)
    freshness_score: float | None = Field(default=None, ge=0.0, le=1.0)
    complexity_score: float = Field(default=0.2, ge=0.0, le=1.0)
    free_tier: bool = True
    category: str = "general"
    profitability_proxy: float = Field(default=0.0, ge=0.0, le=1.0)
    ingestion_adapter: str | None = None
    notes: str | None = None


class SourceRegistry(BaseModel):
    model_config = ConfigDict(frozen=True)

    sources: tuple[DataSourceRecord, ...] = ()
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def load(cls, path: Path) -> SourceRegistry:
        if not path.exists():
            return cls()
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json")
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def by_id(self) -> dict[str, DataSourceRecord]:
        return {source.id: source for source in self.sources}

    def active_sources(self) -> tuple[DataSourceRecord, ...]:
        return tuple(source for source in self.sources if source.status == "active")

    def due_for_validation(
        self,
        *,
        max_age_days: int = 30,
        today: date | None = None,
    ) -> tuple[DataSourceRecord, ...]:
        today = today or datetime.now(UTC).date()
        cutoff = today - timedelta(days=max_age_days)
        return tuple(
            source
            for source in self.sources
            if source.last_validated is None or source.last_validated <= cutoff
        )

    def replace(self, source: DataSourceRecord) -> SourceRegistry:
        records = []
        replaced = False
        for existing in self.sources:
            if existing.id == source.id:
                records.append(source)
                replaced = True
            else:
                records.append(existing)
        if not replaced:
            records.append(source)
        records.sort(key=lambda item: item.id)
        return self.model_copy(
            update={"sources": tuple(records), "updated_at": datetime.now(UTC)}
        )

    def with_sources(self, sources: tuple[DataSourceRecord, ...]) -> SourceRegistry:
        return self.model_copy(
            update={
                "sources": tuple(sorted(sources, key=lambda source: source.id)),
                "updated_at": datetime.now(UTC),
            }
        )


def source_score(source: DataSourceRecord, *, today: date | None = None) -> float:
    today = today or datetime.now(UTC).date()
    coverage = source.coverage_score
    freshness = source.freshness_score
    if freshness is None:
        freshness = _freshness_from_last_validated(source.last_validated, today=today)
    lift = max(-1.0, min(1.0, source.lift_score))
    complexity = source.complexity_score
    return round((0.35 * coverage) + (0.25 * freshness) + (0.30 * lift) - (0.10 * complexity), 6)


def _freshness_from_last_validated(last_validated: date | None, *, today: date) -> float:
    if last_validated is None:
        return 0.0
    age_days = max(0, (today - last_validated).days)
    if age_days <= 1:
        return 1.0
    return max(0.0, min(1.0, 2.718281828 ** (-age_days / 14)))
