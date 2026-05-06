from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ai_trader.evolution.reports import AgentReport, ReportBuilder
from ai_trader.evolution.source_registry import (
    DataSourceRecord,
    SourceRegistry,
    source_score,
)
from ai_trader.evolution.watchlist_manager import WatchlistManager


class DataSourceProposal(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str
    proposed_status: str
    score: float
    coverage: float
    freshness: float
    lift: float
    complexity: float
    reason: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DiscoveryAgent:
    """Scores registered data sources and creates approval proposals for promising ones."""

    def __init__(
        self,
        *,
        registry_path: Path = Path("data/source_registry.json"),
        proposals_dir: Path = Path("data/source_proposals"),
        watchlist_path: Path = Path("data/watchlist.txt"),
        min_score: float = 0.5,
        validation_age_days: int = 30,
        run_id: str = "manual",
    ) -> None:
        self.registry_path = registry_path
        self.proposals_dir = proposals_dir
        self.watchlist_path = watchlist_path
        self.min_score = min_score
        self.validation_age_days = validation_age_days
        self.run_id = run_id

    def run(self) -> AgentReport:
        builder = ReportBuilder("DiscoveryAgent", self.run_id)
        errors: list[str] = []
        try:
            registry = SourceRegistry.load(self.registry_path)
            watchlist = WatchlistManager(watchlist_path=self.watchlist_path).load()
            proposals: list[DataSourceProposal] = []
            updated = registry

            for source in registry.due_for_validation(max_age_days=self.validation_age_days):
                proposal = self._proposal_for(source, watchlist_size=len(watchlist))
                if proposal is not None:
                    proposals.append(proposal)
                    if source.status == "candidate" and proposal.score >= self.min_score:
                        updated = updated.replace(
                            source.model_copy(update={"status": "pending_approval"})
                        )

            for source in self._new_probe_candidates(registry):
                proposal = self._proposal_for(source, watchlist_size=len(watchlist))
                if proposal is None:
                    continue
                proposals.append(proposal)
                if proposal.score >= self.min_score:
                    updated = updated.replace(
                        source.model_copy(update={"status": "pending_approval"})
                    )

            if updated != registry:
                updated.save(self.registry_path)
            written = self._write_proposals(proposals)
            return builder.build(
                status="ok",
                summary={
                    "sources": len(registry.sources),
                    "active_sources": len(registry.active_sources()),
                    "watchlist_size": len(watchlist),
                    "proposals": written,
                    "pending_approval": sum(
                        1 for source in updated.sources if source.status == "pending_approval"
                    ),
                },
            )
        except Exception as exc:
            errors.append(str(exc))
            return builder.build(status="failed", errors=errors)

    def _proposal_for(
        self,
        source: DataSourceRecord,
        *,
        watchlist_size: int,
    ) -> DataSourceProposal | None:
        score = source_score(source)
        freshness = source.freshness_score
        if freshness is None:
            freshness = 1.0 if source.last_validated else 0.0
        if source.status == "active" and score >= 0:
            reason = "scheduled validation refresh"
            proposed_status = "active"
        elif score >= self.min_score:
            reason = "source score met pending approval threshold"
            proposed_status = "pending_approval"
        else:
            return None
        return DataSourceProposal(
            source_id=source.id,
            proposed_status=proposed_status,
            score=score,
            coverage=source.coverage_score,
            freshness=freshness,
            lift=source.lift_score,
            complexity=source.complexity_score,
            reason=f"{reason}; watchlist_size={watchlist_size}",
        )

    def _new_probe_candidates(self, registry: SourceRegistry) -> tuple[DataSourceRecord, ...]:
        existing = registry.by_id()
        candidates = []
        for source in DEFAULT_DISCOVERY_PROBES:
            if source.id not in existing:
                candidates.append(source)
        return tuple(candidates)

    def _write_proposals(self, proposals: list[DataSourceProposal]) -> int:
        if not proposals:
            return 0
        self.proposals_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        count = 0
        for proposal in proposals:
            path = self.proposals_dir / f"{stamp}_{proposal.source_id}.json"
            path.write_text(
                json.dumps(proposal.model_dump(mode="json"), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            count += 1
        return count


DEFAULT_DISCOVERY_PROBES = (
    DataSourceRecord(
        id="gdelt_finance_feed",
        type="rss",
        url="https://api.gdeltproject.org/api/v2/doc/doc",
        status="candidate",
        lift_score=0.08,
        coverage_score=0.35,
        complexity_score=0.35,
        notes="GDELT finance/news probe",
    ),
    DataSourceRecord(
        id="fmp_earnings_surprises",
        type="api",
        url="https://financialmodelingprep.com/api/v3/earning_surprises",
        auth="FMP_API_KEY",
        status="candidate",
        lift_score=0.12,
        coverage_score=0.45,
        complexity_score=0.25,
        notes="FMP free-tier earnings surprise probe",
    ),
)
