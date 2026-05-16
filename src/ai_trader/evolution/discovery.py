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
                    should_escalate = (
                        source.status == "candidate"
                        and proposal.proposed_status == "pending_approval"
                    )
                    if should_escalate:
                        updated = updated.replace(
                            source.model_copy(update={"status": "pending_approval"})
                        )

            for source in self._new_probe_candidates(registry):
                updated = updated.replace(source)
                proposal = self._proposal_for(source, watchlist_size=len(watchlist))
                if proposal is None:
                    continue
                proposals.append(proposal)
                if proposal.proposed_status == "pending_approval":
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
        elif score >= self.min_score or (
            source.status == "candidate" and source.profitability_proxy >= 0.7
        ):
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
        id="sec_form4_cluster",
        type="api",
        url="https://www.sec.gov/edgar/search/",
        auth="SEC_EDGAR_USER_AGENT",
        status="candidate",
        lift_score=0.16,
        coverage_score=0.52,
        freshness_score=0.92,
        complexity_score=0.42,
        free_tier=True,
        category="insider",
        profitability_proxy=0.88,
        ingestion_adapter="sec_form4",
        notes="Cluster insider buying/selling from SEC Form 4 filings.",
    ),
    DataSourceRecord(
        id="quiver_live_house_senate",
        type="api",
        url="https://api.quiverquant.com/beta/live/housetrading",
        auth="QUIVER_API_KEY",
        status="candidate",
        lift_score=0.15,
        coverage_score=0.48,
        freshness_score=0.95,
        complexity_score=0.30,
        free_tier=True,
        category="congress",
        profitability_proxy=0.82,
        ingestion_adapter="quiver_congress",
        notes="Near-real-time House/Senate trade disclosures from Quiver.",
    ),
    DataSourceRecord(
        id="sec_13f_position_initiations",
        type="api",
        url="https://www.sec.gov/edgar/search/",
        auth="SEC_EDGAR_USER_AGENT",
        status="candidate",
        lift_score=0.14,
        coverage_score=0.40,
        freshness_score=0.88,
        complexity_score=0.50,
        free_tier=True,
        category="institutional",
        profitability_proxy=0.79,
        ingestion_adapter="sec_13f",
        notes="Detect new institutional long initiations from quarterly 13F filings.",
    ),
    DataSourceRecord(
        id="openinsider_cluster_buys",
        type="csv",
        url="http://openinsider.com/screener",
        status="candidate",
        lift_score=0.11,
        coverage_score=0.35,
        freshness_score=0.80,
        complexity_score=0.34,
        free_tier=True,
        category="insider",
        profitability_proxy=0.74,
        ingestion_adapter="openinsider_csv",
        notes="Free insider cluster buy/sell tape, useful for conviction overlays.",
    ),
    DataSourceRecord(
        id="fmp_earnings_surprises",
        type="api",
        url="https://financialmodelingprep.com/api/v3/earning-surprises",
        auth="FMP_API_KEY",
        status="candidate",
        lift_score=0.12,
        coverage_score=0.45,
        freshness_score=0.86,
        complexity_score=0.25,
        free_tier=True,
        category="earnings",
        profitability_proxy=0.67,
        ingestion_adapter="fmp_earnings",
        notes="Post-earnings drift source from FMP free tier.",
    ),
    DataSourceRecord(
        id="koyfin_insider_news_rss",
        type="rss",
        url="https://www.marketwatch.com/rss/topstories",
        status="candidate",
        lift_score=0.07,
        coverage_score=0.55,
        freshness_score=0.84,
        complexity_score=0.22,
        free_tier=True,
        category="news",
        profitability_proxy=0.52,
        ingestion_adapter="rss_events",
        notes="Free headline stream for catalyst confirmation around smart-money activity.",
    ),
    DataSourceRecord(
        id="gdelt_finance_feed",
        type="rss",
        url="https://api.gdeltproject.org/api/v2/doc/doc",
        status="candidate",
        lift_score=0.08,
        coverage_score=0.35,
        freshness_score=0.82,
        complexity_score=0.35,
        free_tier=True,
        category="news",
        profitability_proxy=0.44,
        ingestion_adapter="gdelt",
        notes="Global media sentiment around positions and macro regimes.",
    ),
)
