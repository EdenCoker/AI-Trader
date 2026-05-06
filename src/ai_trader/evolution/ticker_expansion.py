from __future__ import annotations

import json
from pathlib import Path

from ai_trader.evolution.reports import AgentReport, ReportBuilder
from ai_trader.evolution.watchlist_manager import TickerCandidate, WatchlistManager


class TickerExpansionAgent:
    """Adds high-scoring ticker candidates to the watchlist within strict caps."""

    def __init__(
        self,
        *,
        watchlist_path: Path = Path("data/watchlist.txt"),
        candidate_path: Path = Path("data/ticker_candidates.json"),
        max_adds: int = 20,
        run_id: str = "manual",
        manager: WatchlistManager | None = None,
        candidates: list[TickerCandidate] | None = None,
    ) -> None:
        self.watchlist_path = watchlist_path
        self.candidate_path = candidate_path
        self.max_adds = max_adds
        self.run_id = run_id
        self.manager = manager or WatchlistManager(watchlist_path=watchlist_path)
        self._candidates = candidates

    def run(self) -> AgentReport:
        builder = ReportBuilder("TickerExpansionAgent", self.run_id)
        try:
            candidates = (
                self._candidates
                if self._candidates is not None
                else self._load_candidates()
            )
            change = self.manager.add_candidates(candidates, max_adds=self.max_adds)
            return builder.build(
                status="ok",
                summary={
                    "candidates": len(candidates),
                    "added": list(change.added),
                    "skipped": list(change.skipped),
                    "total_active": change.total_active,
                    "max_adds": self.max_adds,
                },
            )
        except Exception as exc:
            return builder.build(status="failed", errors=[str(exc)])

    def _load_candidates(self) -> list[TickerCandidate]:
        if not self.candidate_path.exists():
            return []
        payload = json.loads(self.candidate_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("candidates", [])
        return [TickerCandidate.model_validate(item) for item in payload]
