from __future__ import annotations

from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
except ImportError:  # pragma: no cover - fallback for minimal test environments.
    Console = None
    Table = None

from ai_trader.self_improvement.git_store import GitProposalStore
from ai_trader.self_improvement.post_trade_review import PostTradeReviewer
from ai_trader.self_improvement.proposal import TradeOutcome, should_review


class NightlyReviewScheduler:
    def __init__(
        self,
        *,
        reviewer: PostTradeReviewer | None = None,
        store: GitProposalStore | None = None,
        console=None,
    ) -> None:
        self._reviewer = reviewer or PostTradeReviewer()
        self._store = store or GitProposalStore()
        self._console = console or (Console() if Console is not None else None)

    def run(self, outcomes_file: Path) -> dict[str, int]:
        summary = {"created": 0, "skipped": 0, "failed": 0}
        for line in outcomes_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                outcome = TradeOutcome.model_validate_json(line)
                if not should_review(outcome):
                    summary["skipped"] += 1
                    continue
                proposal = self._reviewer.review(outcome)
                if proposal is None:
                    summary["skipped"] += 1
                    continue
                self._store.submit(proposal)
                summary["created"] += 1
            except Exception:
                summary["failed"] += 1

        self._render(summary)
        return summary

    def _render(self, summary: dict[str, int]) -> None:
        if self._console is None or Table is None:
            return
        table = Table(title="Nightly Review")
        table.add_column("Status")
        table.add_column("Count", justify="right")
        for key, value in summary.items():
            table.add_row(key, str(value))
        self._console.print(table)
