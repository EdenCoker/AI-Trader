from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
except ImportError:  # pragma: no cover - fallback for minimal test environments.
    Console = None
    Table = None

from ai_trader.config import AppSettings, get_settings
from ai_trader.self_improvement.git_store import GitProposalStore
from ai_trader.self_improvement.post_trade_review import PostTradeReviewer
from ai_trader.self_improvement.proposal import TradeOutcome, should_review
from ai_trader.training import (
    LocalCalibratorTrainer,
    filter_examples_by_horizon,
    load_training_examples,
)


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


class CalibratorRetrainScheduler:
    def __init__(
        self,
        *,
        settings: AppSettings | None = None,
        history_path: Path = Path("logs/calibrator_history.jsonl"),
        min_new_examples: int = 500,
        min_interval: timedelta = timedelta(days=7),
    ) -> None:
        self._settings = settings or get_settings()
        self._history_path = history_path
        self._min_new_examples = min_new_examples
        self._min_interval = min_interval

    def run_if_due(self, examples_file: Path) -> dict:
        examples = load_training_examples(examples_file)
        now = datetime.now(UTC)
        last = self._last_history()
        last_count = int(last.get("examples_total", 0)) if last else 0
        last_at = _parse_timestamp(last.get("trained_at")) if last else None
        new_count = len(examples) - last_count

        due_by_count = new_count > self._min_new_examples
        due_by_time = last_at is None or now - last_at >= self._min_interval
        if not (due_by_count and due_by_time):
            result = {
                "status": "skipped",
                "examples_total": len(examples),
                "new_examples": max(0, new_count),
                "due_by_count": due_by_count,
                "due_by_time": due_by_time,
            }
            self._append_history({**result, "checked_at": now.isoformat()})
            return result

        trainer = LocalCalibratorTrainer()
        paths = {
            "short": self._settings.local_calibrator_short_path,
            "medium": self._settings.local_calibrator_medium_path,
            "long": self._settings.local_calibrator_long_path,
        }
        outputs = []
        for horizon, path in paths.items():
            subset = filter_examples_by_horizon(examples, horizon)  # type: ignore[arg-type]
            if not subset:
                outputs.append({"horizon": horizon, "status": "skipped", "examples": 0})
                continue
            model = trainer.train(subset)
            model.save(path)
            outputs.append(
                {
                    "horizon": horizon,
                    "status": "trained",
                    "examples": model.training_count,
                    "model_out": str(path),
                    "metrics": model.metrics,
                }
            )

        result = {
            "status": "trained",
            "trained_at": now.isoformat(),
            "examples_total": len(examples),
            "new_examples": max(0, new_count),
            "outputs": outputs,
        }
        self._append_history(result)
        return result

    def _last_history(self) -> dict | None:
        if not self._history_path.exists():
            return None
        records = []
        for line in self._history_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "examples_total" in record:
                records.append(record)
        return records[-1] if records else None

    def _append_history(self, payload: dict) -> None:
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        with self._history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _parse_timestamp(value) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed
