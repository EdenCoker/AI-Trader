from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from ai_trader.evolution.reports import AgentReport, ReportBuilder
from ai_trader.evolution.watchlist_manager import WatchlistManager


class IngestionOrchestrator:
    """Runs the training-data ingestion script and maintains the rolling JSONL window."""

    def __init__(
        self,
        *,
        watchlist: Path = Path("data/watchlist.txt"),
        out: Path = Path("logs/rolling_examples.jsonl"),
        candidate_out: Path = Path("logs/candidate_examples.jsonl"),
        rolling_weeks: int = 52,
        max_api_calls: int = 50_000,
        run_id: str = "manual",
    ) -> None:
        self.watchlist = watchlist
        self.out = out
        self.candidate_out = candidate_out
        self.rolling_weeks = rolling_weeks
        self.max_api_calls = max_api_calls
        self.run_id = run_id

    def run(self) -> AgentReport:
        builder = ReportBuilder("IngestionOrchestrator", self.run_id)
        tickers = WatchlistManager(watchlist_path=self.watchlist).load()
        estimated_calls = max(1, len(tickers)) * 25
        if estimated_calls > self.max_api_calls:
            return builder.build(
                status="failed",
                summary={"tickers": len(tickers), "estimated_api_calls": estimated_calls},
                errors=[
                    f"Estimated API calls exceed budget: {estimated_calls}>{self.max_api_calls}"
                ],
            )

        env = os.environ.copy()
        src_path = str(Path("src").resolve())
        env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
        command = [
            sys.executable,
            "scripts/ingest_training_data.py",
            "--out",
            str(self.candidate_out),
        ]
        if tickers:
            command.extend(["--tickers", *tickers])

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            check=False,
        )
        if result.returncode != 0:
            return builder.build(
                status="failed",
                summary={
                    "tickers": len(tickers),
                    "estimated_api_calls": estimated_calls,
                    "returncode": result.returncode,
                    "stderr_tail": (result.stderr or "")[-2000:],
                },
                errors=["ingest_training_data.py failed"],
            )

        compact_summary = self._merge_and_compact()
        quality = self._quality_report()
        return builder.build(
            status="ok",
            summary={
                "tickers": len(tickers),
                "estimated_api_calls": estimated_calls,
                "records": compact_summary["records"],
                "evicted": compact_summary["evicted"],
                "schema_violations": quality["schema_violations"],
                "ticker_coverage_pct": quality["ticker_coverage_pct"],
            },
        )

    def _merge_and_compact(self) -> dict[str, int]:
        records = _read_jsonl_records(self.out) + _read_jsonl_records(self.candidate_out)
        cutoff = datetime.now(UTC).date() - timedelta(weeks=self.rolling_weeks)
        kept: list[dict] = []
        evicted = 0
        seen: set[str] = set()
        for record in records:
            as_of = _record_date(record)
            if as_of is not None and as_of < cutoff:
                evicted += 1
                continue
            key = json.dumps(record, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            kept.append(record)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        with self.out.open("w", encoding="utf-8") as handle:
            for record in kept:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        return {"records": len(kept), "evicted": evicted}

    def _quality_report(self) -> dict:
        records = _read_jsonl_records(self.out)
        violations = 0
        tickers: set[str] = set()
        for record in records:
            if not {"signal_bundle", "trade_plan", "pnl_pct"}.issubset(record):
                violations += 1
                continue
            ticker = (
                record.get("metadata", {}).get("ticker")
                or record.get("signal_bundle", {}).get("ticker")
                or record.get("trade_plan", {}).get("ticker")
            )
            if ticker:
                tickers.add(str(ticker).upper())
        watchlist = WatchlistManager(watchlist_path=self.watchlist).load()
        coverage = (len(tickers & set(watchlist)) / len(watchlist) * 100) if watchlist else 0.0
        report = {
            "records": len(records),
            "schema_violations": violations,
            "unique_tickers": len(tickers),
            "ticker_coverage_pct": round(coverage, 2),
            "created_at": datetime.now(UTC).isoformat(),
        }
        path = self.out.with_suffix(".quality.json")
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return report


def _read_jsonl_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _record_date(record: dict) -> date | None:
    raw = (
        record.get("metadata", {}).get("as_of")
        or record.get("signal_bundle", {}).get("as_of")
        or record.get("trade_plan", {}).get("as_of")
    )
    if not raw:
        return None
    try:
        return date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None
