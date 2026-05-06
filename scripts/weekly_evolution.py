"""Weekly autonomous evolution cycle."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_trader.alerts import send_alert
from ai_trader.evolution.discovery import DiscoveryAgent
from ai_trader.evolution.ingestion_orchestrator import IngestionOrchestrator
from ai_trader.evolution.promoter import ModelPromoter
from ai_trader.evolution.promotion_gate import PromotionGate
from ai_trader.evolution.reports import AgentReport
from ai_trader.evolution.ticker_expansion import TickerExpansionAgent
from ai_trader.evolution.training_agent import TrainingAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the weekly autonomous evolution cycle")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--no-alerts", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or dt.datetime.now(dt.UTC).strftime("weekly_%Y%m%d")
    report: dict = {
        "run_id": run_id,
        "started_at": dt.datetime.now(dt.UTC).isoformat(),
        "steps": [],
    }

    def append(step: AgentReport) -> AgentReport:
        report["steps"].append(step.model_dump(mode="json"))
        return step

    discovery = append(DiscoveryAgent(run_id=run_id).run())
    expansion = append(TickerExpansionAgent(max_adds=20, run_id=run_id).run())

    if not discovery.ok or not expansion.ok:
        report["status"] = "failed"
        _finish(report, run_id, alert=not args.no_alerts)
        return

    if not args.skip_ingest:
        ingest = append(
            IngestionOrchestrator(
                watchlist=Path("data/watchlist.txt"),
                out=Path("logs/rolling_examples.jsonl"),
                rolling_weeks=52,
                run_id=run_id,
            ).run()
        )
        if not ingest.ok:
            report["status"] = "failed"
            _finish(report, run_id, alert=not args.no_alerts)
            return

    if not args.skip_training:
        training = append(
            TrainingAgent(
                examples=Path("logs/rolling_examples.jsonl"),
                out=Path("data/models/candidate.json"),
                run_id=run_id,
            ).run()
        )
        if not training.ok:
            report["status"] = "failed"
            _finish(report, run_id, alert=not args.no_alerts)
            return

    gate = append(
        PromotionGate(
            current=Path("data/models/production.json"),
            candidate=Path("data/models/candidate.json"),
            pool=Path("data/backtest_pool.json"),
            k=3,
            tolerance=0.05,
            run_id=run_id,
        ).run()
    )

    promoter = ModelPromoter()
    gate_summary = gate.summary
    if gate.ok and gate_summary.get("promoted"):
        promotion = promoter.promote(
            candidate=Path("data/models/candidate.json"),
            reason=str(gate_summary.get("reason", "promotion gate passed")),
            gate_result=gate_summary,
        )
        report["promotion"] = promotion
        _send("model_promoted", gate_summary, enabled=not args.no_alerts)
    elif gate.ok and Path("data/models/candidate.json").exists():
        rejection = promoter.reject_candidate(
            candidate=Path("data/models/candidate.json"),
            reason=str(gate_summary.get("reason", "promotion gate rejected candidate")),
            gate_result=gate_summary,
        )
        report["rejection"] = rejection
        _send("model_rejected", gate_summary, enabled=not args.no_alerts)
    else:
        _send("model_gate_failed", {"errors": gate.errors}, enabled=not args.no_alerts)

    all_steps_ok = all(step.get("status") == "ok" for step in report["steps"])
    report["status"] = "ok" if all_steps_ok else "failed"
    _finish(report, run_id, alert=False)


def _finish(report: dict, run_id: str, *, alert: bool) -> None:
    report["finished_at"] = dt.datetime.now(dt.UTC).isoformat()
    path = Path("logs") / f"weekly_{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if alert:
        _send("weekly_evolution_failed", report, enabled=True)
    print(json.dumps({"status": report.get("status"), "summary": str(path)}, indent=2))


def _send(subject: str, payload: dict, *, enabled: bool) -> None:
    if not enabled:
        return
    send_alert(subject, json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
