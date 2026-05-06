"""Print a compact status table for weekly evolution runs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    from rich.console import Console
    from rich.table import Table
except ImportError:  # pragma: no cover
    Console = None
    Table = None


def main() -> None:
    rows = [_row(path) for path in sorted(Path("logs").glob("weekly_weekly_*.json"))]
    rows = [row for row in rows if row is not None]
    if Console is None or Table is None:
        _plain(rows)
        return
    table = Table(title="Evolution Status")
    columns = ("Week", "Model", "Training Ex", "Tickers", "BT Wins", "Promoted", "Sharpe Delta")
    for column in columns:
        table.add_column(column)
    for row in rows:
        table.add_row(
            row["week"],
            row["model"],
            row["training_examples"],
            row["tickers"],
            row["bt_wins"],
            row["promoted"],
            row["sharpe_delta"],
        )
    Console().print(table)


def _row(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    steps = payload.get("steps", [])
    training = _step(steps, "TrainingAgent")
    ticker = _step(steps, "TickerExpansionAgent")
    gate = _step(steps, "PromotionGate")
    promotion = payload.get("promotion") or {}
    gate_summary = gate.get("summary", {})
    run_id = str(payload.get("run_id", path.stem)).replace("weekly_", "")
    return {
        "week": run_id,
        "model": str(promotion.get("version") or "-"),
        "training_examples": str(training.get("summary", {}).get("training_count", "-")),
        "tickers": str(ticker.get("summary", {}).get("total_active", "-")),
        "bt_wins": f"{gate_summary.get('wins', '-')}/{gate_summary.get('required', '-')}",
        "promoted": "Yes" if gate_summary.get("promoted") else "No",
        "sharpe_delta": str(gate_summary.get("sharpe_delta", "-")),
    }


def _step(steps: list[dict], agent: str) -> dict:
    return next((step for step in steps if step.get("agent") == agent), {})


def _plain(rows: list[dict]) -> None:
    headers = ("Week", "Model", "Training Ex", "Tickers", "BT Wins", "Promoted", "Sharpe Delta")
    print(" | ".join(headers))
    for row in rows:
        print(
            " | ".join(
                [
                    row["week"],
                    row["model"],
                    row["training_examples"],
                    row["tickers"],
                    row["bt_wins"],
                    row["promoted"],
                    row["sharpe_delta"],
                ]
            )
        )


if __name__ == "__main__":
    main()
