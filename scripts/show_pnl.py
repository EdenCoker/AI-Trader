"""Show unrealised P&L for logs/open_positions.json, grouped by horizon."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ingest_training_data as ingest

from ai_trader.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Show open-position unrealised P&L")
    parser.add_argument("--ledger", default="logs/open_positions.json")
    args = parser.parse_args()

    positions = _load_positions(Path(args.ledger))
    if not positions:
        print("No open positions.")
        return

    settings = get_settings()
    if settings.polygon_api_key is None:
        raise SystemExit("POLYGON_API_KEY is required for live P&L")
    api_key = settings.polygon_api_key.get_secret_value()

    today = dt.date.today()
    tickers = sorted({position["ticker"] for position in positions})
    prices = {ticker: _last_close(ticker, api_key, today) for ticker in tickers}
    horizon_totals = defaultdict(lambda: {"notional": 0.0, "pnl": 0.0})

    header = (
        f"{'Ticker':<8} {'Hz':<7} {'Dir':<6} {'Qty':>10} {'Entry':>10} "
        f"{'Last':>10} {'P&L%':>9} {'P&L$':>12} {'vs SPY':>9}"
    )
    print(header)
    print("-" * 92)
    for position in positions:
        ticker = position["ticker"]
        last = prices.get(ticker)
        if last is None:
            continue
        entry = float(position["entry_price"])
        quantity = float(position["quantity"])
        direction = position["direction"]
        multiplier = 1.0 if direction == "long" else -1.0
        pnl_pct = ((last - entry) / entry) * multiplier
        notional = entry * quantity
        pnl_amount = notional * pnl_pct
        spy = _spy_return(position["entry_date"], today, api_key)
        vs_spy = pnl_pct - spy if spy is not None else None
        horizon = position["horizon_class"]
        horizon_totals[horizon]["notional"] += notional
        horizon_totals[horizon]["pnl"] += pnl_amount
        print(
            f"{ticker:<8} {horizon:<7} {direction:<6} {quantity:>10.2f} "
            f"{entry:>10.2f} {last:>10.2f} {pnl_pct*100:>+8.2f}% "
            f"{pnl_amount:>+12,.2f} {_fmt_pct(vs_spy):>9}"
        )

    print("-" * 92)
    for horizon in ("short", "medium", "long"):
        total = horizon_totals[horizon]
        if total["notional"] <= 0:
            continue
        pnl_pct = total["pnl"] / total["notional"]
        print(
            f"{horizon:<8} total notional=${total['notional']:,.2f} "
            f"pnl=${total['pnl']:+,.2f} ({pnl_pct:+.2%})"
        )


def _load_positions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return list(payload.values())
    if isinstance(payload, list):
        return payload
    return []


def _last_close(ticker: str, api_key: str, today: dt.date) -> float | None:
    frame = ingest.fetch_polygon_prices(ticker, today - dt.timedelta(days=10), today, api_key)
    if frame.empty:
        return None
    return float(frame.iloc[-1]["close"])


def _spy_return(entry_date: str, today: dt.date, api_key: str) -> float | None:
    start = dt.date.fromisoformat(entry_date[:10])
    frame = ingest.fetch_polygon_prices("SPY", start, today, api_key)
    if frame.empty or len(frame) < 2:
        return None
    entry = float(frame.iloc[0]["close"])
    last = float(frame.iloc[-1]["close"])
    return (last - entry) / entry if entry else None


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2%}"


if __name__ == "__main__":
    main()
