"""
Live Lobbying Autopilot — implements the lobbying_long strategy on a live paper account.

Strategy: lobbying_long
  - Pull recent lobbying disclosures from Quiver API
  - For each ticker with net-positive lobbying in the last LOOKBACK_DAYS days:
      * Build a SignalBundle with lobbying_activity + fear_greed_macro signals
      * Run through FinalReasoner (LLM + calibrator + insider-news correlation)
      * If conviction >= MIN_CONVICTION and direction == long: place order via IBKR paper
  - Sleep CYCLE_SLEEP_S between full scans
  - Cap sizing at STARTING_BALANCE

Usage:
    python scripts/live_lobby_autopilot.py [--dry-run] [--cycles N]
"""
from __future__ import annotations

import argparse
import datetime
import logging
import sys
import time
from pathlib import Path
from collections import defaultdict

# Ensure project src is importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx
import pandas as pd

from ai_trader.broker.contracts import BrokerOrder, OrderSide
from ai_trader.broker.ibkr import IBKRBroker
from ai_trader.broker.sizing import BalanceSizingConfig, size_order_from_balance
from ai_trader.config import get_settings
from ai_trader.domain.signals import Signal, SignalBundle, SignalDirection
from ai_trader.intelligence.reasoner import FinalReasoner
from ai_trader.intelligence.trade_plan import TradePlan

# Windows-safe logging setup (avoid cp1252 unicode errors)
_stream_handler = logging.StreamHandler(sys.stdout)
try:
    _stream_handler.stream.reconfigure(encoding="utf-8")
except AttributeError:
    pass
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        _stream_handler,
        logging.FileHandler("logs/live_lobby_autopilot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("live_lobby")

# --- Config -----------------------------------------------------------------
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"
LOBBYING_EXCEL = Path("Training Data") / "lobbying-recent.xlsx"
# ----------------------------------------------------------------------------


def fetch_fear_greed() -> float:
    """Return current Fear & Greed index (0–100). Falls back to 50 on error."""
    try:
        r = httpx.get(FEAR_GREED_URL, timeout=10)
        r.raise_for_status()
        return float(r.json()["data"][0]["value"])
    except Exception as e:
        log.warning("Fear & Greed fetch failed (%s), defaulting to 50", e)
        return 50.0


def load_lobbying_tickers(since: datetime.date, lookback_days: int) -> dict[str, float]:
    """
    Return {ticker: total_lobby_amount} for disclosures since `since`.
    Reads from Training Data/lobbying-recent.xlsx (same source as the backtest).
    Falls back to ALL tickers in the file when the window finds nothing.
    """
    if not LOBBYING_EXCEL.exists():
        log.error("Lobbying Excel not found: %s", LOBBYING_EXCEL)
        return {}
    try:
        df = pd.read_excel(LOBBYING_EXCEL, parse_dates=["Date"])
    except Exception as e:
        log.error("Failed to read lobbying Excel: %s", e)
        return {}

    df = df.rename(columns={"Ticker": "ticker", "Date": "date", "Amount": "lobby_amount"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df[df["ticker"].notna() & (df["ticker"] != "") & (df["ticker"] != "NAN")]

    recent = df[df["date"] >= since]
    if recent.empty:
        log.warning(
            "No lobbying disclosures since %s in Excel (last %d days). "
            "Using all tickers as static watchlist.",
            since, lookback_days,
        )
        recent = df  # fall back to full history

    amounts: dict[str, float] = defaultdict(float)
    for _, row in recent.iterrows():
        amounts[str(row["ticker"])] += float(row.get("lobby_amount", 0) or 0)
    return dict(amounts)


def build_bundle(
    ticker: str,
    as_of: datetime.date,
    lobby_amount: float,
    fear_greed: float,
    lookback_days: int,
) -> SignalBundle:
    """Build a SignalBundle with lobbying + fear_greed signals."""
    lobby_strength = min(1.0, lobby_amount / 5_000_000)  # normalise: $5M lobbying = strength 1.0

    fg_direction = (
        SignalDirection.LONG if fear_greed >= 50
        else SignalDirection.NEUTRAL if fear_greed >= 35
        else SignalDirection.SHORT
    )
    fg_strength = abs(fear_greed - 50) / 50  # 0 at 50, 1 at 0 or 100

    return SignalBundle(
        ticker=ticker,
        as_of=as_of,
        signals=(
            Signal(
                name="lobbying_activity",
                ticker=ticker,
                direction=SignalDirection.LONG,
                strength=lobby_strength,
                confidence=0.65,
                effective_date=as_of,
                horizon_days=60,
                reasons=("${:,.0f} lobbying spend (last {}d)".format(lobby_amount, lookback_days),),
            ),
            Signal(
                name="fear_greed_macro",
                ticker=ticker,
                direction=fg_direction,
                strength=fg_strength,
                confidence=0.5,
                effective_date=as_of,
                horizon_days=30,
                reasons=("Fear & Greed index: {:.0f}".format(fear_greed),),
            ),
        ),
    )


def run_cycle(
    *,
    reasoner: FinalReasoner,
    dry_run: bool,
    settings,
    cfg: dict,
) -> list[dict]:
    """Run one full scan: load lobbying -> build bundles -> reason -> trade."""
    today = datetime.date.today()
    since = today - datetime.timedelta(days=cfg["lookback_days"])
    fear_greed = fetch_fear_greed()
    log.info("Fear & Greed: %.0f | scanning lobbying disclosures since %s", fear_greed, since)

    lobby_map = load_lobbying_tickers(since, cfg["lookback_days"])
    if not lobby_map:
        log.warning("No lobbying tickers found -- skipping cycle")
        return []

    # Sort by lobby amount descending; cap to top 20 to limit LLM calls
    top_tickers = sorted(lobby_map, key=lambda t: lobby_map[t], reverse=True)[:20]
    log.info("Top lobbying tickers (%d): %s", len(top_tickers), ", ".join(top_tickers))

    orders_placed = []
    for ticker in top_tickers:
        lobby_amount = lobby_map[ticker]
        bundle = build_bundle(ticker, today, lobby_amount, fear_greed, cfg["lookback_days"])

        try:
            plan: TradePlan = reasoner.reason(
                ticker=ticker,
                as_of=today,
                bundle=bundle,
            )
        except Exception as e:
            log.error("Reasoner failed for %s: %s", ticker, e)
            continue

        log.info(
            "  %s -> direction=%s conviction=%.2f size=%.2f | lobby=$%s",
            ticker, plan.direction.value, plan.conviction, plan.size_multiplier,
            "{:,.0f}".format(lobby_amount),
        )

        if plan.direction is not SignalDirection.LONG:
            log.info("  %s skipped: direction=%s", ticker, plan.direction.value)
            continue
        if plan.conviction < cfg["min_conviction"]:
            log.info("  %s skipped: conviction %.2f < %.2f", ticker, plan.conviction, cfg["min_conviction"])
            continue

        # Size & place order
        if dry_run:
            log.info("  [DRY-RUN] Would BUY %s (conviction=%.2f)", ticker, plan.conviction)
            orders_placed.append({"ticker": ticker, "dry_run": True, "conviction": plan.conviction})
            continue

        try:
            with IBKRBroker.from_settings(settings) as broker:
                account_snap = broker.account_snapshot()
                # Always size from the live balance so profits compound into future positions.
                # cash_fraction limits per-trade exposure regardless of balance size.
                quote = broker.market_price(ticker)
                sizing = size_order_from_balance(
                    plan=plan,
                    side=OrderSide.BUY,
                    account=account_snap,
                    quote=quote,
                    config=BalanceSizingConfig(cash_fraction=cfg["cash_fraction"]),
                )
                order = BrokerOrder(ticker=ticker, side=OrderSide.BUY, quantity=sizing.quantity)
                result = broker.place_order(order)

            log.info(
                "  ORDER PLACED: BUY %s x%.4f @ ~$%.2f | notional=$%s | orderId=%s",
                ticker, sizing.quantity, quote.price,
                "{:,.0f}".format(sizing.order_notional), result.order_id,
            )
            orders_placed.append({
                "ticker": ticker,
                "quantity": sizing.quantity,
                "price": quote.price,
                "notional": sizing.order_notional,
                "conviction": plan.conviction,
                "order_id": result.order_id,
            })
        except Exception as e:
            log.error("  Order failed for %s: %s", ticker, e)

    return orders_placed


def main() -> None:
    parser = argparse.ArgumentParser(description="Live lobbying autopilot — paper account")
    parser.add_argument("--dry-run", action="store_true", help="Simulate orders without submitting to IBKR")
    parser.add_argument("--cycles", type=int, default=None, help="Run N cycles then stop (default: run forever)")
    parser.add_argument("--min-conviction", type=float, default=0.30)
    parser.add_argument("--cash-fraction", type=float, default=0.02)
    parser.add_argument("--starting-balance", type=float, default=999_000.0)
    parser.add_argument("--cycle-sleep", type=int, default=3600)
    parser.add_argument("--lookback-days", type=int, default=30)
    args = parser.parse_args()

    cfg = {
        "min_conviction": args.min_conviction,
        "cash_fraction": args.cash_fraction,
        "starting_balance": args.starting_balance,
        "cycle_sleep": args.cycle_sleep,
        "lookback_days": args.lookback_days,
    }

    settings = get_settings()
    mode_label = "DRY-RUN" if args.dry_run else "LIVE ({})".format(settings.trading_mode.upper())
    log.info("=" * 60)
    log.info("Live Lobbying Autopilot starting")
    log.info("  Mode:             %s", mode_label)
    log.info("  Account:          %s (port %s)", settings.ibkr_account, settings.ibkr_port)
    log.info("  Starting balance: $%s", "{:,.0f}".format(cfg["starting_balance"]))
    log.info("  Cash fraction:    %.1f%%", cfg["cash_fraction"] * 100)
    log.info("  Min conviction:   %.2f", cfg["min_conviction"])
    log.info("  Lookback days:    %d", cfg["lookback_days"])
    log.info("  Cycle sleep:      %ds", cfg["cycle_sleep"])
    log.info("  Lobbying source:  %s", LOBBYING_EXCEL)
    log.info("=" * 60)

    reasoner = FinalReasoner(settings=settings)
    cycle = 0

    try:
        while True:
            cycle += 1
            log.info("---- Cycle %d ----", cycle)
            placed = run_cycle(
                reasoner=reasoner,
                dry_run=args.dry_run,
                settings=settings,
                cfg=cfg,
            )
            log.info("Cycle %d complete: %d orders placed/simulated", cycle, len(placed))

            if args.cycles is not None and cycle >= args.cycles:
                log.info("Reached --cycles %d, stopping.", args.cycles)
                break

            log.info("Sleeping %ds until next cycle (Ctrl-C to stop)...", cfg["cycle_sleep"])
            time.sleep(cfg["cycle_sleep"])
    except KeyboardInterrupt:
        log.info("Interrupted -- autopilot stopped.")


if __name__ == "__main__":
    main()
