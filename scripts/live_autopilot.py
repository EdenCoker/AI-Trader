"""
General live autopilot with horizon-aware position management.

It evaluates the expanded signal set, reasons into TradePlan objects, prevents
same-ticker/same-horizon duplicates, and maintains logs/open_positions.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ingest_training_data as ingest

from ai_trader.alerts import send_alert
from ai_trader.broker.contracts import (
    BrokerAccountSnapshot,
    BrokerOrder,
    BrokerQuote,
    OrderSide,
)
from ai_trader.broker.ibkr import IBKRBroker
from ai_trader.broker.sizing import BalanceSizingConfig, size_order_from_balance
from ai_trader.config import get_settings
from ai_trader.domain.signals import SignalBundle, SignalDirection
from ai_trader.intelligence.reasoner import FinalReasoner
from ai_trader.intelligence.trade_plan import HorizonClass, TradePlan

LEDGER_PATH = Path("logs/open_positions.json")
LOG_PATH = Path("logs/live_autopilot.log")
HORIZON_ALLOC: dict[HorizonClass, float] = {"short": 0.40, "medium": 0.35, "long": 0.25}
STOP_LOSS: dict[HorizonClass, float] = {"short": 0.08, "medium": 0.12, "long": 0.15}

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_PATH, encoding="utf-8")],
)
log = logging.getLogger("live_autopilot")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run horizon-aware live autopilot")
    parser.add_argument(
        "--tickers",
        nargs="*",
        help="Ticker universe. Defaults to data/watchlist.txt",
    )
    parser.add_argument("--watchlist", default="data/watchlist.txt")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--record-dry-run", action="store_true")
    parser.add_argument("--cycles", type=int, default=None)
    parser.add_argument("--cycle-sleep", type=int, default=3600)
    parser.add_argument("--lookback-days", type=int, default=45)
    parser.add_argument("--min-conviction", type=float, default=0.30)
    parser.add_argument("--cash-fraction", type=float, default=0.02)
    parser.add_argument("--starting-balance", type=float, default=100_000.0)
    args = parser.parse_args()

    settings = get_settings()
    tickers = _load_tickers(args.tickers, Path(args.watchlist))
    if not tickers:
        raise SystemExit("No tickers supplied and watchlist is empty")

    stop = False

    def _stop(_sig, _frame):
        nonlocal stop
        stop = True
        log.info("Stopping after current cycle")

    signal.signal(signal.SIGINT, _stop)

    log.info("Autopilot starting: %d tickers, dry_run=%s", len(tickers), args.dry_run)
    reasoner = FinalReasoner(settings=settings)
    cycle = 0
    while not stop:
        cycle += 1
        if args.cycles is not None and cycle > args.cycles:
            break
        log.info("---- cycle %d ----", cycle)
        try:
            run_cycle(
                tickers=tickers,
                settings=settings,
                reasoner=reasoner,
                dry_run=args.dry_run,
                record_dry_run=args.record_dry_run,
                min_conviction=args.min_conviction,
                cash_fraction=args.cash_fraction,
                starting_balance=args.starting_balance,
                lookback_days=args.lookback_days,
            )
        except Exception as exc:
            log.exception("cycle failed: %s", exc)
        if stop or (args.cycles is not None and cycle >= args.cycles):
            break
        time.sleep(args.cycle_sleep)


def run_cycle(
    *,
    tickers: list[str],
    settings,
    reasoner: FinalReasoner,
    dry_run: bool,
    record_dry_run: bool,
    min_conviction: float,
    cash_fraction: float,
    starting_balance: float,
    lookback_days: int,
) -> None:
    today = dt.date.today()
    start = today - dt.timedelta(days=lookback_days)
    ledger = _load_ledger()
    prices = _current_prices(tickers, settings)
    ledger = _manage_exits(ledger, prices, settings, dry_run=dry_run)

    datasets = _load_live_datasets(tickers, start, today)
    account = _account_snapshot(settings, starting_balance, dry_run=dry_run)
    horizon_used = _horizon_notional(ledger)

    for ticker in tickers:
        price = prices.get(ticker)
        if price is None:
            log.info("%s skipped: no current price", ticker)
            continue
        bundle = _build_live_bundle(ticker, today, datasets, settings)
        if not bundle.signals:
            continue
        plan = reasoner.reason(ticker=ticker, as_of=today, bundle=bundle)
        log.info(
            "%s plan: %s conviction=%.2f horizon=%s signals=%d",
            ticker,
            plan.direction.value,
            plan.conviction,
            plan.horizon_class,
            len(bundle.signals),
        )
        if plan.conviction >= 0.50:
            send_alert(
                f"AI-Trader high-conviction signal {ticker}",
                f"{ticker} {plan.direction.value} conviction={plan.conviction:.2f} "
                f"horizon={plan.horizon_class}",
            )
        if plan.direction is SignalDirection.NEUTRAL or plan.conviction < min_conviction:
            continue
        key = _position_key(ticker, plan.horizon_class)
        if key in ledger:
            log.info("%s skipped: duplicate open %s horizon position", ticker, plan.horizon_class)
            continue
        horizon_capital = account.spendable_balance * HORIZON_ALLOC[plan.horizon_class]
        remaining_horizon_capital = max(
            0.0,
            horizon_capital - horizon_used.get(plan.horizon_class, 0.0),
        )
        if remaining_horizon_capital <= 0:
            log.info("%s skipped: %s horizon allocation already used", ticker, plan.horizon_class)
            continue
        side = OrderSide.BUY if plan.direction is SignalDirection.LONG else OrderSide.SELL
        sizing = size_order_from_balance(
            plan=plan,
            side=side,
            account=account.model_copy(update={"available_funds": remaining_horizon_capital}),
            quote=BrokerQuote(ticker=ticker, price=price, source="polygon"),
            config=BalanceSizingConfig(cash_fraction=cash_fraction),
        )
        if sizing.quantity <= 0:
            continue
        order = BrokerOrder(ticker=ticker, side=side, quantity=sizing.quantity)
        if dry_run:
            log.info("[dry-run] would submit %s %s x%.4f", side.value, ticker, order.quantity)
            if not record_dry_run:
                continue
            result_payload = {"status": "dry_run"}
        else:
            with IBKRBroker.from_settings(settings) as broker:
                result = broker.place_order(order)
            result_payload = result.model_dump(mode="json")
            send_alert("AI-Trader trade placed", f"{side.value} {ticker} x{sizing.quantity:.4f}")

        ledger[key] = _ledger_entry(
            plan=plan,
            quantity=sizing.quantity,
            entry_price=price,
            notional=sizing.order_notional,
            order_result=result_payload,
        )
        horizon_used[plan.horizon_class] = (
            horizon_used.get(plan.horizon_class, 0.0) + sizing.order_notional
        )

    _save_ledger(ledger)


def _load_live_datasets(tickers: list[str], start: dt.date, today: dt.date) -> dict:
    return {
        "insider": ingest.load_insider_trades(tickers, start, today),
        "macro": ingest.load_fred_macro(start, today),
        "earnings": ingest.load_earnings_surprises(tickers, start, today),
        "options": ingest.load_options_put_call_ratios(tickers, today),
        "institutional": ingest.load_13f_changes(tickers, start, today),
        "short_interest": ingest.load_short_interest(tickers, start, today),
    }


def _build_live_bundle(ticker: str, today: dt.date, datasets: dict, settings) -> SignalBundle:
    price_start = today - dt.timedelta(days=45)
    price_df = (
        ingest.fetch_polygon_prices(
            ticker,
            price_start,
            today,
            settings.polygon_api_key.get_secret_value(),
        )
        if settings.polygon_api_key is not None
        else None
    )
    price_ctx = ingest._price_context(price_df, today) if price_df is not None else {}
    insider = _latest_row(datasets["insider"], ticker, today)
    earnings = _latest_row(datasets["earnings"], ticker, today)
    options = _latest_row(datasets["options"], ticker, today)
    inst = _latest_row(datasets["institutional"], ticker, today)
    short_interest = _latest_row(datasets["short_interest"], ticker, today)
    macro = ingest._macro_for_date(datasets["macro"], today)
    return ingest.build_signal_bundle(
        ticker,
        today,
        insider_buy_qty=_num(insider.get("insider_buy_qty")),
        insider_sell_qty=_num(insider.get("insider_sell_qty")),
        insider_net_qty=_num(insider.get("insider_net_qty")),
        insider_value_usd=_num(insider.get("insider_value_usd")),
        insider_officer_count=int(_num(insider.get("insider_officer_count"))),
        insider_director_count=int(_num(insider.get("insider_director_count"))),
        yield_spread_2_10=macro.get("yield_spread_2_10"),
        cpi_mom=macro.get("cpi_mom"),
        ism_pmi=macro.get("ism_pmi"),
        unemployment_claims=macro.get("unemployment_claims"),
        fed_funds_rate=macro.get("fed_funds_rate"),
        eps_actual=_num(earnings.get("eps_actual")),
        eps_estimate=_num(earnings.get("eps_estimate")),
        eps_surprise_pct=_num(earnings.get("eps_surprise_pct")),
        put_call_ratio=_num(options.get("put_call_ratio")),
        put_open_interest=_num(options.get("put_open_interest")),
        call_open_interest=_num(options.get("call_open_interest")),
        price_level=str(price_ctx.get("price_level", "unknown")),
        price_momentum_20d=_num(price_ctx.get("price_momentum_20d")),
        institutional_delta_shares=_num(inst.get("institutional_delta_shares")),
        institutional_delta_pct=_num(inst.get("institutional_delta_pct")),
        institutional_manager=str(inst.get("institutional_manager") or ""),
        institutional_market_value_usd=_num(inst.get("institutional_market_value_usd")),
        short_interest_shares=_num(short_interest.get("short_interest_shares")),
        days_to_cover=_num(short_interest.get("days_to_cover")),
        short_interest_change_pct=_num(short_interest.get("short_interest_change_pct")),
    )


def _manage_exits(ledger: dict, prices: dict[str, float], settings, *, dry_run: bool) -> dict:
    today = dt.date.today()
    updated = dict(ledger)
    for key, position in list(ledger.items()):
        ticker = position["ticker"]
        horizon = position["horizon_class"]
        price = prices.get(ticker)
        if price is None:
            continue
        exit_date = dt.date.fromisoformat(position["exit_date"])
        entry_price = float(position["entry_price"])
        direction = position["direction"]
        stop = STOP_LOSS[horizon]
        stop_hit = (
            direction == "long" and price <= entry_price * (1.0 - stop)
        ) or (
            direction == "short" and price >= entry_price * (1.0 + stop)
        )
        time_exit = today >= exit_date
        if not (stop_hit or time_exit):
            continue
        reason = "stop_loss" if stop_hit else "time_exit"
        side = OrderSide.SELL if direction == "long" else OrderSide.BUY
        quantity = float(position["quantity"])
        log.info("closing %s %s x%.4f: %s", ticker, side.value, quantity, reason)
        if not dry_run:
            with IBKRBroker.from_settings(settings) as broker:
                broker.place_order(BrokerOrder(ticker=ticker, side=side, quantity=quantity))
            if stop_hit:
                send_alert(
                    "AI-Trader stop loss triggered",
                    f"{ticker} {horizon} stop loss at {price:.2f}",
                )
        updated.pop(key, None)
    return updated


def _current_prices(tickers: list[str], settings) -> dict[str, float]:
    if settings.polygon_api_key is None:
        return {}
    prices = {}
    today = dt.date.today()
    start = today - dt.timedelta(days=10)
    for ticker in tickers:
        frame = ingest.fetch_polygon_prices(
            ticker,
            start,
            today,
            settings.polygon_api_key.get_secret_value(),
        )
        if not frame.empty:
            prices[ticker] = float(frame.iloc[-1]["close"])
    return prices


def _account_snapshot(settings, starting_balance: float, *, dry_run: bool) -> BrokerAccountSnapshot:
    if dry_run:
        return BrokerAccountSnapshot(
            available_funds=starting_balance,
            net_liquidation=starting_balance,
        )
    with IBKRBroker.from_settings(settings) as broker:
        return broker.account_snapshot()


def _ledger_entry(
    *,
    plan: TradePlan,
    quantity: float,
    entry_price: float,
    notional: float,
    order_result: dict,
) -> dict:
    exit_date = plan.as_of + dt.timedelta(days=plan.holding_period_days)
    return {
        "ticker": plan.ticker,
        "horizon_class": plan.horizon_class,
        "direction": plan.direction.value,
        "quantity": quantity,
        "entry_price": entry_price,
        "entry_date": plan.as_of.isoformat(),
        "exit_date": exit_date.isoformat(),
        "stop_loss_pct": STOP_LOSS[plan.horizon_class],
        "conviction": plan.conviction,
        "notional": notional,
        "order_result": order_result,
    }


def _load_ledger() -> dict:
    if not LEDGER_PATH.exists():
        return {}
    try:
        payload = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if isinstance(payload, list):
        return {_position_key(item["ticker"], item["horizon_class"]): item for item in payload}
    return payload if isinstance(payload, dict) else {}


def _save_ledger(ledger: dict) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")


def _horizon_notional(ledger: dict) -> dict[HorizonClass, float]:
    used: dict[HorizonClass, float] = {"short": 0.0, "medium": 0.0, "long": 0.0}
    for position in ledger.values():
        horizon = position.get("horizon_class")
        if horizon in used:
            used[horizon] += float(position.get("notional") or 0.0)
    return used


def _position_key(ticker: str, horizon: str) -> str:
    return f"{ticker.upper()}:{horizon}"


def _latest_row(frame, ticker: str, as_of: dt.date) -> dict:
    if frame is None or frame.empty or "ticker" not in frame or "date" not in frame:
        return {}
    sub = frame[(frame["ticker"] == ticker.upper()) & (frame["date"] <= as_of)]
    if sub.empty:
        return {}
    return sub.sort_values("date").iloc[-1].to_dict()


def _load_tickers(cli_tickers: list[str] | None, watchlist_path: Path) -> list[str]:
    values = cli_tickers or []
    if not values and watchlist_path.exists():
        values = [
            line.strip()
            for line in watchlist_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    return sorted({ticker.upper() for ticker in values if ticker.strip()})


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
