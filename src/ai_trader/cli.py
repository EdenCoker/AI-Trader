from __future__ import annotations

import json
import logging
import signal
import time
from datetime import date
from pathlib import Path
from typing import Annotated

import typer
try:
    from rich import print as rprint
except ImportError:  # pragma: no cover - fallback for minimal environments.
    rprint = print

from ai_trader.backtesting.engine import WalkForwardConfig, WalkForwardEngine, WalkForwardResult
from ai_trader.backtesting.monte_carlo import StressMonteCarlo
from ai_trader.bridge.bridge_server import BridgeServer
from ai_trader.build_loop.loop import BuildLoop
from ai_trader.broker.contracts import BrokerOrder, OrderSide
from ai_trader.broker.ibkr import IBKRBroker
from ai_trader.config import get_settings
from ai_trader.domain.signals import SignalBundle, SignalDirection
from ai_trader.intelligence.trade_plan import TradePlan
from ai_trader.intelligence.models import NarrativeIntelligence
from ai_trader.intelligence.narrative import NarrativeAnalyzer
from ai_trader.intelligence.reasoner import FinalReasoner
from ai_trader.rag.trader_rag import format_retrieved, get_trader_rag
from ai_trader.self_improvement.scheduler import NightlyReviewScheduler
from ai_trader.training import LocalCalibratorTrainer, load_training_examples


app = typer.Typer(add_completion=False, no_args_is_help=True)
backtest_app = typer.Typer(no_args_is_help=True)
build_loop_app = typer.Typer(no_args_is_help=True)
train_app = typer.Typer(no_args_is_help=True)
app.add_typer(backtest_app, name="backtest")
app.add_typer(build_loop_app, name="build-loop")
app.add_typer(train_app, name="train")


def _configure_logging() -> None:
    settings = get_settings()
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("logs/ai_trader.log")],
        force=True,
    )


@app.command()
def status() -> None:
    """Show runtime config status (secrets redacted)."""

    _configure_logging()
    settings = get_settings()
    rprint(settings.redacted())
    rprint(settings.provider_status())


@app.command("analyze-news")
def analyze_news(
    ticker: str = typer.Option(..., help="Ticker symbol"),
    headline: str = typer.Option(..., help="Headline text"),
    body: str | None = typer.Option(None, help="Body text"),
    body_file: Path | None = typer.Option(None, exists=True, readable=True, help="Body file path"),
    as_of: str | None = typer.Option(None, help="As-of date (YYYY-MM-DD). Defaults to today."),
    market_context: str | None = typer.Option(None, help="Optional market context text"),
    analyst_context: str | None = typer.Option(None, help="Optional analyst expectations text"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Write JSON to file (UTF-8, no BOM) instead of stdout"),
) -> None:
    """Run the 3-stage narrative analyzer and print JSON."""

    _configure_logging()
    if body_file is not None:
        body_text = body_file.read_text(encoding="utf-8")
    else:
        body_text = body or ""
    if not body_text.strip():
        raise typer.BadParameter("Provide --body or --body-file")

    as_of_date = date.today() if as_of is None else date.fromisoformat(as_of)
    analyzer = NarrativeAnalyzer()
    result = analyzer.analyze(
        ticker=ticker,
        as_of=as_of_date,
        headline=headline,
        body=body_text,
        market_context=market_context,
        analyst_context=analyst_context,
    )
    json_str = result.model_dump_json(indent=2)
    if output is not None:
        output.write_text(json_str, encoding="utf-8")
    else:
        print(json_str)

@app.command("ibkr-positions")
def ibkr_positions() -> None:
    """Connect to IBKR and print current positions."""

    _configure_logging()
    with IBKRBroker.from_settings() as broker:
        positions = broker.positions()
    rprint([pos.model_dump() for pos in positions])


@app.command("rag-index")
def rag_index(
    rebuild: bool = typer.Option(False, help="Rebuild the index even if it exists."),
) -> None:
    """Build the trader RAG index from a corpus directory."""

    _configure_logging()
    settings = get_settings()
    rag = get_trader_rag(settings)
    index_dir = settings.rag_index_dir
    chunks_path = index_dir / "chunks.jsonl"
    vectors_path = index_dir / "vectors.npy"

    if not rebuild and chunks_path.exists() and vectors_path.exists():
        rag.load()
        rprint({"status": "already_indexed", "index_dir": str(index_dir), **rag.stats()})
        return

    rag.build()
    rprint({"status": "indexed", "index_dir": str(index_dir), **rag.stats()})


@app.command("rag-query")
def rag_query(
    query: str = typer.Option(..., help="Query text"),
    k: int = typer.Option(3, min=1, max=10, help="Number of chunks to return"),
) -> None:
    """Query the trader RAG index and print the top results."""

    _configure_logging()
    settings = get_settings()
    rag = get_trader_rag(settings)
    results = rag.retrieve(query, k=k)
    rprint(format_retrieved(results))


@app.command("reason")
def reason(
    bundle_file: Path = typer.Option(..., exists=True, readable=True, help="SignalBundle JSON file"),
    narrative_file: Path | None = typer.Option(
        None, exists=True, readable=True, help="Optional NarrativeIntelligence JSON file"
    ),
    ticker: str | None = typer.Option(None, help="Override ticker (defaults to bundle ticker)"),
    as_of: str | None = typer.Option(None, help="Override as-of date (YYYY-MM-DD)"),
    position_context: str | None = typer.Option(None, help="Optional current position context text"),
    rag: bool | None = typer.Option(
        None, help="Override RAG on/off (defaults to AI_TRADER_RAG_ENABLED)"
    ),
    output: Path | None = typer.Option(None, "--output", "-o", help="Write JSON to file (UTF-8, no BOM) instead of stdout"),
) -> None:
    """Run the Final Reasoner and print a TradePlan JSON."""

    _configure_logging()
    settings = get_settings()
    if rag is not None:
        settings = settings.model_copy(update={"rag_enabled": rag})

    bundle = SignalBundle.model_validate_json(bundle_file.read_text(encoding="utf-8"))
    narrative = (
        NarrativeIntelligence.model_validate_json(narrative_file.read_text(encoding="utf-8"))
        if narrative_file is not None
        else None
    )

    ticker_value = ticker or bundle.ticker
    as_of_date = date.fromisoformat(as_of) if as_of is not None else bundle.as_of
    plan = FinalReasoner(settings=settings).reason(
        ticker=ticker_value,
        as_of=as_of_date,
        bundle=bundle,
        narrative=narrative,
        position_context=position_context,
    )
    json_str = plan.model_dump_json(indent=2)
    if output is not None:
        output.write_text(json_str, encoding="utf-8")
    else:
        print(json_str)


@app.command("trade")
def trade(
    plan_file: Path = typer.Option(..., exists=True, readable=True, help="TradePlan JSON (output from `reason --output`)"),
    shares: float = typer.Option(..., min=0.01, help="Number of shares to order"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print order details without submitting to IBKR"),
) -> None:
    """Submit a market order to IBKR paper/live account based on a TradePlan."""

    _configure_logging()
    settings = get_settings()
    plan = TradePlan.model_validate_json(plan_file.read_text(encoding="utf-8"))

    if plan.direction is SignalDirection.NEUTRAL:
        rprint({"status": "skipped", "reason": "direction is neutral — no order placed"})
        raise typer.Exit(0)

    side = OrderSide.BUY if plan.direction is SignalDirection.LONG else OrderSide.SELL
    order = BrokerOrder(ticker=plan.ticker, side=side, quantity=shares)

    rprint({
        "mode": settings.trading_mode,
        "account": settings.ibkr_account,
        "order": order.model_dump(),
        "conviction": plan.conviction,
        "holding_period_days": plan.holding_period_days,
        "exit_trigger": plan.exit_trigger,
    })

    if dry_run:
        rprint({"dry_run": True, "status": "no order submitted"})
        raise typer.Exit(0)

    with IBKRBroker.from_settings(settings) as broker:
        result = broker.place_order(order)
    rprint(result.model_dump_json(indent=2))


@app.command("review-nightly")
def review_nightly(
    outcomes_file: Path = typer.Option(..., exists=True, readable=True, help="TradeOutcome JSONL file"),
) -> None:
    """Run post-trade reviews and submit approved proposals."""

    _configure_logging()
    summary = NightlyReviewScheduler().run(outcomes_file)
    rprint(summary)


@train_app.command("local")
def train_local(
    examples_file: Path = typer.Option(
        ..., exists=True, readable=True, help="LocalTrainingExample JSONL file"
    ),
    model_out: Path | None = typer.Option(None, help="Output calibrator JSON path"),
) -> None:
    """Train a local conviction/size calibrator from your historical trade examples."""

    _configure_logging()
    settings = get_settings()
    examples = load_training_examples(examples_file)
    model = LocalCalibratorTrainer().train(examples)
    output_path = model_out or settings.local_calibrator_path
    model.save(output_path)
    rprint(
        {
            "status": "trained",
            "examples": model.training_count,
            "model_out": str(output_path),
            "metrics": model.metrics,
        }
    )


@app.command("bridge-serve")
def bridge_serve() -> None:
    """Run the shared-memory bridge server."""

    _configure_logging()
    BridgeServer().serve_forever()


@backtest_app.command("run")
def backtest_run(
    tickers: Annotated[list[str], typer.Option(help="Ticker symbols")],
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
    out: Path | None = typer.Option(None, help="Output result JSON path"),
    events_file: Path | None = typer.Option(
        None, exists=True, readable=True, help="Optional smart-money replay events JSONL"
    ),
    train_window_days: int = typer.Option(252),
    test_window_days: int = typer.Option(63),
    step_days: int = typer.Option(21),
    anchored: bool = typer.Option(False),
) -> None:
    """Run walk-forward validation."""

    _configure_logging()
    config = WalkForwardConfig(
        train_window_days=train_window_days,
        test_window_days=test_window_days,
        step_days=step_days,
        anchored=anchored,
        events_file=events_file,
    )
    result = WalkForwardEngine().run(
        list(tickers),
        date.fromisoformat(start),
        date.fromisoformat(end),
        config,
    )
    payload = result.model_dump_json(indent=2)
    if out is not None:
        out.write_text(payload, encoding="utf-8")
    rprint(payload)


@backtest_app.command("monte-carlo")
def backtest_monte_carlo(
    result_file: Path = typer.Option(..., exists=True, readable=True, help="WalkForwardResult JSON"),
    n_sims: int = typer.Option(10_000, help="Number of simulations"),
) -> None:
    """Run stress Monte Carlo over backtest trade PnL."""

    _configure_logging()
    result = WalkForwardResult.model_validate_json(result_file.read_text(encoding="utf-8"))
    pnl = [trade.pnl_pct for trade in result.trades]
    if not pnl:
        pnl = [window.sharpe / 100 for window in result.windows]
    mc = StressMonteCarlo(n_simulations=n_sims).run_stress(pnl_series=pnl)  # type: ignore[arg-type]
    rprint(json.dumps(mc.__dict__, indent=2))


@build_loop_app.command("run")
def build_loop_run(
    tickers: Annotated[list[str], typer.Option(help="Ticker symbols")] = ["AAPL", "MSFT"],
    start: str = typer.Option("2022-01-01", help="Backtest start date"),
    end: str = typer.Option("2024-12-31", help="Backtest end date"),
    max_proposals: int = typer.Option(2, help="Max proposals per run"),
) -> None:
    """Run one proposal-generation/build-loop cycle."""

    _configure_logging()
    report = BuildLoop().run_once(
        list(tickers),
        date.fromisoformat(start),
        date.fromisoformat(end),
        max_proposals=max_proposals,
    )
    rprint(report.model_dump_json(indent=2))


@app.command("autopilot")
def autopilot(
    bundle_file: Path = typer.Option(..., exists=True, readable=True, help="SignalBundle JSON file to reuse each cycle"),
    shares: float = typer.Option(..., min=0.01, help="Shares per order"),
    min_conviction: float = typer.Option(0.5, min=0.0, max=1.0, help="Minimum conviction to place an order"),
    interval_s: int = typer.Option(300, min=10, help="Seconds between cycles"),
    narrative_file: Path | None = typer.Option(None, exists=True, readable=True, help="Optional NarrativeIntelligence JSON"),
    max_cycles: int | None = typer.Option(None, help="Stop after N cycles (default: run forever, Ctrl-C to stop)"),
) -> None:
    """Continuously reason and trade: load bundle → reason → place order → sleep."""

    _configure_logging()
    settings = get_settings()
    logger = logging.getLogger("autopilot")

    stop = False
    def _handle_sigint(sig, frame):
        nonlocal stop
        stop = True
        logger.info("Autopilot stopping after current cycle …")
    signal.signal(signal.SIGINT, _handle_sigint)

    bundle = SignalBundle.model_validate_json(bundle_file.read_text(encoding="utf-8"))
    narrative = (
        NarrativeIntelligence.model_validate_json(narrative_file.read_text(encoding="utf-8"))
        if narrative_file is not None
        else None
    )

    rprint(
        f"[bold green]Autopilot started[/bold green] | "
        f"ticker={bundle.ticker} shares={shares} min_conviction={min_conviction} "
        f"interval={interval_s}s mode={settings.trading_mode} account={settings.ibkr_account}"
    )

    cycle = 0
    while not stop:
        cycle += 1
        if max_cycles is not None and cycle > max_cycles:
            logger.info("Reached max_cycles=%d, stopping.", max_cycles)
            break

        logger.info("--- Autopilot cycle %d ---", cycle)
        try:
            plan = FinalReasoner(settings=settings).reason(
                ticker=bundle.ticker,
                as_of=date.today(),
                bundle=bundle,
                narrative=narrative,
            )
            logger.info(
                "Plan: direction=%s conviction=%.2f size_mult=%.2f",
                plan.direction,
                plan.conviction,
                plan.size_multiplier,
            )

            if plan.direction is SignalDirection.NEUTRAL:
                logger.info("Direction is NEUTRAL — skipping order this cycle.")
            elif plan.conviction < min_conviction:
                logger.info(
                    "Conviction %.2f below threshold %.2f — skipping order this cycle.",
                    plan.conviction,
                    min_conviction,
                )
            else:
                side = OrderSide.BUY if plan.direction is SignalDirection.LONG else OrderSide.SELL
                order = BrokerOrder(ticker=plan.ticker, side=side, quantity=shares)
                logger.info("Submitting %s %s x%.2f …", side.value, plan.ticker, shares)
                with IBKRBroker.from_settings(settings) as broker:
                    result = broker.place_order(order)
                rprint(f"[bold]Cycle {cycle}[/bold] order result: {result.model_dump_json()}")

        except Exception as exc:
            logger.error("Cycle %d failed: %s", cycle, exc)

        if not stop and (max_cycles is None or cycle < max_cycles):
            logger.info("Sleeping %ds until next cycle …", interval_s)
            # Sleep in 1-second chunks so Ctrl-C is responsive
            for _ in range(interval_s):
                if stop:
                    break
                time.sleep(1)

    rprint("[bold yellow]Autopilot stopped.[/bold yellow]")


if __name__ == "__main__":
    app()
