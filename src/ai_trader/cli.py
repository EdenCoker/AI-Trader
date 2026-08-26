from __future__ import annotations

import json
import logging
import re
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
from ai_trader.broker.contracts import BrokerAccountSnapshot, BrokerOrder, BrokerQuote, OrderSide
from ai_trader.broker.ibkr import IBKRBroker
from ai_trader.broker.sizing import (
    BalanceSizingConfig,
    BalanceSizingResult,
    size_order_from_balance,
)
from ai_trader.build_loop.loop import BuildLoop
from ai_trader.config import get_settings
from ai_trader.domain.signals import SignalBundle, SignalDirection
from ai_trader.evolution.promoter import ModelPromoter
from ai_trader.gui import run_gui
from ai_trader.intelligence.models import NarrativeIntelligence
from ai_trader.intelligence.narrative import NarrativeAnalyzer
from ai_trader.intelligence.reasoner import FinalReasoner
from ai_trader.intelligence.trade_plan import TradePlan
from ai_trader.news import (
    NewsIntelligenceEngine,
    build_news_signals,
)
from ai_trader.providers.fear_greed import LiveFearGreedProvider
from ai_trader.rag.trader_rag import format_retrieved, get_trader_rag
from ai_trader.self_improvement.scheduler import NightlyReviewScheduler
from ai_trader.training import (
    ConvictionMetric,
    LocalCalibratorTrainer,
    StrategyBacktestConfig,
    filter_examples_by_horizon,
    load_training_examples,
    run_strategy_backtest,
)

app = typer.Typer(add_completion=False, no_args_is_help=True)
backtest_app = typer.Typer(no_args_is_help=True)
build_loop_app = typer.Typer(no_args_is_help=True)
train_app = typer.Typer(no_args_is_help=True)
model_app = typer.Typer(no_args_is_help=True)
news_app = typer.Typer(no_args_is_help=True)
app.add_typer(backtest_app, name="backtest")
app.add_typer(build_loop_app, name="build-loop")
app.add_typer(train_app, name="train")
app.add_typer(model_app, name="model")
app.add_typer(news_app, name="news")


class _SecretRedactionFilter(logging.Filter):
    _patterns = (
        re.compile(r"apiKey=[^&\s\"']+"),
        re.compile(r"(OPENAI_API_KEY|POLYGON_API_KEY|QUIVER_API_KEY)=\S+"),
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for pattern in self._patterns:
            message = pattern.sub(
                lambda match: match.group(0).split("=")[0] + "=REDACTED",
                message,
            )
        record.msg = message
        record.args = ()
        return True


def _configure_logging() -> None:
    settings = get_settings()
    Path("logs").mkdir(exist_ok=True)
    redaction_filter = _SecretRedactionFilter()
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler("logs/ai_trader.log"),
    ]
    for handler in handlers:
        handler.addFilter(redaction_filter)
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


@app.command()
def status() -> None:
    """Show runtime config status (secrets redacted)."""

    _configure_logging()
    settings = get_settings()
    rprint(settings.redacted())
    rprint(settings.provider_status())


@app.command("fear-greed")
def fear_greed(
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write the current snapshot JSON to this file.",
    ),
    no_append: bool = typer.Option(
        False,
        "--no-append",
        help="Do not append the snapshot to AI_TRADER_FEAR_GREED_SNAPSHOT_PATH.",
    ),
) -> None:
    """Fetch the live composite fear/greed snapshot."""

    _configure_logging()
    settings = get_settings()
    provider = LiveFearGreedProvider(settings=settings)
    snapshot = provider.fetch_snapshot()
    if not no_append:
        provider.append_snapshot(snapshot)

    json_str = snapshot.model_dump_json(indent=2)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json_str, encoding="utf-8")
    else:
        print(json_str)


@app.command("gui")
def gui(
    host: str = typer.Option("127.0.0.1", help="Host interface for the local GUI"),
    port: int = typer.Option(8787, help="Port for the local GUI"),
    open_browser: bool = typer.Option(True, help="Open the GUI in the default browser"),
) -> None:
    """Launch the local browser GUI."""

    run_gui(host=host, port=port, open_browser=open_browser)


def _parse_as_of_utc(as_of: str | None):
    """Parse --as-of. A timestamp without an offset means UTC — matching
    the archive's storage convention — NOT machine-local time, so replays
    are reproducible across machines."""

    from datetime import UTC, datetime

    if not as_of:
        return datetime.now(UTC)
    parsed = datetime.fromisoformat(as_of)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


@news_app.command("pull")
def news_pull(
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write the acquisition report JSON to this file."
    ),
) -> None:
    """Run one news acquisition pass (worldmonitor digest + direct RSS) and archive sightings."""

    _configure_logging()
    engine = NewsIntelligenceEngine()
    report = engine.collect()
    json_str = report.model_dump_json(indent=2)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json_str, encoding="utf-8")
    else:
        print(json_str)


@news_app.command("stories")
def news_stories(
    limit: int = typer.Option(15, help="Maximum stories to print"),
    ticker: str | None = typer.Option(None, help="Only stories naming this ticker"),
    as_of: str | None = typer.Option(
        None, help="Replay the archive as of this ISO timestamp (default: now)"
    ),
) -> None:
    """Cluster and score archived news into ranked stories."""

    _configure_logging()
    engine = NewsIntelligenceEngine()
    as_of_dt = _parse_as_of_utc(as_of)
    stories = engine.stories(as_of_dt)
    if ticker:
        wanted = ticker.upper()
        stories = [story for story in stories if wanted in story.tickers]
    for story in stories[:limit]:
        rprint(
            f"[{story.importance_score:>3}] {story.phase.value:<10} "
            f"{story.classification.category:<18} "
            f"pubs={story.publisher_count} cred={story.credibility_score:<3} "
            f"{story.canonical_title[:90]}"
        )
    if not stories:
        rprint("No stories in the archive window. Run `ai-trader news pull` first.")


@news_app.command("signal")
def news_signal(
    ticker: str = typer.Option(..., help="Ticker symbol"),
    as_of: str | None = typer.Option(
        None, help="As-of ISO timestamp for no-lookahead replay (default: now)"
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write signals JSON to this file"
    ),
) -> None:
    """Build news Signals for a ticker from the archived story window."""

    _configure_logging()
    engine = NewsIntelligenceEngine()
    as_of_dt = _parse_as_of_utc(as_of)
    stories = engine.stories(as_of_dt)
    signals = build_news_signals(stories, ticker, as_of_dt)
    payload = json.dumps(
        [json.loads(signal.model_dump_json()) for signal in signals], indent=2
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    else:
        print(payload)


@app.command("analyze-news")
def analyze_news(
    ticker: str = typer.Option(..., help="Ticker symbol"),
    headline: str = typer.Option(..., help="Headline text"),
    body: str | None = typer.Option(None, help="Body text"),
    body_file: Path | None = typer.Option(None, exists=True, readable=True, help="Body file path"),
    as_of: str | None = typer.Option(None, help="As-of date (YYYY-MM-DD). Defaults to today."),
    market_context: str | None = typer.Option(None, help="Optional market context text"),
    analyst_context: str | None = typer.Option(None, help="Optional analyst expectations text"),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write JSON to file (UTF-8, no BOM) instead of stdout"
    ),
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
    bundle_file: Path = typer.Option(
        ..., exists=True, readable=True, help="SignalBundle JSON file"
    ),
    narrative_file: Path | None = typer.Option(
        None, exists=True, readable=True, help="Optional NarrativeIntelligence JSON file"
    ),
    ticker: str | None = typer.Option(None, help="Override ticker (defaults to bundle ticker)"),
    as_of: str | None = typer.Option(None, help="Override as-of date (YYYY-MM-DD)"),
    position_context: str | None = typer.Option(
        None, help="Optional current position context text"
    ),
    rag: bool | None = typer.Option(
        None, help="Override RAG on/off (defaults to AI_TRADER_RAG_ENABLED)"
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write JSON to file (UTF-8, no BOM) instead of stdout"
    ),
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
    plan_file: Path = typer.Option(
        ..., exists=True, readable=True, help="TradePlan JSON (output from `reason --output`)"
    ),
    shares: float | None = typer.Option(
        None,
        min=0.01,
        help="Fixed share override. Omit to size from available balance.",
    ),
    cash_fraction: float = typer.Option(
        0.02,
        min=0.0001,
        max=1.0,
        help="Fraction of balance to deploy at conviction=1 and size=1.",
    ),
    starting_balance: float | None = typer.Option(
        None,
        min=0.01,
        help="Optional starting balance/risk budget cap for sizing.",
    ),
    reference_price: float | None = typer.Option(
        None,
        min=0.01,
        help="Optional price override for balance-based sizing.",
    ),
    fractional_shares: bool = typer.Option(
        False,
        "--fractional-shares",
        help="Allow fractional calculated share quantities.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print order details without submitting to IBKR"
    ),
) -> None:
    """Submit a market order to IBKR paper/live account based on a TradePlan."""

    _configure_logging()
    settings = get_settings()
    plan = TradePlan.model_validate_json(plan_file.read_text(encoding="utf-8"))

    if plan.direction is SignalDirection.NEUTRAL:
        rprint({"status": "skipped", "reason": "direction is neutral — no order placed"})
        raise typer.Exit(0)

    side = OrderSide.BUY if plan.direction is SignalDirection.LONG else OrderSide.SELL

    can_size_without_broker = shares is not None or (
        starting_balance is not None and reference_price is not None
    )
    if dry_run and can_size_without_broker:
        order, sizing = _build_order_or_skip(
            plan=plan,
            side=side,
            shares=shares,
            cash_fraction=cash_fraction,
            starting_balance=starting_balance,
            reference_price=reference_price,
            fractional_shares=fractional_shares,
        )
        _print_order_preview(settings, plan, order, sizing=sizing)
        rprint({"dry_run": True, "status": "no order submitted"})
        raise typer.Exit(0)

    with IBKRBroker.from_settings(settings) as broker:
        order, sizing = _build_order_or_skip(
            plan=plan,
            side=side,
            shares=shares,
            cash_fraction=cash_fraction,
            starting_balance=starting_balance,
            reference_price=reference_price,
            fractional_shares=fractional_shares,
            broker=broker,
        )
        _print_order_preview(settings, plan, order, sizing=sizing)

        if dry_run:
            rprint({"dry_run": True, "status": "no order submitted"})
            raise typer.Exit(0)

        result = broker.place_order(order)
    rprint(result.model_dump_json(indent=2))


def _build_order_or_skip(**kwargs) -> tuple[BrokerOrder, BalanceSizingResult | None]:
    try:
        return _build_order_from_plan(**kwargs)
    except ValueError as exc:
        rprint({"status": "skipped", "reason": str(exc)})
        raise typer.Exit(0) from exc


def _build_order_from_plan(
    *,
    plan: TradePlan,
    side: OrderSide,
    shares: float | None,
    cash_fraction: float,
    starting_balance: float | None,
    reference_price: float | None,
    fractional_shares: bool,
    broker: IBKRBroker | None = None,
) -> tuple[BrokerOrder, BalanceSizingResult | None]:
    if shares is not None:
        return BrokerOrder(ticker=plan.ticker, side=side, quantity=shares), None

    if broker is None and (starting_balance is None or reference_price is None):
        raise typer.BadParameter(
            "balance-based dry runs without IBKR require both --starting-balance and --reference-price"
        )

    account = _sizing_account(broker, starting_balance)
    quote = (
        BrokerQuote(ticker=plan.ticker, price=reference_price, source="override")
        if reference_price is not None
        else broker.market_price(plan.ticker)
    )
    sizing = size_order_from_balance(
        plan=plan,
        side=side,
        account=account,
        quote=quote,
        config=BalanceSizingConfig(
            cash_fraction=cash_fraction,
            allow_fractional_shares=fractional_shares,
        ),
    )
    return BrokerOrder(ticker=plan.ticker, side=side, quantity=sizing.quantity), sizing


def _sizing_account(
    broker: IBKRBroker | None,
    starting_balance: float | None,
) -> BrokerAccountSnapshot:
    if broker is None:
        return BrokerAccountSnapshot(available_funds=starting_balance or 0.0)

    snapshot = broker.account_snapshot()
    if starting_balance is None:
        return snapshot

    broker_balance = snapshot.spendable_balance
    sizing_balance = (
        min(broker_balance, starting_balance) if broker_balance > 0 else starting_balance
    )
    return BrokerAccountSnapshot(
        account=snapshot.account,
        currency=snapshot.currency,
        available_funds=sizing_balance,
        net_liquidation=snapshot.net_liquidation,
    )


def _print_order_preview(
    settings,
    plan: TradePlan,
    order: BrokerOrder,
    *,
    sizing: BalanceSizingResult | None,
) -> None:
    payload = {
        "mode": settings.trading_mode,
        "account": settings.ibkr_account,
        "order": order.model_dump(mode="json"),
        "conviction": plan.conviction,
        "size_multiplier": plan.size_multiplier,
        "holding_period_days": plan.holding_period_days,
        "exit_trigger": plan.exit_trigger,
    }
    if sizing is not None:
        payload["sizing"] = sizing.model_dump(mode="json")
    rprint(payload)


@app.command("review-nightly")
def review_nightly(
    outcomes_file: Path = typer.Option(
        ..., exists=True, readable=True, help="TradeOutcome JSONL file"
    ),
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
    model_out: Path | None = typer.Option(
        None,
        "--out",
        "--model-out",
        help="Output calibrator JSON path",
    ),
    horizon: str = typer.Option(
        "all",
        "--horizon",
        help="Calibrator horizon to train: all, short, medium, or long.",
    ),
) -> None:
    """Train a local conviction/size calibrator from your historical trade examples."""

    _configure_logging()
    settings = get_settings()
    examples = load_training_examples(examples_file)
    horizon = horizon.casefold().strip()
    valid_horizons = {"all", "short", "medium", "long"}
    if horizon not in valid_horizons:
        raise typer.BadParameter("--horizon must be one of: all, short, medium, long")

    outputs = []
    if horizon == "all":
        model = LocalCalibratorTrainer().train(examples)
        output_path = model_out or settings.local_calibrator_path
        model.save(output_path)
        outputs.append(
            {
                "horizon": "all",
                "examples": model.training_count,
                "model_out": str(output_path),
                "metrics": model.metrics,
            }
        )
        horizon_paths = {
            "short": settings.local_calibrator_short_path,
            "medium": settings.local_calibrator_medium_path,
            "long": settings.local_calibrator_long_path,
        }
        for horizon_name, output_path in horizon_paths.items():
            subset = filter_examples_by_horizon(examples, horizon_name)  # type: ignore[arg-type]
            if not subset:
                outputs.append({"horizon": horizon_name, "status": "skipped", "examples": 0})
                continue
            model = LocalCalibratorTrainer().train(subset)
            model.save(output_path)
            outputs.append(
                {
                    "horizon": horizon_name,
                    "examples": model.training_count,
                    "model_out": str(output_path),
                    "metrics": model.metrics,
                }
            )
    else:
        subset = filter_examples_by_horizon(examples, horizon)  # type: ignore[arg-type]
        if not subset:
            raise typer.BadParameter(f"No {horizon} horizon examples found in {examples_file}")
        model = LocalCalibratorTrainer().train(subset)
        default_path = {
            "short": settings.local_calibrator_short_path,
            "medium": settings.local_calibrator_medium_path,
            "long": settings.local_calibrator_long_path,
        }[horizon]
        output_path = model_out or default_path
        model.save(output_path)
        outputs.append(
            {
                "horizon": horizon,
                "examples": model.training_count,
                "model_out": str(output_path),
                "metrics": model.metrics,
            }
        )

    rprint({"status": "trained", "outputs": outputs})


@model_app.command("rollback")
def model_rollback(
    to: str = typer.Option(..., "--to", help="Version to restore, for example v0003"),
) -> None:
    """Rollback production.json to a versioned archived model."""

    _configure_logging()
    result = ModelPromoter().rollback(to_version=to)
    rprint({"status": "rolled_back", **result})


@train_app.command("backtest")
def train_backtest(
    examples_file: Path = typer.Option(
        Path("logs/training_examples.jsonl"),
        exists=True,
        readable=True,
        help="LocalTrainingExample JSONL file",
    ),
    start_date: str | None = typer.Option(
        None,
        help="Optional first example date YYYY-MM-DD",
    ),
    end_date: str | None = typer.Option(
        None,
        help="Optional last example date YYYY-MM-DD",
    ),
    split_date: str | None = typer.Option(
        None,
        help="Optional out-of-sample split date YYYY-MM-DD",
    ),
    conviction_metric: str = typer.Option(
        ConvictionMetric.PLAN.value,
        help="Conviction score to use: plan, bundle, or agreement_adjusted",
    ),
    min_trades: int = typer.Option(50, min=1, help="Minimum trades for a strategy"),
    min_active_months: int = typer.Option(3, min=1, help="Minimum active months"),
    min_active_years: int = typer.Option(
        1,
        min=1,
        help="Minimum active calendar years for a strategy",
    ),
    min_trades_per_month: int = typer.Option(
        5,
        min=1,
        help="Minimum trades in a month for that month to count in equity",
    ),
    max_ticker_concentration: float | None = typer.Option(
        None,
        min=0.0,
        max=1.0,
        help="Reject strategies above this single-ticker trade share",
    ),
    max_month_concentration: float | None = typer.Option(
        None,
        min=0.0,
        max=1.0,
        help="Reject strategies above this single-month trade share",
    ),
    max_drawdown: float | None = typer.Option(
        None,
        min=0.0,
        max=1.0,
        help="Reject strategies above this max drawdown fraction",
    ),
    require_positive_oos_score: bool = typer.Option(
        False,
        help="Reject split-date candidates with non-positive OOS score",
    ),
    max_train_test_sharpe_decay: float | None = typer.Option(
        None,
        min=0.0,
        help="Reject split-date candidates with train-test Sharpe decay above this",
    ),
    top_n: int = typer.Option(10, min=1, help="Number of strategies to report"),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write JSON report to file",
    ),
) -> None:
    """Backtest local training examples across deterministic policy rules."""

    _configure_logging()
    try:
        conviction_metric_value = ConvictionMetric(conviction_metric)
    except ValueError as exc:
        raise typer.BadParameter(
            "conviction_metric must be plan, bundle, or agreement_adjusted"
        ) from exc

    config = StrategyBacktestConfig(
        start_date=date.fromisoformat(start_date) if start_date is not None else None,
        end_date=date.fromisoformat(end_date) if end_date is not None else None,
        split_date=date.fromisoformat(split_date) if split_date is not None else None,
        conviction_metric=conviction_metric_value,
        min_trades=min_trades,
        min_active_months=min_active_months,
        min_active_years=min_active_years,
        min_trades_per_month=min_trades_per_month,
        max_ticker_concentration=max_ticker_concentration,
        max_month_concentration=max_month_concentration,
        max_drawdown=max_drawdown,
        require_positive_oos_score=require_positive_oos_score,
        max_train_test_sharpe_decay=max_train_test_sharpe_decay,
        top_n=top_n,
    )
    report = run_strategy_backtest(load_training_examples(examples_file), config)
    payload = report.model_dump_json(indent=2)
    if output is not None:
        output.write_text(payload, encoding="utf-8")
    rprint(payload)


@app.command("bridge-serve")
def bridge_serve() -> None:
    """Run the shared-memory bridge server."""

    _configure_logging()
    BridgeServer().serve_forever()


@backtest_app.command("run")
def backtest_run(
    tickers: Annotated[
        list[str] | None,
        typer.Option(help="Optional ticker override. Omit to infer from --events-file."),
    ] = None,
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
    out: Path | None = typer.Option(None, help="Output result JSON path"),
    events_file: Path | None = typer.Option(
        None,
        exists=True,
        readable=True,
        help="Optional smart-money replay events JSONL. Used for automatic ticker discovery.",
    ),
    train_window_days: int = typer.Option(252),
    test_window_days: int = typer.Option(63),
    step_days: int = typer.Option(21),
    anchored: bool = typer.Option(False),
    signal_threshold: float = typer.Option(0.10, min=0.0, max=1.0),
    max_holding_days: int = typer.Option(63, min=1),
    stop_loss_pct: float | None = typer.Option(None, min=0.0, max=1.0),
    take_profit_pct: float | None = typer.Option(None, min=0.0, max=5.0),
    starting_balance: float = typer.Option(
        10_000.0,
        min=0.01,
        help="Starting simulated account balance.",
    ),
    cash_fraction: float = typer.Option(
        0.02,
        min=0.0001,
        max=1.0,
        help="Fraction of current balance to deploy at conviction=1 and size=1.",
    ),
    fractional_shares: bool = typer.Option(
        False,
        "--fractional-shares",
        help="Allow fractional simulated share quantities.",
    ),
) -> None:
    """Run walk-forward validation."""

    _configure_logging()
    config = WalkForwardConfig(
        train_window_days=train_window_days,
        test_window_days=test_window_days,
        step_days=step_days,
        anchored=anchored,
        events_file=events_file,
        signal_threshold=signal_threshold,
        max_holding_days=max_holding_days,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        starting_balance=starting_balance,
        cash_fraction=cash_fraction,
        fractional_shares=fractional_shares,
    )
    result = WalkForwardEngine().run(
        list(tickers) if tickers is not None else None,
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
    result_file: Path = typer.Option(
        ..., exists=True, readable=True, help="WalkForwardResult JSON"
    ),
    n_sims: int = typer.Option(10_000, help="Number of simulations"),
) -> None:
    """Run stress Monte Carlo over backtest trade PnL."""

    _configure_logging()
    result = WalkForwardResult.model_validate_json(result_file.read_text(encoding="utf-8"))
    pnl = [trade.account_return if trade.notional > 0 else trade.pnl_pct for trade in result.trades]
    if not pnl:
        pnl = [window.sharpe / 100 for window in result.windows]
    mc = StressMonteCarlo(n_simulations=n_sims).run_stress(pnl_series=pnl)  # type: ignore[arg-type]
    rprint(json.dumps(mc.__dict__, indent=2))


@build_loop_app.command("run")
def build_loop_run(
    tickers: Annotated[
        list[str] | None,
        typer.Option(help="Optional ticker override. Omit to infer from replay events."),
    ] = None,
    start: str = typer.Option("2022-01-01", help="Backtest start date"),
    end: str = typer.Option("2024-12-31", help="Backtest end date"),
    events_file: Path | None = typer.Option(
        Path("examples/sample_events.jsonl"),
        help="Replay events JSONL used for automatic ticker discovery",
    ),
    max_proposals: int = typer.Option(2, help="Max proposals per run"),
) -> None:
    """Run one proposal-generation/build-loop cycle."""

    _configure_logging()
    report = BuildLoop().run_once(
        list(tickers) if tickers is not None else None,
        date.fromisoformat(start),
        date.fromisoformat(end),
        max_proposals=max_proposals,
        events_file=events_file,
    )
    rprint(report.model_dump_json(indent=2))


@app.command("autopilot")
def autopilot(
    bundle_file: Path = typer.Option(
        ..., exists=True, readable=True, help="SignalBundle JSON file to reuse each cycle"
    ),
    shares: float | None = typer.Option(
        None,
        min=0.01,
        help="Fixed shares per order. Omit to size from available balance.",
    ),
    cash_fraction: float = typer.Option(
        0.02,
        min=0.0001,
        max=1.0,
        help="Fraction of balance to deploy at conviction=1 and size=1.",
    ),
    starting_balance: float | None = typer.Option(
        None,
        min=0.01,
        help="Optional starting balance/risk budget cap for sizing.",
    ),
    fractional_shares: bool = typer.Option(
        False,
        "--fractional-shares",
        help="Allow fractional calculated share quantities.",
    ),
    min_conviction: float = typer.Option(
        0.5, min=0.0, max=1.0, help="Minimum conviction to place an order"
    ),
    interval_s: int = typer.Option(300, min=10, help="Seconds between cycles"),
    narrative_file: Path | None = typer.Option(
        None, exists=True, readable=True, help="Optional NarrativeIntelligence JSON"
    ),
    max_cycles: int | None = typer.Option(
        None, help="Stop after N cycles (default: run forever, Ctrl-C to stop)"
    ),
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
    fixed_shares = shares

    sizing_label = (
        f"fixed_shares={shares}"
        if shares is not None
        else f"cash_fraction={cash_fraction} starting_balance={starting_balance or 'broker'}"
    )
    rprint(
        f"[bold green]Autopilot started[/bold green] | "
        f"ticker={bundle.ticker} {sizing_label} min_conviction={min_conviction} "
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
                with IBKRBroker.from_settings(settings) as broker:
                    order, sizing = _build_order_from_plan(
                        plan=plan,
                        side=side,
                        shares=fixed_shares,
                        cash_fraction=cash_fraction,
                        starting_balance=starting_balance,
                        reference_price=None,
                        fractional_shares=fractional_shares,
                        broker=broker,
                    )
                    logger.info("Submitting %s %s x%.4f", side.value, plan.ticker, order.quantity)
                    result = broker.place_order(order)
                rprint(
                    {
                        "cycle": cycle,
                        "order": order.model_dump(mode="json"),
                        "sizing": sizing.model_dump(mode="json") if sizing is not None else None,
                        "result": result.model_dump(mode="json"),
                    }
                )

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
