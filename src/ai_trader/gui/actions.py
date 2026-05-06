from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FieldSpec:
    name: str
    label: str
    field_type: str = "text"
    required: bool = False
    default: str | int | float | bool | None = None
    placeholder: str = ""
    min_value: float | None = None
    max_value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "type": self.field_type,
            "required": self.required,
            "default": self.default,
            "placeholder": self.placeholder,
            "min": self.min_value,
            "max": self.max_value,
        }


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    label: str
    group: str
    command: tuple[str, ...]
    fields: tuple[FieldSpec, ...] = ()
    timeout_s: int = 300
    background: bool = False
    streaming: bool = False  # stream stdout live via SSE instead of waiting for completion

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.action_id,
            "label": self.label,
            "group": self.group,
            "fields": [field.to_dict() for field in self.fields],
            "timeout_s": self.timeout_s,
            "background": self.background,
            "streaming": self.streaming,
        }


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    background_pid: int | None = None
    log_file: str | None = None

    @property
    def success(self) -> bool:
        return self.returncode == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": list(self.command),
            "returncode": self.returncode,
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_s": self.duration_s,
            "background_pid": self.background_pid,
            "log_file": self.log_file,
        }


ACTIONS: dict[str, ActionSpec] = {
    "status": ActionSpec(
        action_id="status",
        label="Status",
        group="Core",
        command=("status",),
        timeout_s=60,
    ),
    "analyze_news": ActionSpec(
        action_id="analyze_news",
        label="Analyze News",
        group="Intelligence",
        command=("analyze-news",),
        fields=(
            FieldSpec("ticker", "Ticker", required=True, default="MSFT"),
            FieldSpec("headline", "Headline", required=True),
            FieldSpec("body", "Body", field_type="textarea", placeholder="Paste article text"),
            FieldSpec("body_file", "Body File", placeholder="examples/sample_news.txt"),
            FieldSpec("as_of", "As Of", placeholder="YYYY-MM-DD"),
            FieldSpec("market_context", "Market Context", field_type="textarea"),
            FieldSpec("analyst_context", "Analyst Context", field_type="textarea"),
            FieldSpec("output", "Output File", placeholder="logs/narrative_output.json"),
        ),
        timeout_s=600,
    ),
    "reason": ActionSpec(
        action_id="reason",
        label="Final Reasoner",
        group="Intelligence",
        command=("reason",),
        fields=(
            FieldSpec(
                "bundle_file",
                "Signal Bundle",
                required=True,
                default="examples/sample_signal_bundle.json",
            ),
            FieldSpec("narrative_file", "Narrative File", placeholder="logs/narrative_output.json"),
            FieldSpec("as_of", "As Of", placeholder="YYYY-MM-DD"),
            FieldSpec("position_context", "Position Context", field_type="textarea"),
            FieldSpec("rag", "Use RAG", field_type="checkbox"),
            FieldSpec("output", "Output File", placeholder="logs/trade_plan.json"),
        ),
        timeout_s=600,
    ),
    "train_local": ActionSpec(
        action_id="train_local",
        label="Train Local",
        group="Training",
        command=("train", "local"),
        fields=(
            FieldSpec(
                "examples_file",
                "Examples JSONL",
                required=True,
                default="examples/sample_training_examples.jsonl",
            ),
            FieldSpec("model_out", "Model Output", default="data/models/local_calibrator.json"),
        ),
        timeout_s=300,
    ),
    "train_backtest": ActionSpec(
        action_id="train_backtest",
        label="Training Backtest",
        group="Training",
        command=("train", "backtest"),
        fields=(
            FieldSpec(
                "examples_file",
                "Examples JSONL",
                required=True,
                default="logs/training_examples.jsonl",
            ),
            FieldSpec("start_date", "Start Date", placeholder="YYYY-MM-DD"),
            FieldSpec("end_date", "End Date", placeholder="YYYY-MM-DD"),
            FieldSpec("split_date", "Split Date", placeholder="YYYY-MM-DD"),
            FieldSpec("min_trades", "Min Trades", field_type="number", default=50, min_value=1),
            FieldSpec(
                "min_active_months",
                "Min Months",
                field_type="number",
                default=3,
                min_value=1,
            ),
            FieldSpec(
                "min_trades_per_month",
                "Trades/Month",
                field_type="number",
                default=5,
                min_value=1,
            ),
            FieldSpec("top_n", "Top N", field_type="number", default=10, min_value=1),
            FieldSpec("output", "Output File", default="logs/training_backtest.json"),
        ),
        timeout_s=900,
    ),
    "rag_index": ActionSpec(
        action_id="rag_index",
        label="Build RAG Index",
        group="RAG",
        command=("rag-index",),
        fields=(FieldSpec("rebuild", "Rebuild", field_type="checkbox"),),
        timeout_s=900,
    ),
    "rag_query": ActionSpec(
        action_id="rag_query",
        label="Query RAG",
        group="RAG",
        command=("rag-query",),
        fields=(
            FieldSpec("query", "Query", required=True),
            FieldSpec("k", "Results", field_type="number", default=3, min_value=1, max_value=10),
        ),
        timeout_s=120,
    ),
    "ibkr_positions": ActionSpec(
        action_id="ibkr_positions",
        label="IBKR Positions",
        group="Broker",
        command=("ibkr-positions",),
        timeout_s=120,
    ),
    "trade": ActionSpec(
        action_id="trade",
        label="Trade Plan",
        group="Broker",
        command=("trade",),
        fields=(
            FieldSpec(
                "plan_file",
                "Trade Plan File",
                required=True,
                default="logs/trade_plan.json",
            ),
            FieldSpec(
                "shares",
                "Shares",
                field_type="number",
                placeholder="auto",
                min_value=0.01,
            ),
            FieldSpec("cash_fraction", "Cash Fraction", field_type="number", default=0.02),
            FieldSpec("starting_balance", "Starting Balance", field_type="number"),
            FieldSpec("reference_price", "Reference Price", field_type="number"),
            FieldSpec("fractional_shares", "Fractional Shares", field_type="checkbox"),
            FieldSpec("dry_run", "Dry Run", field_type="checkbox", default=True),
        ),
        timeout_s=120,
    ),
    "backtest_run": ActionSpec(
        action_id="backtest_run",
        label="Backtest",
        group="Backtesting",
        command=("backtest", "run"),
        fields=(
            FieldSpec("start", "Start", required=True, default="2022-01-01"),
            FieldSpec("end", "End", required=True, default="2024-12-31"),
            FieldSpec("events_file", "Events File", default="examples/sample_events.jsonl"),
            FieldSpec("out", "Output File", default="logs/backtest_result.json"),
            FieldSpec("train_window_days", "Train Days", field_type="number", default=252),
            FieldSpec("test_window_days", "Test Days", field_type="number", default=63),
            FieldSpec("step_days", "Step Days", field_type="number", default=21),
            FieldSpec("signal_threshold", "Signal Threshold", field_type="number", default=0.1),
            FieldSpec("max_holding_days", "Max Holding Days", field_type="number", default=63),
            FieldSpec("stop_loss_pct", "Stop Loss", field_type="number", placeholder="0.08"),
            FieldSpec("take_profit_pct", "Take Profit", field_type="number", placeholder="0.20"),
            FieldSpec("starting_balance", "Starting Balance", field_type="number", default=10000),
            FieldSpec("cash_fraction", "Cash Fraction", field_type="number", default=0.02),
            FieldSpec("fractional_shares", "Fractional Shares", field_type="checkbox"),
            FieldSpec("anchored", "Anchored", field_type="checkbox"),
        ),
        timeout_s=1800,
    ),
    "monte_carlo": ActionSpec(
        action_id="monte_carlo",
        label="Monte Carlo",
        group="Backtesting",
        command=("backtest", "monte-carlo"),
        fields=(
            FieldSpec(
                "result_file",
                "Backtest Result",
                required=True,
                default="logs/backtest_result.json",
            ),
            FieldSpec("n_sims", "Simulations", field_type="number", default=10000, min_value=1),
        ),
        timeout_s=600,
    ),
    "review_nightly": ActionSpec(
        action_id="review_nightly",
        label="Nightly Review",
        group="Automation",
        command=("review-nightly",),
        fields=(
            FieldSpec(
                "outcomes_file",
                "Outcomes JSONL",
                required=True,
                default="outcomes.jsonl",
            ),
        ),
        timeout_s=900,
    ),
    "build_loop": ActionSpec(
        action_id="build_loop",
        label="Build Loop",
        group="Automation",
        command=("build-loop", "run"),
        fields=(
            FieldSpec("start", "Start", required=True, default="2022-01-01"),
            FieldSpec("end", "End", required=True, default="2024-12-31"),
            FieldSpec("events_file", "Events File", default="examples/sample_events.jsonl"),
            FieldSpec(
                "max_proposals",
                "Max Proposals",
                field_type="number",
                default=2,
                min_value=1,
            ),
        ),
        timeout_s=3600,
    ),
    "autopilot": ActionSpec(
        action_id="autopilot",
        label="Autopilot Cycle",
        group="Automation",
        command=("autopilot",),
        fields=(
            FieldSpec(
                "bundle_file",
                "Signal Bundle",
                required=True,
                default="examples/sample_signal_bundle.json",
            ),
            FieldSpec(
                "shares",
                "Shares",
                field_type="number",
                placeholder="auto",
                min_value=0.01,
            ),
            FieldSpec("cash_fraction", "Cash Fraction", field_type="number", default=0.02),
            FieldSpec("starting_balance", "Starting Balance", field_type="number"),
            FieldSpec("fractional_shares", "Fractional Shares", field_type="checkbox"),
            FieldSpec("min_conviction", "Min Conviction", field_type="number", default=0.5),
            FieldSpec(
                "interval_s",
                "Interval Seconds",
                field_type="number",
                default=300,
                min_value=10,
            ),
            FieldSpec("narrative_file", "Narrative File", placeholder="logs/narrative_output.json"),
            FieldSpec("max_cycles", "Max Cycles", field_type="number", default=1, min_value=1),
        ),
        timeout_s=900,
    ),
    "ingest": ActionSpec(
        action_id="ingest",
        label="Ingest Training Data",
        group="Training",
        command=("ingest",),
        fields=(
            FieldSpec("out", "Output JSONL", default="logs/training_examples.jsonl"),
            FieldSpec("min_date", "Min Date", default="2018-01-01"),
            FieldSpec("max_date", "Max Date", placeholder="today"),
            FieldSpec("tickers", "Tickers (space-separated)", placeholder="MSFT AAPL (blank = all)"),
        ),
        timeout_s=7200,
        streaming=True,
    ),
    "bridge_serve": ActionSpec(
        action_id="bridge_serve",
        label="Bridge Server",
        group="Bridge",
        command=("bridge-serve",),
        timeout_s=0,
        background=True,
    ),
}


def action_specs() -> list[dict[str, Any]]:
    return [spec.to_dict() for spec in ACTIONS.values()]


def build_command(action_id: str, inputs: dict[str, Any]) -> tuple[str, ...]:
    try:
        spec = ACTIONS[action_id]
    except KeyError as exc:
        raise ValueError(f"Unknown GUI action: {action_id}") from exc

    # Ingest calls the script directly (not a CLI subcommand)
    if action_id == "ingest":
        script = str(Path(__file__).resolve().parent.parent.parent.parent / "scripts" / "ingest_training_data.py")
        args: list[str] = [sys.executable, script]
        out_val = inputs.get("out", "logs/training_examples.jsonl")
        if out_val:
            args += ["--out", str(out_val)]
        min_date = inputs.get("min_date", "")
        if min_date:
            args += ["--min-date", str(min_date)]
        max_date = inputs.get("max_date", "")
        if max_date:
            args += ["--max-date", str(max_date)]
        tickers = str(inputs.get("tickers", "")).strip()
        if tickers:
            args += ["--tickers"] + tickers.split()
        return tuple(args)

    args = [sys.executable, "-m", "ai_trader.cli", *spec.command]
    for field in spec.fields:
        value = inputs.get(field.name, field.default)
        if field.field_type == "checkbox":
            _append_checkbox(args, field.name, bool(value))
            continue
        if value is None or str(value).strip() == "":
            if field.required:
                raise ValueError(f"{field.label} is required")
            continue
        _append_field(args, field.name, value)
    return tuple(args)


def run_action(action_id: str, inputs: dict[str, Any], *, cwd: Path | None = None) -> CommandResult:
    spec = ACTIONS.get(action_id)
    if spec is None:
        raise ValueError(f"Unknown GUI action: {action_id}")

    command = build_command(action_id, inputs)
    cwd = cwd or Path.cwd()
    started = datetime.now()
    if spec.background:
        log_dir = cwd / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"gui_{action_id}_{started.strftime('%Y%m%d_%H%M%S')}.log"
        handle = log_file.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        handle.close()
        return CommandResult(
            command=command,
            returncode=0,
            stdout=f"Started background process {process.pid}.",
            stderr="",
            duration_s=0.0,
            background_pid=process.pid,
            log_file=str(log_file),
        )

    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=spec.timeout_s,
        )
        elapsed = (datetime.now() - started).total_seconds()
        return CommandResult(
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_s=elapsed,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = (datetime.now() - started).total_seconds()
        return CommandResult(
            command=command,
            returncode=124,
            stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
            stderr=f"Command timed out after {spec.timeout_s} seconds.",
            duration_s=elapsed,
        )


def _append_field(args: list[str], name: str, value: Any) -> None:
    flag = "--" + name.replace("_", "-")
    if name == "tickers":
        for ticker in _split_tickers(str(value)):
            args.extend(["--tickers", ticker])
        return
    args.extend([flag, str(value)])


def _append_checkbox(args: list[str], name: str, value: bool) -> None:
    if not value:
        if name == "rag":
            args.append("--no-rag")
        return
    args.append("--" + name.replace("_", "-"))


def _split_tickers(value: str) -> list[str]:
    return [item.strip().upper() for item in value.replace(",", " ").split() if item.strip()]
