"""Nightly unattended pipeline: ingest -> retrain -> backtest -> summary."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AI-Trader nightly pipeline")
    parser.add_argument("--watchlist", default="data/watchlist.txt")
    parser.add_argument("--tickers", nargs="*", help="Override watchlist tickers")
    parser.add_argument("--examples-out", default="logs/training_examples.jsonl")
    parser.add_argument("--summary-out", default="logs/nightly_summary.json")
    parser.add_argument("--skip-backtest", action="store_true")
    args = parser.parse_args()

    tickers = _load_tickers(args.tickers, Path(args.watchlist))
    summary = {
        "started_at": dt.datetime.now(dt.UTC).isoformat(),
        "tickers": tickers,
        "steps": [],
    }
    env = os.environ.copy()
    src_path = str(Path("src").resolve())
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    ingest_cmd = [
        sys.executable,
        "scripts/ingest_training_data.py",
        "--out",
        args.examples_out,
    ]
    if tickers:
        ingest_cmd.extend(["--tickers", *tickers])
    summary["steps"].append(_run_step("ingest", ingest_cmd, env=env))

    retrain_cmd = [
        sys.executable,
        "-m",
        "ai_trader.cli",
        "train",
        "local",
        "--examples-file",
        args.examples_out,
        "--horizon",
        "all",
    ]
    summary["steps"].append(_run_step("retrain", retrain_cmd, env=env))

    if not args.skip_backtest:
        backtest_cmd = [
            sys.executable,
            "scripts/strategy_backtest.py",
            args.examples_out,
            "--walk-forward",
            "--monte-carlo",
        ]
        summary["steps"].append(_run_step("backtest", backtest_cmd, env=env))

    summary["finished_at"] = dt.datetime.now(dt.UTC).isoformat()
    summary["status"] = (
        "ok" if all(step["returncode"] == 0 for step in summary["steps"]) else "failed"
    )
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _notify(summary["status"], summary_path)
    print(json.dumps({"status": summary["status"], "summary": str(summary_path)}, indent=2))


def _run_step(name: str, command: list[str], *, env: dict[str, str]) -> dict:
    started = time.perf_counter()
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )
    duration = time.perf_counter() - started
    return {
        "name": name,
        "command": command,
        "returncode": result.returncode,
        "duration_s": round(duration, 2),
        "stdout_tail": (result.stdout or "")[-4000:],
        "stderr_tail": (result.stderr or "")[-4000:],
    }


def _load_tickers(cli_tickers: list[str] | None, watchlist_path: Path) -> list[str]:
    values = cli_tickers or []
    if not values and watchlist_path.exists():
        values = [
            line.strip()
            for line in watchlist_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    return sorted({ticker.upper() for ticker in values if ticker.strip()})


def _notify(status: str, summary_path: Path) -> None:
    try:
        from win10toast import ToastNotifier
    except ImportError:
        return
    ToastNotifier().show_toast(
        "AI-Trader nightly pipeline",
        f"{status.upper()} - {summary_path}",
        duration=8,
        threaded=True,
    )


if __name__ == "__main__":
    main()
