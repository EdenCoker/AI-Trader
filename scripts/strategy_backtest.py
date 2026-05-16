"""
Strategy backtester - tests multiple signal-based strategies on training data.
Each strategy is a filter function over LocalTrainingExample records.
Metric: equal-weight monthly return -> Sharpe, CAGR, max drawdown.

Key design choices:
- No global deduplication: lobby/patent/wsb signals have near-zero conviction so
  dedup-by-conviction would wipe them out. Each strategy filters the raw universe.
- Minimum 5 trades per month for a month to count in the equity curve.
- Outlier filter: -95% to +300%.
"""

import datetime
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ai_trader.backtesting.monte_carlo import StressMonteCarlo

MIN_TRADES_PER_MONTH = 5  # months with fewer trades are excluded from equity curve
CONVICTION_MODE = "plan"
CONVICTION_MODE_CHOICES = ("plan", "bundle", "agreement_adjusted")


# ---------------------------------------------------------------------------
# Load & clean
# ---------------------------------------------------------------------------


def load(
    path: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict]:
    examples = []
    for line in Path(path).read_bytes().decode("utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        e = json.loads(line)
        pnl = e["pnl_pct"]
        # Remove extreme outliers (penny stocks / data errors)
        if not (-0.95 <= pnl <= 3.0):
            continue
        as_of = e["metadata"].get("as_of", "")[:10]
        if start_date and as_of < start_date:
            continue
        if end_date and as_of > end_date:
            continue
        examples.append(e)
    return examples


def fetch_spy_return(start_date: str, end_date: str) -> float | None:
    """Fetch SPY total return for the period using Polygon API. Returns None on failure."""
    if httpx is None:
        return None
    api_key = os.environ.get("POLYGON_API_KEY", "")
    if not api_key:
        # Try loading from .env
        env_path = Path(__file__).resolve().parents[1] / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("POLYGON_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not api_key:
        return None
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/{start_date}/{end_date}"
        r = httpx.get(
            url,
            params={"adjusted": "true", "sort": "asc", "limit": 5000, "apiKey": api_key},
            timeout=20,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if len(results) < 2:
            return None
        start_price = results[0]["c"]
        end_price = results[-1]["c"]
        return (end_price - start_price) / start_price
    except Exception as exc:
        print(f"SPY fetch failed: {exc}")
        return None


def signal_names(e: dict) -> set[str]:
    return {s["name"] for s in e["signal_bundle"]["signals"]}


def has_signal(e: dict, name: str) -> bool:
    return name in signal_names(e)


def fear_greed_val(e: dict) -> float:
    return float(e["metadata"].get("fear_greed", 50.0))


def direction(e: dict) -> str:
    return e["trade_plan"]["direction"]


def set_conviction_mode(mode: str) -> None:
    if mode not in CONVICTION_MODE_CHOICES:
        raise ValueError(f"unknown conviction mode: {mode}")
    global CONVICTION_MODE
    CONVICTION_MODE = mode


def conviction(e: dict) -> float:
    if CONVICTION_MODE == "bundle":
        return bundle_conviction(e)
    if CONVICTION_MODE == "agreement_adjusted":
        return agreement_adjusted_conviction(e)
    return plan_conviction(e)


def plan_conviction(e: dict) -> float:
    return float(e["trade_plan"]["conviction"])


def bundle_conviction(e: dict) -> float:
    signals = e["signal_bundle"].get("signals", [])
    if not signals:
        return 0.0
    weighted = 0.0
    confidence = 0.0
    for signal in signals:
        signal_confidence = float(signal.get("confidence", 0.5) or 0.0)
        weighted += (
            _direction_multiplier(signal.get("direction"))
            * float(signal.get("strength", 0.0) or 0.0)
            * signal_confidence
        )
        confidence += signal_confidence
    if confidence == 0:
        return 0.0
    return abs(max(-1.0, min(1.0, weighted / confidence)))


def agreement_adjusted_conviction(e: dict) -> float:
    target_direction = direction(e)
    if target_direction == "neutral":
        return 0.0

    support_weight = 0.0
    opposing_weight = 0.0
    support_weights = []
    support_names = set()
    support_count = 0
    for signal in e["signal_bundle"].get("signals", []):
        signal_direction = signal.get("direction")
        if signal_direction == "neutral":
            continue
        weight = float(signal.get("strength", 0.0) or 0.0) * float(
            signal.get("confidence", 0.5) or 0.0
        )
        if signal_direction == target_direction:
            support_weight += weight
            support_weights.append(weight)
            support_names.add(signal.get("name", ""))
            support_count += 1
        else:
            opposing_weight += weight

    total_directional_weight = support_weight + opposing_weight
    if support_count == 0 or total_directional_weight <= 0:
        return 0.0

    agreement = max(0.0, (support_weight - opposing_weight) / total_directional_weight)
    avg_support = support_weight / support_count
    breadth = min(1.0, math.sqrt(len(support_names) / 3.0))
    max_share = max(support_weights) / support_weight if support_weight > 0 else 0.0
    concentration_penalty = (
        1.0 if max_share <= 0.55 else max(0.35, 1.0 - ((max_share - 0.55) / 0.45))
    )
    score = (
        agreement
        * math.sqrt(max(0.0, min(1.0, avg_support)))
        * (0.55 + 0.45 * breadth)
        * concentration_penalty
    )
    return max(0.0, min(1.0, score))


def _direction_multiplier(signal_direction: str | None) -> int:
    if signal_direction == "long":
        return 1
    if signal_direction == "short":
        return -1
    return 0


def wsb_mentions(e: dict) -> int:
    return int(e["metadata"].get("wsb_mentions", 0))


def congress_net(e: dict) -> int:
    return int(e["metadata"].get("congress_buy", 0)) - int(e["metadata"].get("congress_sell", 0))


def metadata_float(e: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(e["metadata"].get(key, default) or default)
    except (TypeError, ValueError):
        return default


def signal_horizon(e: dict, name: str) -> int:
    for signal in e["signal_bundle"]["signals"]:
        if signal["name"] == name:
            return int(signal.get("horizon_days", 0))
    return 0


def annotate_combo_insider_congress(examples: list[dict]) -> None:
    congress_buys = defaultdict(list)
    insider_dates = defaultdict(list)
    for example in examples:
        ticker = example["metadata"].get("ticker") or example["signal_bundle"]["ticker"]
        as_of = example["metadata"].get("as_of", "")[:10]
        if not ticker or not as_of:
            continue
        if congress_net(example) > 0:
            congress_buys[ticker].append(datetime.date.fromisoformat(as_of))
        if has_signal(example, "insider_buy"):
            insider_dates[ticker].append(datetime.date.fromisoformat(as_of))

    for example in examples:
        ticker = example["metadata"].get("ticker") or example["signal_bundle"]["ticker"]
        as_of_raw = example["metadata"].get("as_of", "")[:10]
        if not ticker or not as_of_raw or not has_signal(example, "insider_buy"):
            example["metadata"]["combo_insider_congress"] = False
            continue
        as_of = datetime.date.fromisoformat(as_of_raw)
        example["metadata"]["combo_insider_congress"] = any(
            abs((as_of - congress_date).days) <= 30
            for congress_date in congress_buys.get(ticker, [])
        )


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

STRATEGIES = {
    # ---- Baselines ----
    "baseline_all_longs": lambda e: direction(e) == "long",
    "baseline_high_conviction_long": lambda e: conviction(e) >= 0.27 and direction(e) == "long",
    "baseline_top10pct_conviction": lambda e: conviction(e) >= 0.33 and direction(e) == "long",
    # ---- Lobbying ----
    "lobbying_long": lambda e: has_signal(e, "lobbying_activity") and direction(e) == "long",
    "lobbying_greed_long": lambda e: (
        has_signal(e, "lobbying_activity") and fear_greed_val(e) >= 45 and direction(e) == "long"
    ),
    "lobbying_no_wsb": lambda e: (
        has_signal(e, "lobbying_activity")
        and direction(e) == "long"
        and not has_signal(e, "wsb_sentiment")
    ),
    # ---- Congress (metadata: congress_buy > congress_sell) ----
    "congress_net_buy_long": lambda e: congress_net(e) > 0 and direction(e) == "long",
    "congress_net_buy_greed_long": lambda e: (
        congress_net(e) > 0 and fear_greed_val(e) >= 45 and direction(e) == "long"
    ),
    "congress_strong_buy_long": lambda e: congress_net(e) >= 3 and direction(e) == "long",
    # ---- Lobby + Congress together ----
    "lobby_or_congress_long": lambda e: (
        (has_signal(e, "lobbying_activity") or congress_net(e) > 0) and direction(e) == "long"
    ),
    "lobby_or_congress_greed": lambda e: (
        (has_signal(e, "lobbying_activity") or congress_net(e) > 0)
        and fear_greed_val(e) >= 50
        and direction(e) == "long"
        and not has_signal(e, "wsb_sentiment")
    ),
    # ---- Fear & Greed macro ----
    "greed_long_no_wsb": lambda e: (
        fear_greed_val(e) >= 60 and direction(e) == "long" and not has_signal(e, "wsb_sentiment")
    ),
    "fear_contrarian_long": lambda e: (
        fear_greed_val(e) <= 25 and direction(e) == "long" and not has_signal(e, "wsb_sentiment")
    ),
    "extreme_greed_long": lambda e: (
        fear_greed_val(e) >= 75 and direction(e) == "long" and not has_signal(e, "wsb_sentiment")
    ),
    # ---- WSB contrarian (short the Reddit crowd) ----
    "wsb_contrarian_short": lambda e: has_signal(e, "wsb_sentiment") and direction(e) == "short",
    "wsb_contrarian_high_mentions": lambda e: (
        has_signal(e, "wsb_sentiment") and wsb_mentions(e) >= 20 and direction(e) == "short"
    ),
    # ---- Patent signals ----
    "patent_long": lambda e: has_signal(e, "patent_filings") and direction(e) == "long",
    "patent_high_conviction_long": lambda e: (
        has_signal(e, "patent_filings") and conviction(e) >= 0.25 and direction(e) == "long"
    ),
    # ---- Expanded signal set ----
    "insider_buy_short": lambda e: (
        has_signal(e, "insider_buy")
        and signal_horizon(e, "insider_buy") <= 20
        and conviction(e) >= 0.25
        and direction(e) == "long"
    ),
    "earnings_beat_drift": lambda e: (
        has_signal(e, "earnings_beat")
        and metadata_float(e, "eps_surprise_pct") > 10
        and not has_signal(e, "wsb_sentiment")
        and direction(e) == "long"
    ),
    "macro_bullish_lobby": lambda e: (
        metadata_float(e, "ism_pmi") > 52
        and has_signal(e, "lobbying_activity")
        and direction(e) == "long"
    ),
    "short_squeeze_setup": lambda e: (
        has_signal(e, "short_squeeze")
        and metadata_float(e, "days_to_cover") > 10
        and metadata_float(e, "price_momentum_20d") > 0
        and direction(e) == "long"
    ),
    "combo_insider_congress": lambda e: (
        bool(e["metadata"].get("combo_insider_congress")) and direction(e) == "long"
    ),
    # ---- Kitchen-sink combos ----
    "lobby_congress_fear_greed_long": lambda e: (
        (has_signal(e, "lobbying_activity") or congress_net(e) >= 2)
        and 40 <= fear_greed_val(e) <= 75
        and direction(e) == "long"
    ),
    "no_wsb_high_conviction_congress_long": lambda e: (
        congress_net(e) > 0
        and conviction(e) >= 0.20
        and direction(e) == "long"
        and not has_signal(e, "wsb_sentiment")
    ),
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def monthly_returns(trades: list[dict]) -> dict[str, list[float]]:
    m = defaultdict(list)
    for e in trades:
        m[e["metadata"]["as_of"][:7]].append(e["pnl_pct"])
    return m


def compute_metrics(trades: list[dict], starting_balance: float = 1_000_000.0) -> dict:
    if not trades:
        return {
            "n": 0,
            "cagr": None,
            "sharpe": None,
            "max_dd": None,
            "win_pct": None,
            "mean_ret": None,
            "calmar": None,
            "final_balance": starting_balance,
            "total_pnl": 0.0,
            "max_dd_dollars": 0.0,
        }

    returns = [e["pnl_pct"] for e in trades]
    wins = [r for r in returns if r > 0]

    monthly = monthly_returns(trades)
    # Only include months with enough trades for reliable signal
    monthly_avgs = {
        m: statistics.mean(v) for m, v in monthly.items() if len(v) >= MIN_TRADES_PER_MONTH
    }

    if len(monthly_avgs) < 2:
        return {
            "n": len(trades),
            "months": 0,
            "cagr": None,
            "sharpe": None,
            "max_dd": None,
            "win_pct": len(wins) / len(returns) * 100,
            "mean_ret": statistics.mean(returns) * 100,
            "calmar": None,
            "final_balance": starting_balance,
            "total_pnl": 0.0,
            "max_dd_dollars": 0.0,
        }

    balance = starting_balance
    equity = 1.0
    peak = 1.0
    peak_balance = starting_balance
    max_dd = 0.0
    max_dd_dollars = 0.0
    monthly_seq = []
    monthly_balances: dict[str, float] = {}
    for m in sorted(monthly_avgs):
        r = monthly_avgs[m]
        equity *= 1 + r
        balance *= 1 + r
        monthly_seq.append(r)
        monthly_balances[m] = balance
        if equity > peak:
            peak = equity
            peak_balance = balance
        dd = (peak - equity) / peak
        if dd > max_dd:
            max_dd = dd
            max_dd_dollars = peak_balance - balance

    years = max(len(set(m[:4] for m in monthly_avgs)), 1)
    cagr = ((equity ** (1 / years)) - 1) * 100

    mean_m = statistics.mean(monthly_seq)
    std_m = statistics.stdev(monthly_seq) if len(monthly_seq) >= 2 else 0.0
    sharpe = (mean_m / std_m * math.sqrt(12)) if std_m > 0 else 0.0

    calmar = cagr / (max_dd * 100) if max_dd > 0 else float("inf")

    return {
        "n": len(trades),
        "months": len(monthly_avgs),
        "win_pct": len(wins) / len(returns) * 100,
        "mean_ret": statistics.mean(returns) * 100,
        "cagr": cagr,
        "sharpe": round(sharpe, 2),
        "max_dd": max_dd * 100,
        "calmar": round(calmar, 2),
        "final_balance": round(balance, 2),
        "total_pnl": round(balance - starting_balance, 2),
        "max_dd_dollars": round(max_dd_dollars, 2),
        "monthly_balances": monthly_balances,
    }


def robustness_profile(
    trades: list[dict],
    *,
    split_date: str | None = None,
    max_ticker_share: float = 0.25,
    max_month_share: float = 0.30,
    max_drawdown_pct: float = 75.0,
    min_active_years: int = 3,
    min_oos_trades: int = 50,
) -> dict:
    if not trades:
        return {"passed": False, "notes": ("no trades",)}

    ticker_counts = Counter(_ticker(e) for e in trades)
    month_counts = Counter(e["metadata"]["as_of"][:7] for e in trades)
    active_years = len({e["metadata"]["as_of"][:4] for e in trades})
    metrics = compute_metrics(trades)
    max_ticker = max(ticker_counts.values()) / len(trades)
    max_month = max(month_counts.values()) / len(trades)
    notes = []
    if active_years < min_active_years:
        notes.append(f"active_years={active_years}<{min_active_years}")
    if max_ticker > max_ticker_share:
        ticker, count = ticker_counts.most_common(1)[0]
        notes.append(f"ticker_concentration={ticker}:{count / len(trades):.1%}")
    if max_month > max_month_share:
        month, count = month_counts.most_common(1)[0]
        notes.append(f"month_concentration={month}:{count / len(trades):.1%}")
    if metrics.get("max_dd") is not None and metrics["max_dd"] > max_drawdown_pct:
        notes.append(f"max_drawdown={metrics['max_dd']:.1f}%>{max_drawdown_pct:.1f}%")

    oos_metrics = None
    if split_date:
        oos = [e for e in trades if e["metadata"]["as_of"][:10] >= split_date]
        oos_metrics = compute_metrics(oos)
        if len(oos) < min_oos_trades:
            notes.append(f"oos_trades={len(oos)}<{min_oos_trades}")
        if oos and (oos_metrics.get("mean_ret") or 0.0) <= 0:
            notes.append(f"oos_mean={oos_metrics['mean_ret']:.2f}%<=0")
        if oos_metrics.get("sharpe") is not None and oos_metrics["sharpe"] <= 0:
            notes.append(f"oos_sharpe={oos_metrics['sharpe']:.2f}<=0")

    return {
        "passed": not notes,
        "notes": tuple(notes),
        "active_years": active_years,
        "max_ticker_concentration": max_ticker,
        "max_month_concentration": max_month,
        "top_tickers": ticker_counts.most_common(5),
        "top_months": month_counts.most_common(5),
        "oos_metrics": oos_metrics,
    }


def conviction_diagnostic_report(
    universe: list[dict],
    *,
    mode: str,
    top_fraction: float = 0.10,
    starting_balance: float = 1_000_000.0,
    split_date: str | None = None,
) -> dict:
    original_mode = CONVICTION_MODE
    set_conviction_mode(mode)
    try:
        longs = [e for e in universe if direction(e) == "long"]
        threshold = _top_fraction_threshold([conviction(e) for e in longs], top_fraction)
        selected = [e for e in longs if conviction(e) >= threshold]
        signal_counter = Counter()
        signal_direction_counter = Counter()
        combo_counter = Counter()
        for example in selected:
            names = sorted(signal_names(example))
            signal_counter.update(names)
            signal_direction_counter.update(
                (signal.get("name"), signal.get("direction"))
                for signal in example["signal_bundle"].get("signals", [])
            )
            if len(names) <= 5:
                combo_counter[tuple(names)] += 1

        return {
            "mode": mode,
            "threshold": threshold,
            "selected": selected,
            "metrics": compute_metrics(selected, starting_balance=starting_balance),
            "robustness": robustness_profile(selected, split_date=split_date),
            "years": Counter(e["metadata"]["as_of"][:4] for e in selected).most_common(),
            "top_tickers": Counter(_ticker(e) for e in selected).most_common(10),
            "top_signals": signal_counter.most_common(15),
            "top_signal_directions": signal_direction_counter.most_common(15),
            "top_combos": combo_counter.most_common(8),
        }
    finally:
        set_conviction_mode(original_mode)


def print_conviction_diagnostics(
    universe: list[dict],
    *,
    starting_balance: float,
    split_date: str | None,
) -> None:
    print("\nConviction diagnostics (long top decile by score mode)")
    print("-" * 80)
    for mode in CONVICTION_MODE_CHOICES:
        report = conviction_diagnostic_report(
            universe,
            mode=mode,
            starting_balance=starting_balance,
            split_date=split_date,
        )
        metrics = report["metrics"]
        robustness = report["robustness"]
        print(
            f"\n[{mode}] threshold={report['threshold']:.4f} "
            f"N={metrics['n']} win={metrics['win_pct']:.1f}% "
            f"mean={metrics['mean_ret']:+.2f}% sharpe={metrics['sharpe']} "
            f"maxDD={metrics['max_dd']:.1f}% robust={'PASS' if robustness['passed'] else 'FAIL'}"
        )
        if robustness["notes"]:
            print("  Checks:", "; ".join(robustness["notes"]))
        print("  Years:", _format_counter(report["years"], limit=8))
        print("  Tickers:", _format_counter(report["top_tickers"], limit=8))
        print("  Signals:", _format_counter(report["top_signals"], limit=10))
        print("  Signal dirs:", _format_counter(report["top_signal_directions"], limit=8))
        print("  Combos:", _format_counter(report["top_combos"], limit=5))


def _top_fraction_threshold(values: list[float], fraction: float) -> float:
    if not values:
        return 1.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * (1.0 - fraction)) - 1))
    return ordered[index]


def _ticker(e: dict) -> str:
    return str(e["metadata"].get("ticker") or e["signal_bundle"].get("ticker", "")).upper()


def _format_counter(items, *, limit: int) -> str:
    formatted = []
    for key, count in list(items)[:limit]:
        if isinstance(key, tuple):
            key = "/".join(str(part) for part in key)
        formatted.append(f"{key}:{count}")
    return ", ".join(formatted) if formatted else "-"


def walk_forward_strategy_report(
    universe: list[dict],
    *,
    starting_balance: float,
    train_days: int = 252,
    test_days: int = 63,
    step_days: int = 21,
) -> list[dict]:
    if not universe:
        return []
    dated = sorted(universe, key=lambda e: e["metadata"].get("as_of", "")[:10])
    start = datetime.date.fromisoformat(dated[0]["metadata"]["as_of"][:10])
    end = datetime.date.fromisoformat(dated[-1]["metadata"]["as_of"][:10])
    rows = []
    cursor = start + datetime.timedelta(days=train_days)
    while cursor + datetime.timedelta(days=test_days) <= end:
        train_start = cursor - datetime.timedelta(days=train_days)
        train_end = cursor - datetime.timedelta(days=1)
        test_start = cursor
        test_end = cursor + datetime.timedelta(days=test_days - 1)
        for name, fn in STRATEGIES.items():
            train = [e for e in universe if train_start <= _example_date(e) <= train_end and fn(e)]
            test = [e for e in universe if test_start <= _example_date(e) <= test_end and fn(e)]
            train_metrics = compute_metrics(train, starting_balance=starting_balance)
            test_metrics = compute_metrics(test, starting_balance=starting_balance)
            rows.append(
                {
                    "strategy": name,
                    "train_start": train_start.isoformat(),
                    "train_end": train_end.isoformat(),
                    "test_start": test_start.isoformat(),
                    "test_end": test_end.isoformat(),
                    "train_sharpe": train_metrics.get("sharpe"),
                    "test_sharpe": test_metrics.get("sharpe"),
                    "train_n": train_metrics.get("n", 0),
                    "test_n": test_metrics.get("n", 0),
                }
            )
        cursor += datetime.timedelta(days=step_days)
    return rows


def summarize_walk_forward(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["strategy"]].append(row)
    summary = []
    for strategy, strategy_rows in grouped.items():
        train_sharpes = [r["train_sharpe"] for r in strategy_rows if r["train_sharpe"] is not None]
        test_sharpes = [r["test_sharpe"] for r in strategy_rows if r["test_sharpe"] is not None]
        summary.append(
            {
                "strategy": strategy,
                "windows": len(strategy_rows),
                "avg_train_sharpe": statistics.mean(train_sharpes) if train_sharpes else None,
                "avg_test_sharpe": statistics.mean(test_sharpes) if test_sharpes else None,
                "test_trades": sum(r["test_n"] for r in strategy_rows),
            }
        )
    summary.sort(
        key=lambda row: row["avg_test_sharpe"] if row["avg_test_sharpe"] is not None else -99,
        reverse=True,
    )
    return summary


def run_monte_carlo(trades: list[dict], starting_balance: float, n_sims: int = 1_000) -> dict:
    returns = [e["pnl_pct"] for e in trades]
    result = StressMonteCarlo(n_simulations=n_sims).run(returns)  # type: ignore[arg-type]
    return {
        **result.__dict__,
        "terminal_pnl_p5": starting_balance * result.terminal_return_p5,
        "terminal_pnl_p95": starting_balance * result.terminal_return_p95,
    }


def _example_date(example: dict) -> datetime.date:
    return datetime.date.fromisoformat(example["metadata"]["as_of"][:10])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Strategy backtester with optional date filter and SPY benchmark"
    )
    parser.add_argument("path", nargs="?", default="logs/training_examples.jsonl")
    parser.add_argument("starting_balance", nargs="?", type=float, default=999_000.0)
    parser.add_argument(
        "--start-date", default=None, help="Filter examples from YYYY-MM-DD (inclusive)"
    )
    parser.add_argument(
        "--end-date", default=None, help="Filter examples to YYYY-MM-DD (inclusive)"
    )
    parser.add_argument(
        "--split-date", default=None, help="Optional OOS split date for robustness checks"
    )
    parser.add_argument("--conviction-mode", choices=CONVICTION_MODE_CHOICES, default="plan")
    parser.add_argument(
        "--diagnose-conviction", action="store_true", help="Explain top-decile conviction buckets"
    )
    parser.add_argument(
        "--robustness-checks", action="store_true", help="Flag concentrated or weak-OOS strategies"
    )
    parser.add_argument(
        "--walk-forward", action="store_true", help="Run 252/63/21 rolling strategy validation"
    )
    parser.add_argument(
        "--monte-carlo", action="store_true", help="Run 1,000 random return shuffles on the winner"
    )
    parser.add_argument("--no-spy", action="store_true", help="Skip SPY benchmark fetch")
    args = parser.parse_args()
    set_conviction_mode(args.conviction_mode)

    print(f"Loading {args.path}...")
    if args.start_date or args.end_date:
        print(f"Date filter: {args.start_date or 'beginning'} -> {args.end_date or 'end'}")
    print(f"Conviction mode: {CONVICTION_MODE}")
    universe = load(args.path, start_date=args.start_date, end_date=args.end_date)
    annotate_combo_insider_congress(universe)
    starting_balance = args.starting_balance
    print(f"Universe: {len(universe):,} examples after outlier filter")
    print(f"Starting balance: ${starting_balance:,.0f}")
    print(f"(Monthly equity only counts months with >={MIN_TRADES_PER_MONTH} trades)\n")

    # Fetch SPY benchmark for the same period
    spy_start = args.start_date or (
        min(e["metadata"]["as_of"][:10] for e in universe) if universe else None
    )
    spy_end = args.end_date or (
        max(e["metadata"]["as_of"][:10] for e in universe) if universe else None
    )
    spy_return = None
    if spy_start and spy_end and not args.no_spy:
        print(f"Fetching SPY benchmark {spy_start} -> {spy_end} ...")
        spy_return = fetch_spy_return(spy_start, spy_end)
        if spy_return is not None:
            print(f"SPY return over period: {spy_return * 100:+.2f}%\n")
        else:
            print("SPY benchmark unavailable (no POLYGON_API_KEY or fetch failed)\n")
    elif args.no_spy:
        print("SPY benchmark skipped (--no-spy)\n")

    if args.diagnose_conviction:
        print_conviction_diagnostics(
            universe,
            starting_balance=starting_balance,
            split_date=args.split_date,
        )

    results = []
    for name, fn in STRATEGIES.items():
        selected = [e for e in universe if fn(e)]
        m = compute_metrics(selected, starting_balance=starting_balance)
        m["strategy"] = name
        if args.robustness_checks:
            m["robustness"] = robustness_profile(selected, split_date=args.split_date)
        results.append(m)

    # Sort by Sharpe descending
    results.sort(key=lambda r: r.get("sharpe") or -99, reverse=True)

    # Compute SPY-equivalent balance for comparison column
    spy_final = round(starting_balance * (1 + spy_return), 2) if spy_return is not None else None
    spy_pnl = round(starting_balance * spy_return, 2) if spy_return is not None else None

    # Print table
    col_w = 38
    check_header = " Check" if args.robustness_checks else ""
    header = (
        f"{'Strategy':<{col_w}} {'N':>6} {'Win%':>6} {'Mean%':>7} "
        f"{'CAGR%':>7} {'Shrpe':>6} {'MaxDD%':>7} {'Final $':>14} "
        f"{'P&L $':>13} {'vs SPY':>8}{check_header}"
    )
    print(header)
    print("-" * (col_w + 92 if args.robustness_checks else col_w + 85))
    for r in results:
        n = r["n"]
        if n == 0:
            print(f"{r['strategy']:<{col_w}} {'(no trades)':>55}")
            continue
        wp = f"{r['win_pct']:.1f}" if r["win_pct"] is not None else "-"
        mr = f"{r['mean_ret']:+.2f}" if r["mean_ret"] is not None else "-"
        cg = f"{r['cagr']:+.1f}" if r["cagr"] is not None else "-"
        sh = f"{r['sharpe']:.2f}" if r["sharpe"] is not None else "-"
        dd = f"{r['max_dd']:.1f}" if r["max_dd"] is not None else "-"
        fb = f"${r['final_balance']:>12,.0f}" if r.get("final_balance") else "-"
        pnl = r.get("total_pnl", 0.0)
        pnl_s = f"{'+' if pnl >= 0 else ''}${pnl:>10,.0f}"
        if spy_pnl is not None:
            alpha = pnl - spy_pnl
            vs_spy = f"{'+' if alpha >= 0 else ''}${alpha:,.0f}"
        else:
            vs_spy = "-"
        check = ""
        if args.robustness_checks:
            robust = r.get("robustness", {})
            check = " PASS" if robust.get("passed") else " FAIL"
        row = (
            f"{r['strategy']:<{col_w}} {n:>6} {wp:>6} {mr:>7} "
            f"{cg:>7} {sh:>6} {dd:>7} {fb:>14} {pnl_s:>13} "
            f"{vs_spy:>8}{check}"
        )
        print(row)

    if args.robustness_checks:
        print("\nRobustness notes:")
        for r in results:
            robust = r.get("robustness", {})
            notes = robust.get("notes") or ()
            if r["n"] and notes:
                print(f"   {r['strategy']}: {'; '.join(notes)}")

    if spy_return is not None:
        spy_pnl_s = f"{'+' if spy_pnl >= 0 else ''}${spy_pnl:>10,.0f}"
        spy_row = (
            f"{'[SPY benchmark]':<{col_w}} {'':>6} {'':>6} "
            f"{spy_return * 100:>+7.2f} {'':>7} {'':>6} {'':>7} "
            f"${spy_final:>12,.0f} {spy_pnl_s:>13} {'+/-$0':>8}"
        )
        print(spy_row)

    print("\n" + "-" * (col_w + 85))
    robust_results = [
        result
        for result in results
        if not args.robustness_checks or result.get("robustness", {}).get("passed")
    ]
    winner = robust_results[0] if robust_results else results[0]
    print(f"\nBest strategy: {winner['strategy']}")
    winner_line = (
        f"   N={winner['n']}, Sharpe={winner['sharpe']}, "
        f"CAGR={winner['cagr']:+.1f}%, MaxDD={winner['max_dd']:.1f}%"
    )
    print(winner_line)
    print(f"   Starting balance: ${starting_balance:>12,.0f}")
    print(f"   Final balance:    ${winner['final_balance']:>12,.0f}")
    pnl = winner["total_pnl"]
    print(f"   Total P&L:        {'+' if pnl >= 0 else ''}${pnl:>12,.0f}")
    print(f"   Max DD ($):      -${winner['max_dd_dollars']:>12,.0f}")
    if spy_return is not None:
        alpha = pnl - spy_pnl
        print(f"   SPY over period:  {spy_return * 100:>+12.2f}%  (${spy_pnl:>+,.0f})")
        print(f"   Alpha vs SPY:     {'+' if alpha >= 0 else ''}${alpha:>12,.0f}")

    selected = [e for e in universe if STRATEGIES[winner["strategy"]](e)]
    if args.monte_carlo:
        mc = run_monte_carlo(selected, starting_balance=starting_balance, n_sims=1_000)
        print("\n   Monte Carlo stress (1,000 shuffles of winning strategy returns):")
        print(
            "   Terminal return p5/p50/p95: "
            f"{mc['terminal_return_p5']:+.1%} / "
            f"{mc['terminal_return_p50']:+.1%} / "
            f"{mc['terminal_return_p95']:+.1%}"
        )
        print(
            f"   P&L band p5/p95: ${mc['terminal_pnl_p5']:+,.0f} / ${mc['terminal_pnl_p95']:+,.0f}"
        )
        print(
            "   Sharpe p5/p50/p95: "
            f"{mc['sharpe_p5']:.2f} / {mc['sharpe_p50']:.2f} / {mc['sharpe_p95']:.2f}"
        )

    if args.walk_forward:
        wf_rows = walk_forward_strategy_report(universe, starting_balance=starting_balance)
        wf_summary = summarize_walk_forward(wf_rows)
        print("\n   Walk-forward validation (252d train / 63d test / 21d step):")
        wf_header = (
            f"   {'Strategy':<{col_w}} {'Win':>4} {'Train Sh':>10} {'Test Sh':>9} {'Test N':>8}"
        )
        print(wf_header)
        for row in wf_summary[:15]:
            train_sh = "-" if row["avg_train_sharpe"] is None else f"{row['avg_train_sharpe']:.2f}"
            test_sh = "-" if row["avg_test_sharpe"] is None else f"{row['avg_test_sharpe']:.2f}"
            print(
                f"   {row['strategy']:<{col_w}} {row['windows']:>4} "
                f"{train_sh:>10} {test_sh:>9} {row['test_trades']:>8}"
            )

    # Detailed monthly breakdown of winner
    monthly = monthly_returns(selected)
    monthly_balances = winner.get("monthly_balances", {})
    print(f"\n   Monthly equity curve ({len(monthly)} months):")
    for m in sorted(monthly)[-24:]:
        avg = statistics.mean(monthly[m]) * 100
        n = len(monthly[m])
        bal_val = monthly_balances.get(m, None)
        bar_len = max(0, int(abs(avg) * 2))
        bar = ("#" * bar_len)[:40]
        sign = "+" if avg >= 0 else ""
        flag = "^" if avg >= 0 else "v"
        bal_str = f"  balance=${bal_val:>12,.0f}" if bal_val else ""
        print(f"   {m}  {flag} {sign}{avg:5.2f}%  n={n:>4}{bal_str}  {bar}")


if __name__ == "__main__":
    main()
