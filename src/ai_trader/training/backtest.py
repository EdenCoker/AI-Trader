from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from statistics import mean, median, stdev

from pydantic import BaseModel, ConfigDict, Field

from ai_trader.domain.signals import SignalDirection
from ai_trader.training.conviction import (
    ConvictionMetric,
    normalize_conviction_metric,
    score_training_example,
)
from ai_trader.training.data import LocalTrainingExample


class StrategyRule(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    direction: SignalDirection
    min_conviction: float = Field(default=0.0, ge=0.0, le=1.0)
    conviction_metric: ConvictionMetric = ConvictionMetric.PLAN
    signal_name: str | None = None
    min_fear_greed: float | None = None
    max_fear_greed: float | None = None
    exclude_wsb: bool = False


class StrategyMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    trades: int = 0
    active_months: int = 0
    active_years: int = 0
    excluded_sparse_months: int = 0
    max_ticker_concentration: float = 0.0
    max_month_concentration: float = 0.0
    win_rate: float = 0.0
    mean_return: float = 0.0
    median_return: float = 0.0
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    score: float = 0.0
    monthly_returns: dict[str, float] = Field(default_factory=dict)


class StrategyEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True)

    rule: StrategyRule
    metrics: StrategyMetrics
    train_metrics: StrategyMetrics | None = None
    test_metrics: StrategyMetrics | None = None
    robustness_passed: bool = True
    robustness_notes: tuple[str, ...] = ()


class StrategyBacktestConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    start_date: date | None = None
    end_date: date | None = None
    split_date: date | None = None
    conviction_metric: ConvictionMetric = ConvictionMetric.PLAN
    min_trades: int = Field(default=50, ge=1)
    min_active_months: int = Field(default=3, ge=1)
    min_trades_per_month: int = Field(default=5, ge=1)
    outlier_floor: float = -0.95
    outlier_cap: float = 3.0
    top_n: int = Field(default=10, ge=1)
    dedupe_by_ticker_date: bool = True
    min_active_years: int = Field(default=1, ge=1)
    max_ticker_concentration: float | None = Field(default=None, gt=0.0, le=1.0)
    max_month_concentration: float | None = Field(default=None, gt=0.0, le=1.0)
    max_drawdown: float | None = Field(default=None, ge=0.0, le=1.0)
    require_positive_oos_score: bool = False
    max_train_test_sharpe_decay: float | None = Field(default=None, ge=0.0)


class StrategyBacktestReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    examples_loaded: int
    examples_used: int
    examples_dropped_as_outliers: int
    candidate_count: int
    selected_strategy: StrategyEvaluation | None
    leaderboard: tuple[StrategyEvaluation, ...]


@dataclass(frozen=True)
class _ExampleView:
    example: LocalTrainingExample
    ticker: str
    as_of: date
    month: str
    pnl_pct: float
    direction: SignalDirection
    conviction: float
    conviction_scores: dict[ConvictionMetric, float]
    signal_names: frozenset[str]
    fear_greed: float


def run_strategy_backtest(
    examples: Iterable[LocalTrainingExample],
    config: StrategyBacktestConfig | None = None,
    rules: Iterable[StrategyRule] | None = None,
) -> StrategyBacktestReport:
    config = config or StrategyBacktestConfig()
    loaded = 0
    dropped = 0
    views: list[_ExampleView] = []
    for example in examples:
        loaded += 1
        if example.pnl_pct < config.outlier_floor or example.pnl_pct > config.outlier_cap:
            dropped += 1
            continue
        view = _view(example)
        if config.start_date is not None and view.as_of < config.start_date:
            continue
        if config.end_date is not None and view.as_of > config.end_date:
            continue
        views.append(view)

    if config.dedupe_by_ticker_date:
        views = _dedupe_by_highest_conviction(views)

    candidate_rules = tuple(
        rules or default_strategy_rules(conviction_metric=config.conviction_metric)
    )
    leaderboard: list[StrategyEvaluation] = []
    for rule in candidate_rules:
        selected = [view for view in views if _matches(rule, view)]
        metrics = _metrics(
            selected,
            min_trades_per_month=config.min_trades_per_month,
            min_trades=config.min_trades,
            min_active_months=config.min_active_months,
        )

        train_metrics = None
        test_metrics = None
        sort_metrics = metrics
        if config.split_date is not None:
            train = [view for view in selected if view.as_of < config.split_date]
            test = [view for view in selected if view.as_of >= config.split_date]
            train_metrics = _metrics(
                train,
                min_trades_per_month=config.min_trades_per_month,
                min_trades=config.min_trades,
                min_active_months=config.min_active_months,
            )
            test_metrics = _metrics(
                test,
                min_trades_per_month=config.min_trades_per_month,
                min_trades=config.min_trades,
                min_active_months=config.min_active_months,
            )
            sort_metrics = train_metrics

        robustness_notes = _robustness_notes(
            sort_metrics,
            config,
            test_metrics=test_metrics,
        )
        evaluation = StrategyEvaluation(
            rule=rule,
            metrics=metrics,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            robustness_passed=not robustness_notes,
            robustness_notes=tuple(robustness_notes),
        )
        if _is_eligible(sort_metrics, config, robustness_notes=robustness_notes):
            leaderboard.append(evaluation)

    leaderboard.sort(
        key=lambda evaluation: (evaluation.train_metrics or evaluation.metrics).score,
        reverse=True,
    )
    return StrategyBacktestReport(
        examples_loaded=loaded,
        examples_used=len(views),
        examples_dropped_as_outliers=dropped,
        candidate_count=len(candidate_rules),
        selected_strategy=leaderboard[0] if leaderboard else None,
        leaderboard=tuple(leaderboard[: config.top_n]),
    )


def default_strategy_rules(
    conviction_metric: ConvictionMetric | str = ConvictionMetric.PLAN,
) -> tuple[StrategyRule, ...]:
    conviction_metric = normalize_conviction_metric(conviction_metric)
    rules: list[StrategyRule] = []
    signal_names: tuple[str | None, ...] = (
        None,
        "lobbying_activity",
        "patent_filings",
        "wsb_sentiment",
        "government_contract",
        "fear_greed_macro",
        "insider_buy",
        "insider_sell",
        "earnings_beat",
        "earnings_miss",
        "macro_regime",
        "options_put_call_contrarian",
        "institutional_accumulation",
        "short_squeeze",
    )
    conviction_thresholds = (0.0, 0.10, 0.15, 0.20, 0.25, 0.27, 0.30, 0.33, 0.40)
    fear_greed_ranges: tuple[tuple[str, float | None, float | None], ...] = (
        ("any", None, None),
        ("balanced", 40.0, 75.0),
        ("constructive", 50.0, None),
        ("greed", 60.0, None),
        ("extreme_greed", 75.0, None),
        ("fear", None, 25.0),
    )
    for signal_name in signal_names:
        for direction in (SignalDirection.LONG, SignalDirection.SHORT):
            for min_conviction in conviction_thresholds:
                for range_name, min_fear_greed, max_fear_greed in fear_greed_ranges:
                    for exclude_wsb in (False, True):
                        if signal_name == "wsb_sentiment" and exclude_wsb:
                            continue
                        name = _rule_name(
                            signal_name=signal_name,
                            direction=direction,
                            min_conviction=min_conviction,
                            conviction_metric=conviction_metric,
                            fear_greed_range=range_name,
                            exclude_wsb=exclude_wsb,
                        )
                        rules.append(
                            StrategyRule(
                                name=name,
                                signal_name=signal_name,
                                direction=direction,
                                min_conviction=min_conviction,
                                conviction_metric=conviction_metric,
                                min_fear_greed=min_fear_greed,
                                max_fear_greed=max_fear_greed,
                                exclude_wsb=exclude_wsb,
                            )
                        )
    return tuple(rules)


def _view(example: LocalTrainingExample) -> _ExampleView:
    ticker = str(
        example.metadata.get("ticker") or example.signal_bundle.ticker or example.trade_plan.ticker
    ).upper()
    as_of = example.signal_bundle.as_of
    signal_names = frozenset(signal.name for signal in example.signal_bundle.signals)
    return _ExampleView(
        example=example,
        ticker=ticker,
        as_of=as_of,
        month=as_of.isoformat()[:7],
        pnl_pct=float(example.pnl_pct),
        direction=example.trade_plan.direction,
        conviction=float(example.trade_plan.conviction),
        conviction_scores={
            metric: score_training_example(example, metric) for metric in ConvictionMetric
        },
        signal_names=signal_names,
        fear_greed=float(example.metadata.get("fear_greed", 50.0)),
    )


def _dedupe_by_highest_conviction(views: Iterable[_ExampleView]) -> list[_ExampleView]:
    selected: dict[tuple[str, date], _ExampleView] = {}
    for view in views:
        key = (view.ticker, view.as_of)
        existing = selected.get(key)
        if existing is None or view.conviction > existing.conviction:
            selected[key] = view
    return sorted(selected.values(), key=lambda view: (view.as_of, view.ticker))


def _matches(rule: StrategyRule, view: _ExampleView) -> bool:
    if view.direction is not rule.direction:
        return False
    metric = normalize_conviction_metric(rule.conviction_metric)
    if view.conviction_scores.get(metric, 0.0) < rule.min_conviction:
        return False
    if rule.signal_name is not None and rule.signal_name not in view.signal_names:
        return False
    if rule.exclude_wsb and "wsb_sentiment" in view.signal_names:
        return False
    if rule.min_fear_greed is not None and view.fear_greed < rule.min_fear_greed:
        return False
    return not (rule.max_fear_greed is not None and view.fear_greed > rule.max_fear_greed)


def _metrics(
    views: Iterable[_ExampleView],
    *,
    min_trades_per_month: int,
    min_trades: int,
    min_active_months: int,
) -> StrategyMetrics:
    selected = tuple(views)
    if not selected:
        return StrategyMetrics()

    returns = [view.pnl_pct for view in selected]
    wins = [value for value in returns if value > 0]
    losses = [value for value in returns if value < 0]
    monthly = _monthly_returns(selected, min_trades_per_month=min_trades_per_month)
    monthly_values = [monthly[month] for month in sorted(monthly)]
    total_return = _compound(monthly_values) - 1.0 if monthly_values else 0.0
    years = max(_year_count(monthly), 1)
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if total_return > -1.0 else -1.0
    sharpe = _annualized_sharpe(monthly_values, periods_per_year=12)
    max_dd = _max_drawdown(monthly_values)
    profit_factor = _profit_factor(wins=wins, losses=losses)
    excluded_sparse_months = len({view.month for view in selected}) - len(monthly)
    ticker_counts = Counter(view.ticker for view in selected)
    month_counts = Counter(view.month for view in selected)
    raw_score = sharpe + cagr - max_dd + mean(returns)
    if len(selected) < min_trades or len(monthly) < min_active_months:
        raw_score -= 100.0

    return StrategyMetrics(
        trades=len(selected),
        active_months=len(monthly),
        active_years=len({view.as_of.year for view in selected}),
        excluded_sparse_months=max(0, excluded_sparse_months),
        max_ticker_concentration=max(ticker_counts.values()) / len(selected),
        max_month_concentration=max(month_counts.values()) / len(selected),
        win_rate=len(wins) / len(returns),
        mean_return=mean(returns),
        median_return=median(returns),
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        max_drawdown=max_dd,
        profit_factor=profit_factor,
        score=raw_score,
        monthly_returns=monthly,
    )


def _monthly_returns(
    views: Iterable[_ExampleView],
    *,
    min_trades_per_month: int,
) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for view in views:
        buckets[view.month].append(view.pnl_pct)
    return {
        month: mean(values)
        for month, values in buckets.items()
        if len(values) >= min_trades_per_month
    }


def _annualized_sharpe(values: list[float], periods_per_year: int) -> float:
    if len(values) < 2:
        return 0.0
    std = stdev(values)
    if std == 0:
        return 0.0
    return mean(values) / std * math.sqrt(periods_per_year)


def _compound(values: Iterable[float]) -> float:
    equity = 1.0
    for value in values:
        equity *= 1.0 + value
    return equity


def _max_drawdown(values: Iterable[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for value in values:
        equity *= 1.0 + value
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / max(peak, 1e-12))
    return max_dd


def _profit_factor(*, wins: list[float], losses: list[float]) -> float:
    loss_sum = abs(sum(losses))
    if loss_sum == 0:
        return 999.0 if wins else 0.0
    return sum(wins) / loss_sum


def _year_count(monthly_returns: dict[str, float]) -> int:
    return len({month[:4] for month in monthly_returns})


def _robustness_notes(
    metrics: StrategyMetrics,
    config: StrategyBacktestConfig,
    *,
    test_metrics: StrategyMetrics | None = None,
) -> list[str]:
    notes: list[str] = []
    if metrics.active_years < config.min_active_years:
        notes.append(f"active years {metrics.active_years} < required {config.min_active_years}")
    if (
        config.max_ticker_concentration is not None
        and metrics.max_ticker_concentration > config.max_ticker_concentration
    ):
        notes.append(
            "ticker concentration "
            f"{metrics.max_ticker_concentration:.1%} > "
            f"{config.max_ticker_concentration:.1%}"
        )
    if (
        config.max_month_concentration is not None
        and metrics.max_month_concentration > config.max_month_concentration
    ):
        notes.append(
            "month concentration "
            f"{metrics.max_month_concentration:.1%} > "
            f"{config.max_month_concentration:.1%}"
        )
    if config.max_drawdown is not None and metrics.max_drawdown > config.max_drawdown:
        notes.append(f"max drawdown {metrics.max_drawdown:.1%} > {config.max_drawdown:.1%}")
    if test_metrics is not None:
        if config.require_positive_oos_score and test_metrics.score <= 0:
            notes.append(f"out-of-sample score {test_metrics.score:.2f} <= 0")
        if (
            config.max_train_test_sharpe_decay is not None
            and metrics.sharpe - test_metrics.sharpe > config.max_train_test_sharpe_decay
        ):
            notes.append(
                "train/test sharpe decay "
                f"{metrics.sharpe - test_metrics.sharpe:.2f} > "
                f"{config.max_train_test_sharpe_decay:.2f}"
            )
    return notes


def _is_eligible(
    metrics: StrategyMetrics,
    config: StrategyBacktestConfig,
    *,
    robustness_notes: list[str],
) -> bool:
    return (
        metrics.trades >= config.min_trades
        and metrics.active_months >= config.min_active_months
        and not robustness_notes
    )


def _rule_name(
    *,
    signal_name: str | None,
    direction: SignalDirection,
    min_conviction: float,
    conviction_metric: ConvictionMetric,
    fear_greed_range: str,
    exclude_wsb: bool,
) -> str:
    signal = signal_name or "any_signal"
    metric = "" if conviction_metric is ConvictionMetric.PLAN else f"_{conviction_metric.value}"
    suffix = "_no_wsb" if exclude_wsb else ""
    return f"{signal}_{direction.value}_c{min_conviction:.2f}{metric}_{fear_greed_range}{suffix}"
