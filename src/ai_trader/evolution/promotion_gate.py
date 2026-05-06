from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ai_trader.evolution.backtest_pool import BacktestPoolEntry, ensure_backtest_pool
from ai_trader.evolution.reports import AgentReport, ReportBuilder


class ModelBacktestScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    sharpe: float
    max_drawdown: float
    win_rate: float = 0.0
    training_count: int = 0
    metadata: dict = Field(default_factory=dict)


class BacktestComparison(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    baseline_sharpe: float
    candidate_sharpe: float
    baseline_max_drawdown: float
    candidate_max_drawdown: float
    passed: bool


ScoreRunner = Callable[[Path, BacktestPoolEntry], ModelBacktestScore]


class PromotionGate:
    """Runs the randomized promotion challenge for a candidate model."""

    def __init__(
        self,
        *,
        current: Path = Path("data/models/production.json"),
        candidate: Path = Path("data/models/candidate.json"),
        pool: Path = Path("data/backtest_pool.json"),
        version_registry: Path = Path("data/models/version_registry.json"),
        k: int = 3,
        tolerance: float = 0.05,
        min_training_count: int = 10_000,
        max_drawdown_limit: float = 0.30,
        min_win_rate: float = 0.48,
        min_hold_days: int = 5,
        run_id: str = "manual",
        random_seed: int | None = None,
        score_runner: ScoreRunner | None = None,
    ) -> None:
        self.current = current
        self.candidate = candidate
        self.pool = pool
        self.version_registry = version_registry
        self.k = k
        self.tolerance = tolerance
        self.min_training_count = min_training_count
        self.max_drawdown_limit = max_drawdown_limit
        self.min_win_rate = min_win_rate
        self.min_hold_days = min_hold_days
        self.run_id = run_id
        self.random_seed = random_seed
        self.score_runner = score_runner or metadata_backtest_score

    def run(self) -> AgentReport:
        builder = ReportBuilder("PromotionGate", self.run_id)
        try:
            if not self.candidate.exists():
                return builder.build(status="failed", errors=[f"{self.candidate} does not exist"])

            pool = ensure_backtest_pool(self.pool)
            if not pool.entries:
                return builder.build(status="failed", errors=["backtest pool is empty"])

            hold_error = self._hold_period_error()
            if hold_error is not None:
                return builder.build(status="ok", summary={"promoted": False, "reason": hold_error})

            selected = self._select(pool.entries)
            comparisons: list[BacktestComparison] = []
            candidate_scores: list[ModelBacktestScore] = []
            current_scores: list[ModelBacktestScore] = []
            current_exists = self.current.exists()

            for entry in selected:
                baseline = (
                    self.score_runner(self.current, entry)
                    if current_exists
                    else ModelBacktestScore(sharpe=0.0, max_drawdown=1.0)
                )
                candidate = self.score_runner(self.candidate, entry)
                current_scores.append(baseline)
                candidate_scores.append(candidate)
                comparisons.append(
                    BacktestComparison(
                        id=entry.id,
                        baseline_sharpe=baseline.sharpe,
                        candidate_sharpe=candidate.sharpe,
                        baseline_max_drawdown=baseline.max_drawdown,
                        candidate_max_drawdown=candidate.max_drawdown,
                        passed=candidate.sharpe >= baseline.sharpe - self.tolerance,
                    )
                )

            wins = sum(1 for item in comparisons if item.passed)
            candidate_max_dd = max(score.max_drawdown for score in candidate_scores)
            current_max_dd = max((score.max_drawdown for score in current_scores), default=1.0)
            candidate_win_rate = _mean(score.win_rate for score in candidate_scores)
            candidate_training_count = min(score.training_count for score in candidate_scores)
            sharpe_delta = _mean(
                item.candidate_sharpe - item.baseline_sharpe for item in comparisons
            )

            safety_errors = self._safety_errors(
                training_count=candidate_training_count,
                max_drawdown=candidate_max_dd,
                win_rate=candidate_win_rate,
            )
            drawdown_ok = candidate_max_dd <= current_max_dd * 1.05 if current_exists else True
            aggregate_passed = (
                not current_exists
                or wins == self.k
                or (wins == self.k - 1 and drawdown_ok)
            )
            promoted = not safety_errors and aggregate_passed
            reason = "passed"
            if safety_errors:
                reason = "; ".join(safety_errors)
            elif not promoted:
                reason = f"{wins}/{self.k} backtests passed"
            elif not current_exists:
                reason = "bootstrap promotion; no current production model"

            return builder.build(
                status="ok",
                summary={
                    "promoted": promoted,
                    "wins": wins,
                    "required": self.k,
                    "reason": reason,
                    "selected_backtests": [entry.id for entry in selected],
                    "results": [item.model_dump(mode="json") for item in comparisons],
                    "candidate_training_count": candidate_training_count,
                    "candidate_max_drawdown": candidate_max_dd,
                    "candidate_win_rate": candidate_win_rate,
                    "current_max_drawdown": current_max_dd,
                    "drawdown_ok": drawdown_ok,
                    "sharpe_delta": round(sharpe_delta, 6),
                },
            )
        except Exception as exc:
            return builder.build(status="failed", errors=[str(exc)])

    def _select(self, entries: tuple[BacktestPoolEntry, ...]) -> tuple[BacktestPoolEntry, ...]:
        rng = random.Random(self.random_seed)
        size = min(self.k, len(entries))
        return tuple(rng.sample(list(entries), k=size))

    def _safety_errors(
        self,
        *,
        training_count: int,
        max_drawdown: float,
        win_rate: float,
    ) -> list[str]:
        errors = []
        if training_count < self.min_training_count:
            errors.append(
                f"training_count {training_count} below minimum {self.min_training_count}"
            )
        if max_drawdown > self.max_drawdown_limit:
            errors.append(
                f"max_drawdown {max_drawdown:.4f} above limit {self.max_drawdown_limit:.4f}"
            )
        if win_rate < self.min_win_rate:
            errors.append(f"win_rate {win_rate:.4f} below minimum {self.min_win_rate:.4f}")
        return errors

    def _hold_period_error(self) -> str | None:
        if not self.version_registry.exists():
            return None
        try:
            registry = json.loads(self.version_registry.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        current = registry.get("current")
        history = registry.get("history", [])
        current_entry = next(
            (item for item in reversed(history) if item.get("version") == current),
            None,
        )
        if not current_entry:
            return None
        promoted_at = current_entry.get("promoted_at")
        if not promoted_at:
            return None
        try:
            promoted_dt = datetime.fromisoformat(str(promoted_at).replace("Z", "+00:00"))
        except ValueError:
            return None
        if promoted_dt.tzinfo is None:
            promoted_dt = promoted_dt.replace(tzinfo=UTC)
        eligible_at = promoted_dt + timedelta(days=self.min_hold_days)
        if datetime.now(UTC) < eligible_at:
            return f"current model hold period active until {eligible_at.isoformat()}"
        return None


def metadata_backtest_score(model_path: Path, entry: BacktestPoolEntry) -> ModelBacktestScore:
    if not model_path.exists():
        return ModelBacktestScore(
            sharpe=0.0,
            max_drawdown=1.0,
            win_rate=0.0,
            training_count=0,
            metadata={"missing": True},
        )
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    training_count = int(payload.get("training_count") or metrics.get("training_count") or 0)
    win_rate = float(metrics.get("win_rate", metrics.get("directional_accuracy", 0.0)))
    mse = float(metrics.get("mse", 0.05))
    target_mean = float(metrics.get("target_mean", 0.0))
    sharpe = metrics.get("backtest_sharpe")
    if sharpe is None:
        sharpe = _metadata_sharpe(
            model_path=model_path,
            entry=entry,
            training_count=training_count,
            win_rate=win_rate,
            mse=mse,
            target_mean=target_mean,
        )
    max_dd = float(metrics.get("max_drawdown", max(0.02, 0.42 - (win_rate * 0.35) + (mse * 2.0))))
    return ModelBacktestScore(
        sharpe=round(float(sharpe), 6),
        max_drawdown=round(max_dd, 6),
        win_rate=round(win_rate, 6),
        training_count=training_count,
        metadata={"entry_id": entry.id, "source": "model_metadata"},
    )


def _metadata_sharpe(
    *,
    model_path: Path,
    entry: BacktestPoolEntry,
    training_count: int,
    win_rate: float,
    mse: float,
    target_mean: float,
) -> float:
    scale = math.log10(max(training_count, 1)) / 8.0
    digest = hashlib.sha256(model_path.read_bytes() + entry.id.encode()).digest()
    jitter = ((digest[0] / 255.0) - 0.5) * 0.04
    return (win_rate - 0.5) * 4.0 + scale + (target_mean * 3.0) - (mse * 5.0) + jitter


def _mean(values) -> float:
    collected = list(values)
    if not collected:
        return 0.0
    return float(sum(collected) / len(collected))
