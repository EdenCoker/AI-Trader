from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ai_trader.domain.signals import SignalBundle, SignalDirection
from ai_trader.intelligence.models import NarrativeIntelligence
from ai_trader.intelligence.trade_plan import HorizonClass, TradePlan
from ai_trader.training.features import FEATURE_NAMES, extract_features

if TYPE_CHECKING:
    from ai_trader.training.data import LocalTrainingExample


class LocalCalibratorModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    feature_names: tuple[str, ...] = FEATURE_NAMES
    coefficients: tuple[float, ...]
    intercept: float
    feature_mean: tuple[float, ...]
    feature_std: tuple[float, ...]
    training_count: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metrics: dict[str, float] = Field(default_factory=dict)

    def predict_pnl(
        self,
        *,
        bundle: SignalBundle,
        plan: TradePlan,
        narrative: NarrativeIntelligence | None = None,
    ) -> float:
        features = _features_for_model(
            self.feature_names,
            extract_features(bundle=bundle, plan=plan, narrative=narrative),
            expected_length=len(self.feature_mean),
        )
        normalized = (features - np.asarray(self.feature_mean)) / np.asarray(self.feature_std)
        return float(np.dot(normalized, np.asarray(self.coefficients)) + self.intercept)

    def apply(
        self,
        *,
        plan: TradePlan,
        bundle: SignalBundle,
        narrative: NarrativeIntelligence | None = None,
    ) -> TradePlan:
        predicted_pnl = self.predict_pnl(bundle=bundle, plan=plan, narrative=narrative)
        notes = list(plan.guardrails)
        notes.append(f"local calibrator expected pnl {predicted_pnl:.2%}")

        direction = plan.direction
        conviction = plan.conviction
        size_multiplier = plan.size_multiplier

        if direction is SignalDirection.NEUTRAL:
            return plan.model_copy(update={"guardrails": tuple(notes)})

        max_conviction = _max_conviction_from_expected_pnl(predicted_pnl)
        if conviction > max_conviction:
            notes.append(
                f"local calibrator capped conviction from {conviction:.2f} "
                f"to {max_conviction:.2f}"
            )
            conviction = max_conviction

        max_size = min(2.0, conviction * 2.0)
        if predicted_pnl <= 0:
            max_size = min(max_size, 0.25)
        if size_multiplier > max_size:
            notes.append(
                f"local calibrator capped size from {size_multiplier:.2f} to {max_size:.2f}"
            )
            size_multiplier = max_size

        return plan.model_copy(
            update={
                "conviction": round(float(conviction), 6),
                "size_multiplier": round(float(size_multiplier), 6),
                "guardrails": tuple(notes),
            }
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> LocalCalibratorModel:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class LocalCalibratorTrainer:
    def __init__(self, *, l2: float = 1e-3, target_clip: float = 0.25) -> None:
        self._l2 = l2
        self._target_clip = target_clip

    def train(self, examples: tuple[LocalTrainingExample, ...]) -> LocalCalibratorModel:
        if not examples:
            raise ValueError("At least one local training example is required")

        x = np.vstack(
            [
                extract_features(
                    bundle=example.signal_bundle,
                    plan=example.trade_plan,
                    narrative=example.narrative,
                )
                for example in examples
            ]
        )
        y = np.asarray(
            [
                float(np.clip(example.pnl_pct, -self._target_clip, self._target_clip))
                for example in examples
            ],
            dtype=float,
        )

        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        std[std == 0] = 1.0
        x_norm = (x - mean) / std
        design = np.column_stack([np.ones(x_norm.shape[0]), x_norm])
        penalty = np.eye(design.shape[1]) * self._l2
        penalty[0, 0] = 0.0
        weights = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
        predictions = design @ weights

        return LocalCalibratorModel(
            coefficients=tuple(float(value) for value in weights[1:]),
            intercept=float(weights[0]),
            feature_mean=tuple(float(value) for value in mean),
            feature_std=tuple(float(value) for value in std),
            training_count=len(examples),
            metrics={
                "mse": float(np.mean((predictions - y) ** 2)),
                "directional_accuracy": _directional_accuracy(predictions, y),
                "target_mean": float(np.mean(y)),
            },
        )


def filter_examples_by_horizon(
    examples: tuple[LocalTrainingExample, ...],
    horizon: HorizonClass,
) -> tuple[LocalTrainingExample, ...]:
    return tuple(
        example
        for example in examples
        if example.trade_plan.horizon_class == horizon
    )


def _features_for_model(
    feature_names: tuple[str, ...],
    current_features: np.ndarray,
    *,
    expected_length: int,
) -> np.ndarray:
    if feature_names == FEATURE_NAMES and expected_length == len(current_features):
        return current_features
    by_name = dict(zip(FEATURE_NAMES, current_features, strict=False))
    features = np.asarray([by_name.get(name, 0.0) for name in feature_names], dtype=float)
    if len(features) == expected_length:
        return features
    if expected_length <= len(current_features):
        return current_features[:expected_length]
    padded = np.zeros(expected_length, dtype=float)
    padded[: len(current_features)] = current_features
    return padded


def _max_conviction_from_expected_pnl(expected_pnl: float) -> float:
    if expected_pnl <= 0:
        return 0.20
    if expected_pnl >= 0.10:
        return 0.95
    return 0.20 + 0.75 * (expected_pnl / 0.10)


def _directional_accuracy(predictions: np.ndarray, actuals: np.ndarray) -> float:
    if predictions.size == 0:
        return 0.0
    return float(np.mean(np.sign(predictions) == np.sign(actuals)))
