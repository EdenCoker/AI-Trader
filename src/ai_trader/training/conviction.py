from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from ai_trader.domain.signals import SignalBundle, SignalDirection
from ai_trader.training.data import LocalTrainingExample


class ConvictionMetric(StrEnum):
    PLAN = "plan"
    BUNDLE = "bundle"
    AGREEMENT_ADJUSTED = "agreement_adjusted"


@dataclass(frozen=True)
class ConvictionEvidence:
    support_weight: float
    opposing_weight: float
    support_count: int
    opposing_count: int
    unique_support_signals: int
    max_support_share: float
    agreement: float
    breadth: float
    concentration_penalty: float


def score_training_example(
    example: LocalTrainingExample,
    metric: ConvictionMetric | str = ConvictionMetric.PLAN,
) -> float:
    metric = normalize_conviction_metric(metric)
    if metric is ConvictionMetric.PLAN:
        return float(example.trade_plan.conviction)
    if metric is ConvictionMetric.BUNDLE:
        return float(example.signal_bundle.conviction)
    return agreement_adjusted_conviction(
        example.signal_bundle,
        direction=example.trade_plan.direction,
    )


def normalize_conviction_metric(metric: ConvictionMetric | str) -> ConvictionMetric:
    if isinstance(metric, ConvictionMetric):
        return metric
    return ConvictionMetric(str(metric))


def agreement_adjusted_conviction(
    bundle: SignalBundle,
    *,
    direction: SignalDirection | None = None,
) -> float:
    evidence = conviction_evidence(bundle, direction=direction)
    if evidence.support_count == 0:
        return 0.0

    avg_support = evidence.support_weight / evidence.support_count
    score = (
        evidence.agreement
        * math.sqrt(max(0.0, min(1.0, avg_support)))
        * (0.55 + 0.45 * evidence.breadth)
        * evidence.concentration_penalty
    )
    return round(max(0.0, min(1.0, score)), 6)


def conviction_evidence(
    bundle: SignalBundle,
    *,
    direction: SignalDirection | None = None,
) -> ConvictionEvidence:
    direction = direction or bundle.direction
    if direction is SignalDirection.NEUTRAL:
        return _empty_evidence()

    support_weight = 0.0
    opposing_weight = 0.0
    support_count = 0
    opposing_count = 0
    unique_support_signals: set[str] = set()
    support_weights: list[float] = []

    for signal in bundle.signals:
        if signal.direction is SignalDirection.NEUTRAL:
            continue
        weight = float(signal.strength) * float(signal.confidence)
        if signal.direction is direction:
            support_weight += weight
            support_count += 1
            support_weights.append(weight)
            unique_support_signals.add(signal.name)
        else:
            opposing_weight += weight
            opposing_count += 1

    total_directional_weight = support_weight + opposing_weight
    if support_count == 0 or total_directional_weight <= 0:
        return _empty_evidence(opposing_weight=opposing_weight, opposing_count=opposing_count)

    agreement = max(0.0, (support_weight - opposing_weight) / total_directional_weight)
    breadth = min(1.0, math.sqrt(len(unique_support_signals) / 3.0))
    max_support_share = max(support_weights) / support_weight if support_weight > 0 else 0.0
    concentration_penalty = _concentration_penalty(max_support_share)

    return ConvictionEvidence(
        support_weight=support_weight,
        opposing_weight=opposing_weight,
        support_count=support_count,
        opposing_count=opposing_count,
        unique_support_signals=len(unique_support_signals),
        max_support_share=max_support_share,
        agreement=agreement,
        breadth=breadth,
        concentration_penalty=concentration_penalty,
    )


def _empty_evidence(
    *,
    opposing_weight: float = 0.0,
    opposing_count: int = 0,
) -> ConvictionEvidence:
    return ConvictionEvidence(
        support_weight=0.0,
        opposing_weight=opposing_weight,
        support_count=0,
        opposing_count=opposing_count,
        unique_support_signals=0,
        max_support_share=0.0,
        agreement=0.0,
        breadth=0.0,
        concentration_penalty=0.0,
    )


def _concentration_penalty(max_support_share: float) -> float:
    if max_support_share <= 0.55:
        return 1.0
    # A single dominant signal can still count, but cannot carry "top conviction" alone.
    return max(0.35, 1.0 - ((max_support_share - 0.55) / 0.45))
