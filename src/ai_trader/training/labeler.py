"""Automatic labeling for LocalTrainingExample instances.

Each example gets three labels applied deterministically from its data:

  outcome_label   — trade outcome tier based on realized pnl_pct
  signal_quality  — quality of the signal bundle that drove the decision
  label_confidence— the labeler's own confidence in its signal_quality assignment
  needs_review    — True when the auto-label is uncertain; human review requested

Outcome thresholds (configurable via module-level constants):
  strong_win  : pnl_pct >=  STRONG_WIN_PCT  (default  10 %)
  win         : pnl_pct >=  WIN_PCT         (default   2 %)
  neutral     : |pnl_pct| < WIN_PCT
  loss        : pnl_pct <= -WIN_PCT
  strong_loss : pnl_pct <= -STRONG_WIN_PCT

Signal quality tiers:
  high   : avg_confidence >= 0.65 AND signal_count >= 3 AND combined_strength >= 0.55
  medium : avg_confidence >= 0.45 OR  signal_count >= 2
  low    : everything else

Needs-review triggers:
  - label_confidence < REVIEW_CONFIDENCE_THRESHOLD
  - Direction mismatch: plan direction contradicts dominant signal direction
  - Extreme outcome (|pnl_pct| > EXTREME_PCT) with fewer than MIN_SIGNALS_FOR_EXTREME signals
  - Conflicting signals (both LONG and SHORT present in the bundle)
"""
from __future__ import annotations

from dataclasses import dataclass

from ai_trader.domain.signals import SignalDirection
from ai_trader.training.data import LocalTrainingExample, LabelSource

# ── Outcome thresholds ────────────────────────────────────────────────────────
STRONG_WIN_PCT: float = 0.10   # +10 %
WIN_PCT: float = 0.02          # +2 %
EXTREME_PCT: float = 0.15      # |±15 %| — surprises the labeler

# ── Signal quality thresholds ─────────────────────────────────────────────────
HIGH_CONFIDENCE_FLOOR: float = 0.65
HIGH_SIGNAL_COUNT: int = 3
HIGH_STRENGTH_FLOOR: float = 0.55
MED_CONFIDENCE_FLOOR: float = 0.45
MED_SIGNAL_COUNT: int = 2

# ── Review gate ───────────────────────────────────────────────────────────────
REVIEW_CONFIDENCE_THRESHOLD: float = 0.50
MIN_SIGNALS_FOR_EXTREME: int = 2


@dataclass(frozen=True)
class LabelResult:
    outcome_label: str          # strong_win | win | neutral | loss | strong_loss
    signal_quality: str         # high | medium | low
    label_confidence: float     # 0–1
    needs_review: bool
    review_reasons: tuple[str, ...]


def _outcome_label(pnl_pct: float) -> str:
    if pnl_pct >= STRONG_WIN_PCT:
        return "strong_win"
    if pnl_pct >= WIN_PCT:
        return "win"
    if pnl_pct <= -STRONG_WIN_PCT:
        return "strong_loss"
    if pnl_pct <= -WIN_PCT:
        return "loss"
    return "neutral"


def _signal_quality(example: LocalTrainingExample) -> tuple[str, float]:
    """Return (quality_tier, labeler_confidence)."""
    signals = example.signal_bundle.signals
    n = len(signals)
    if n == 0:
        return "low", 0.90  # unambiguous: no signals → low quality

    avg_conf = sum(s.confidence for s in signals) / n
    avg_str = sum(s.strength for s in signals) / n
    combined = example.signal_bundle.combined_strength

    if avg_conf >= HIGH_CONFIDENCE_FLOOR and n >= HIGH_SIGNAL_COUNT and combined >= HIGH_STRENGTH_FLOOR:
        # Clear high-quality bundle; confidence in this assignment is high
        margin = min(
            (avg_conf - HIGH_CONFIDENCE_FLOOR) / (1.0 - HIGH_CONFIDENCE_FLOOR),
            (combined - HIGH_STRENGTH_FLOOR) / (1.0 - HIGH_STRENGTH_FLOOR),
        )
        labeler_conf = round(0.75 + 0.25 * margin, 4)
        return "high", labeler_conf

    if avg_conf >= MED_CONFIDENCE_FLOOR or n >= MED_SIGNAL_COUNT:
        # Medium quality — confidence is lower near the boundaries
        dist_from_high = min(
            abs(avg_conf - HIGH_CONFIDENCE_FLOOR),
            abs(n - HIGH_SIGNAL_COUNT) / HIGH_SIGNAL_COUNT,
        )
        labeler_conf = round(max(0.40, 0.65 - dist_from_high * 0.5), 4)
        return "medium", labeler_conf

    # Low quality — but we're fairly confident it's low
    labeler_conf = round(0.55 + 0.15 * (1.0 - avg_conf), 4)
    return "low", min(labeler_conf, 0.80)


def _direction_mismatch(example: LocalTrainingExample) -> bool:
    """True when the trade plan direction contradicts the dominant signal direction."""
    signals = example.signal_bundle.signals
    if not signals:
        return False
    long_strength = sum(s.strength for s in signals if s.direction is SignalDirection.LONG)
    short_strength = sum(s.strength for s in signals if s.direction is SignalDirection.SHORT)
    if long_strength == short_strength:
        return False
    dominant = SignalDirection.LONG if long_strength > short_strength else SignalDirection.SHORT
    return example.trade_plan.direction != dominant


def _has_conflicting_signals(example: LocalTrainingExample) -> bool:
    """True when both LONG and SHORT signals are present with meaningful strength."""
    signals = example.signal_bundle.signals
    long_present = any(s.direction is SignalDirection.LONG and s.strength >= 0.20 for s in signals)
    short_present = any(s.direction is SignalDirection.SHORT and s.strength >= 0.20 for s in signals)
    return long_present and short_present


def auto_label(example: LocalTrainingExample) -> LabelResult:
    """Assign outcome + signal-quality labels and flag examples for human review."""
    outcome = _outcome_label(example.pnl_pct)
    quality, labeler_conf = _signal_quality(example)

    review_reasons: list[str] = []

    if labeler_conf < REVIEW_CONFIDENCE_THRESHOLD:
        review_reasons.append(
            f"label_confidence {labeler_conf:.2f} below threshold {REVIEW_CONFIDENCE_THRESHOLD}"
        )

    if _direction_mismatch(example):
        review_reasons.append("plan direction contradicts dominant signal direction")
        labeler_conf = round(labeler_conf * 0.80, 4)  # penalise confidence

    if _has_conflicting_signals(example):
        review_reasons.append("conflicting LONG and SHORT signals in bundle")
        labeler_conf = round(labeler_conf * 0.85, 4)

    n_signals = len(example.signal_bundle.signals)
    if abs(example.pnl_pct) > EXTREME_PCT and n_signals < MIN_SIGNALS_FOR_EXTREME:
        review_reasons.append(
            f"extreme outcome ({example.pnl_pct:+.1%}) with only {n_signals} signal(s)"
        )

    needs_review = bool(review_reasons)

    return LabelResult(
        outcome_label=outcome,
        signal_quality=quality,
        label_confidence=round(labeler_conf, 4),
        needs_review=needs_review,
        review_reasons=tuple(review_reasons),
    )


def apply_label(
    example: LocalTrainingExample,
    result: LabelResult,
    source: LabelSource = "auto",
) -> LocalTrainingExample:
    """Return a new (frozen) example with labels applied."""
    return example.model_copy(
        update={
            "outcome_label": result.outcome_label,
            "signal_quality": result.signal_quality,
            "label_confidence": result.label_confidence,
            "label_source": source,
            "needs_review": result.needs_review,
        }
    )
