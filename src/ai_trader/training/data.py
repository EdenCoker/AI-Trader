from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ai_trader.domain.signals import SignalBundle
from ai_trader.intelligence.models import NarrativeIntelligence
from ai_trader.intelligence.trade_plan import TradePlan

OutcomeLabel = Literal["strong_win", "win", "neutral", "loss", "strong_loss"]
SignalQuality = Literal["high", "medium", "low"]
LabelSource = Literal["auto", "human", "none"]


class LocalTrainingExample(BaseModel):
    model_config = ConfigDict(frozen=True)

    signal_bundle: SignalBundle
    trade_plan: TradePlan
    pnl_pct: float
    narrative: NarrativeIntelligence | None = None
    metadata: dict = Field(default_factory=dict)

    # --- Auto/human labels (added May 2026) ---
    outcome_label: OutcomeLabel | None = None
    signal_quality: SignalQuality | None = None
    label_confidence: float | None = None        # 0–1; labeler's confidence in its own labels
    label_source: LabelSource = "none"
    needs_review: bool = False


def load_training_examples(path: Path) -> tuple[LocalTrainingExample, ...]:
    examples: list[LocalTrainingExample] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                examples.append(LocalTrainingExample.model_validate_json(line))
            except Exception as exc:
                message = f"Invalid training example at {path}:{line_number}: {exc}"
                raise ValueError(message) from exc
    return tuple(examples)
