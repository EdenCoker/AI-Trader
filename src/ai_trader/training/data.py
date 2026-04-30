from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ai_trader.domain.signals import SignalBundle
from ai_trader.intelligence.models import NarrativeIntelligence
from ai_trader.intelligence.trade_plan import TradePlan


class LocalTrainingExample(BaseModel):
    model_config = ConfigDict(frozen=True)

    signal_bundle: SignalBundle
    trade_plan: TradePlan
    pnl_pct: float
    narrative: NarrativeIntelligence | None = None
    metadata: dict = Field(default_factory=dict)


def load_training_examples(path: Path) -> tuple[LocalTrainingExample, ...]:
    examples: list[LocalTrainingExample] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            examples.append(LocalTrainingExample.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"Invalid training example at {path}:{line_number}: {exc}") from exc
    return tuple(examples)

