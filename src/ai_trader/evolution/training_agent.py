from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ai_trader.evolution.reports import AgentReport, ReportBuilder
from ai_trader.training import LocalCalibratorTrainer, load_training_examples


class TrainingAgent:
    """Trains a shadow candidate LocalCalibratorModel from the rolling window."""

    def __init__(
        self,
        *,
        examples: Path = Path("logs/rolling_examples.jsonl"),
        out: Path = Path("data/models/candidate.json"),
        run_id: str = "manual",
    ) -> None:
        self.examples = examples
        self.out = out
        self.run_id = run_id

    def run(self) -> AgentReport:
        builder = ReportBuilder("TrainingAgent", self.run_id)
        try:
            if not self.examples.exists():
                return builder.build(status="failed", errors=[f"{self.examples} does not exist"])
            examples = load_training_examples(self.examples)
            model = LocalCalibratorTrainer().train(examples)
            model.save(self.out)
            metadata = {
                "created_at": datetime.now(UTC).isoformat(),
                "examples_file": str(self.examples),
                "model_out": str(self.out),
                "training_count": model.training_count,
                "metrics": model.metrics,
            }
            self.out.with_suffix(".metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            return builder.build(
                status="ok",
                summary={
                    "training_count": model.training_count,
                    "model_out": str(self.out),
                    "metrics": model.metrics,
                },
            )
        except Exception as exc:
            return builder.build(status="failed", errors=[str(exc)])
