"""Review queue for low-confidence auto-labeled training examples.

Low-confidence examples are appended to ``data/review_queue.jsonl``.
Each line is a JSON object with the full labeled example plus review metadata.

A human reviewer runs ``scripts/review_labels.py`` to step through the queue,
confirm or override each label, and write corrected examples to
``logs/human_labeled_examples.jsonl``.

Queue entry schema (JSON):
    {
        "queue_id": "uuid4",
        "queued_at": "ISO-8601",
        "review_reasons": ["..."],
        "auto_outcome_label": "...",
        "auto_signal_quality": "...",
        "auto_label_confidence": 0.42,
        "reviewed": false,
        "reviewed_at": null,
        "reviewer_notes": null,
        "example": { ...LocalTrainingExample JSON... }
    }
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, ConfigDict, Field

from ai_trader.training.data import LocalTrainingExample, OutcomeLabel, SignalQuality
from ai_trader.training.labeler import LabelResult

DEFAULT_QUEUE_PATH = Path("data/review_queue.jsonl")
DEFAULT_HUMAN_OUT_PATH = Path("logs/human_labeled_examples.jsonl")


class ReviewEntry(BaseModel):
    model_config = ConfigDict(frozen=False)

    queue_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    queued_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    review_reasons: list[str]
    auto_outcome_label: str
    auto_signal_quality: str
    auto_label_confidence: float
    reviewed: bool = False
    reviewed_at: str | None = None
    reviewer_notes: str | None = None
    example: LocalTrainingExample

    def to_jsonl(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_jsonl(cls, line: str) -> ReviewEntry:
        return cls.model_validate_json(line)


def enqueue(
    example: LocalTrainingExample,
    result: LabelResult,
    queue_path: Path = DEFAULT_QUEUE_PATH,
) -> ReviewEntry:
    """Append a low-confidence example to the review queue and return the entry."""
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    entry = ReviewEntry(
        review_reasons=list(result.review_reasons),
        auto_outcome_label=result.outcome_label,
        auto_signal_quality=result.signal_quality,
        auto_label_confidence=result.label_confidence,
        example=example,
    )
    with queue_path.open("a", encoding="utf-8") as fh:
        fh.write(entry.to_jsonl() + "\n")
    return entry


def iter_queue(queue_path: Path = DEFAULT_QUEUE_PATH) -> Iterator[ReviewEntry]:
    """Yield all entries from the review queue, skipping blank lines."""
    if not queue_path.exists():
        return
    with queue_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield ReviewEntry.from_jsonl(line)


def pending_count(queue_path: Path = DEFAULT_QUEUE_PATH) -> int:
    """Return the number of un-reviewed entries in the queue."""
    return sum(1 for e in iter_queue(queue_path) if not e.reviewed)


def rewrite_queue(
    entries: list[ReviewEntry],
    queue_path: Path = DEFAULT_QUEUE_PATH,
) -> None:
    """Atomically overwrite the queue file with the provided entries."""
    tmp = queue_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(entry.to_jsonl() + "\n")
    tmp.replace(queue_path)


def confirm_label(
    entry: ReviewEntry,
    outcome_label: OutcomeLabel,
    signal_quality: SignalQuality,
    notes: str = "",
) -> LocalTrainingExample:
    """Apply human-confirmed labels and return the corrected example."""
    from ai_trader.training.labeler import LabelResult, apply_label

    confirmed = LabelResult(
        outcome_label=outcome_label,
        signal_quality=signal_quality,
        label_confidence=1.0,  # human label → full confidence
        needs_review=False,
        review_reasons=(),
    )
    corrected = apply_label(entry.example, confirmed, source="human")

    entry.reviewed = True
    entry.reviewed_at = datetime.now(timezone.utc).isoformat()
    entry.reviewer_notes = notes or None
    return corrected
