from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class ProfileEvent:
    name: str
    seconds: float
    rows: int | None = None
    status: str = "ok"
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "seconds": round(self.seconds, 4),
            "status": self.status,
        }
        if self.rows is not None:
            payload["rows"] = self.rows
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


class IngestionProfiler:
    """Thread-safe stage timing for ingestion runs."""

    def __init__(self, *, enabled: bool = True, run_id: str | None = None) -> None:
        self.enabled = enabled
        self.run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.started_at = datetime.now(UTC)
        self._events: list[ProfileEvent] = []
        self._lock = threading.Lock()

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        rows: Callable[[], int | None] | int | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        start = perf_counter()
        status = "ok"
        try:
            yield
        except Exception:
            status = "failed"
            raise
        finally:
            row_count = rows() if callable(rows) else rows
            self.record(
                name,
                seconds=perf_counter() - start,
                rows=row_count,
                status=status,
                metadata=metadata or {},
            )

    def record(
        self,
        name: str,
        *,
        seconds: float,
        rows: int | None = None,
        status: str = "ok",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        event = ProfileEvent(
            name=name,
            seconds=seconds,
            rows=rows,
            status=status,
            metadata=metadata or {},
        )
        with self._lock:
            self._events.append(event)

    def as_dict(self) -> dict[str, Any]:
        ended_at = datetime.now(UTC)
        events = [event.as_dict() for event in self.events]
        slowest = sorted(events, key=lambda event: event["seconds"], reverse=True)[:15]
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "total_seconds": round((ended_at - self.started_at).total_seconds(), 4),
            "events": events,
            "slowest": slowest,
        }

    @property
    def events(self) -> tuple[ProfileEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def write(self, path: Path) -> None:
        if not self.enabled:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True), encoding="utf-8")


def run_named_tasks(
    tasks: Mapping[str, Callable[[], T]],
    *,
    max_workers: int,
    profiler: IngestionProfiler | None = None,
    log: Any | None = None,
) -> dict[str, T]:
    """Run named blocking tasks concurrently and preserve exception semantics."""

    if not tasks:
        return {}
    workers = max(1, min(max_workers, len(tasks)))
    results: dict[str, T] = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ingest-source") as executor:
        future_to_name = {}
        for name, task in tasks.items():
            start = perf_counter()

            def _runner(task=task, start=start, name=name):
                value = task()
                if profiler is not None:
                    profiler.record(
                        f"source.{name}",
                        seconds=perf_counter() - start,
                        rows=_row_count(value),
                    )
                return value

            future_to_name[executor.submit(_runner)] = name

        for future in as_completed(future_to_name):
            name = future_to_name[future]
            if log is not None:
                log.info("Finished source loader %s", name)
            results[name] = future.result()
    return results


def _row_count(value: Any) -> int | None:
    if hasattr(value, "shape"):
        try:
            return int(value.shape[0])
        except (TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    return None

