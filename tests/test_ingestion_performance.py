from __future__ import annotations

from datetime import date

import pandas as pd

from ai_trader.ingestion import IngestionProfiler, PriceCache, run_named_tasks


def test_price_cache_reuses_covering_range(tmp_path):
    cache = PriceCache(tmp_path)
    frame = pd.DataFrame(
        {
            "date": [date(2026, 5, 1), date(2026, 5, 2), date(2026, 5, 3)],
            "close": [100.0, 101.0, 102.0],
        }
    )

    cache.put("MSFT", date(2026, 5, 1), date(2026, 5, 3), frame)
    cached = cache.get("MSFT", date(2026, 5, 2), date(2026, 5, 3))

    assert cached is not None
    assert cached["close"].tolist() == [101.0, 102.0]


def test_run_named_tasks_profiles_rows():
    profiler = IngestionProfiler()
    results = run_named_tasks(
        {
            "a": lambda: pd.DataFrame({"x": [1, 2]}),
            "b": lambda: [1, 2, 3],
        },
        max_workers=2,
        profiler=profiler,
    )

    assert set(results) == {"a", "b"}
    event_rows = {event.name: event.rows for event in profiler.events}
    assert event_rows["source.a"] == 2
    assert event_rows["source.b"] == 3

