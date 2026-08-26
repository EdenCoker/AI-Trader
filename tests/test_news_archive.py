"""Archive durability: torn tails, malformed rows, prune tolerance."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from ai_trader.news.archive import NewsArchive
from ai_trader.news.models import ObservedArticle

T0 = datetime(2026, 8, 25, 12, tzinfo=UTC)


def article(i: int, ts: datetime = T0) -> ObservedArticle:
    return ObservedArticle(
        article_id=f"a{i}",
        title=f"story {i}",
        link=f"http://x/{i}",
        source_name="Reuters",
        published_at=ts,
        first_seen_at=ts,
        last_seen_at=ts,
    )


def test_torn_tail_is_self_healed(tmp_path: Path):
    path = tmp_path / "arch.jsonl"
    archive = NewsArchive(path)
    archive.append([article(1)], T0)
    # Simulate a process killed mid-write: strip the trailing newline and
    # half the last row.
    content = path.read_text()
    path.write_text(content.rstrip("\n")[: len(content) // 2])
    archive.append([article(2)], T0 + timedelta(hours=1))
    visible = archive.load_visible(T0 + timedelta(hours=2))
    # The torn row is lost (skipped as malformed), the new row survives.
    assert [a.article_id for a in visible] == ["a2"]


def test_malformed_lines_do_not_poison_reads(tmp_path: Path):
    path = tmp_path / "arch.jsonl"
    archive = NewsArchive(path)
    archive.append([article(1)], T0)
    with path.open("a") as handle:
        handle.write("null\n")          # valid JSON, not an object
        handle.write('"a string"\n')
        handle.write("42\n")
        handle.write("[1,2,3]\n")
        handle.write("{not json\n")
        handle.write('{"_fetched_at": null}\n')
    visible = archive.load_visible(T0 + timedelta(hours=1))
    assert [a.article_id for a in visible] == ["a1"]


def test_prune_tolerates_malformed_and_keeps_recent(tmp_path: Path):
    path = tmp_path / "arch.jsonl"
    archive = NewsArchive(path)
    archive.append([article(1, T0 - timedelta(days=60))], T0 - timedelta(days=60))
    archive.append([article(2)], T0)
    with path.open("a") as handle:
        handle.write("[1,2,3]\n")
        handle.write('{"_fetched_at": null}\n')
    kept = archive.prune(keep_days=30, now=T0 + timedelta(hours=1))
    assert kept == 1
    visible = archive.load_visible(T0 + timedelta(hours=1))
    assert [a.article_id for a in visible] == ["a2"]


def test_naive_timestamps_are_treated_as_utc(tmp_path: Path):
    archive = NewsArchive(tmp_path / "arch.jsonl")
    archive.append([article(1)], T0.replace(tzinfo=None))
    visible = archive.load_visible(T0.replace(tzinfo=None) + timedelta(hours=1))
    assert len(visible) == 1
    assert visible[0].first_seen_at == T0
