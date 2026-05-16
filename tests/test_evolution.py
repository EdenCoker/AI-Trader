import json
from datetime import date
from pathlib import Path

import pytest

from ai_trader.evolution.backtest_pool import BacktestPool, BacktestPoolEntry
from ai_trader.evolution.discovery import DiscoveryAgent
from ai_trader.evolution.promoter import ModelPromoter
from ai_trader.evolution.promotion_gate import PromotionGate
from ai_trader.evolution.source_implementation import SourceImplementationAgent
from ai_trader.evolution.source_registry import DataSourceRecord, SourceRegistry
from ai_trader.evolution.watchlist_manager import TickerCandidate, WatchlistManager


def _model(path: Path, *, training_count: int, accuracy: float, mse: float = 0.01) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "coefficients": [0.1],
                "intercept": 0.0,
                "feature_mean": [0.0],
                "feature_std": [1.0],
                "training_count": training_count,
                "metrics": {
                    "directional_accuracy": accuracy,
                    "mse": mse,
                    "target_mean": 0.02,
                    "max_drawdown": 0.12,
                },
            }
        ),
        encoding="utf-8",
    )


def _pool(path: Path) -> None:
    BacktestPool(
        entries=(
            BacktestPoolEntry(
                id="bt_001",
                tickers=("MSFT",),
                start=date(2023, 1, 1),
                end=date(2023, 12, 31),
            ),
            BacktestPoolEntry(
                id="bt_002",
                tickers=("NVDA",),
                start=date(2023, 1, 1),
                end=date(2023, 12, 31),
            ),
            BacktestPoolEntry(
                id="bt_003",
                tickers=("AAPL",),
                start=date(2023, 1, 1),
                end=date(2023, 12, 31),
            ),
        )
    ).save(path)


def test_watchlist_manager_enforces_cap_and_exclusions(tmp_path: Path):
    watchlist = tmp_path / "watchlist.txt"
    exclusions = tmp_path / "exclusions.txt"
    watchlist.write_text("MSFT\n", encoding="utf-8")
    exclusions.write_text("TSLA\n", encoding="utf-8")
    manager = WatchlistManager(
        watchlist_path=watchlist,
        exclusion_path=exclusions,
        cold_storage_path=tmp_path / "cold.txt",
        max_size=3,
    )

    change = manager.add_candidates(
        [
            TickerCandidate(ticker="tsla", score=1.0),
            TickerCandidate(ticker="nvda", score=0.9),
            TickerCandidate(ticker="aapl", score=0.8),
            TickerCandidate(ticker="amd", score=0.7),
        ],
        max_adds=5,
    )

    assert change.added == ("NVDA", "AAPL")
    assert "TSLA" in change.skipped
    assert manager.load() == ["MSFT", "NVDA", "AAPL"]


def test_discovery_agent_marks_high_scoring_candidate_pending(tmp_path: Path):
    registry_path = tmp_path / "source_registry.json"
    proposals_dir = tmp_path / "proposals"
    SourceRegistry(
        sources=(
            DataSourceRecord(
                id="candidate_feed",
                type="api",
                url="https://example.com/feed",
                status="candidate",
                lift_score=0.8,
                coverage_score=0.8,
                freshness_score=1.0,
                complexity_score=0.1,
            ),
        )
    ).save(registry_path)

    report = DiscoveryAgent(
        registry_path=registry_path,
        proposals_dir=proposals_dir,
        watchlist_path=tmp_path / "watchlist.txt",
        min_score=0.5,
        run_id="test",
    ).run()

    updated = SourceRegistry.load(registry_path)
    assert report.ok
    assert updated.by_id()["candidate_feed"].status == "pending_approval"
    assert report.summary["proposals"] >= 1
    assert list(proposals_dir.glob("*.json"))


def test_discovery_agent_registers_default_probe_candidates(tmp_path: Path):
    registry_path = tmp_path / "source_registry.json"
    SourceRegistry(sources=()).save(registry_path)

    report = DiscoveryAgent(
        registry_path=registry_path,
        proposals_dir=tmp_path / "proposals",
        watchlist_path=tmp_path / "watchlist.txt",
        min_score=0.9,
        run_id="test",
    ).run()

    updated = SourceRegistry.load(registry_path)
    assert report.ok
    assert "sec_form4_cluster" in updated.by_id()
    assert updated.by_id()["sec_form4_cluster"].category == "insider"


def test_source_implementation_agent_outputs_tasks(tmp_path: Path):
    registry_path = tmp_path / "source_registry.json"
    out_path = tmp_path / "implementation_tasks.json"
    SourceRegistry(
        sources=(
            DataSourceRecord(
                id="sec_form4_cluster",
                type="api",
                url="https://www.sec.gov/edgar/search/",
                auth="SEC_EDGAR_USER_AGENT",
                status="pending_approval",
                lift_score=0.16,
                coverage_score=0.52,
                freshness_score=0.9,
                complexity_score=0.35,
                category="insider",
                profitability_proxy=0.88,
                ingestion_adapter="sec_form4",
            ),
        )
    ).save(registry_path)

    report = SourceImplementationAgent(
        registry_path=registry_path,
        out_path=out_path,
        min_confidence=0.45,
        run_id="test",
    ).run()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert report.ok
    assert payload["tasks"]
    first = payload["tasks"][0]
    assert first["source_id"] == "sec_form4_cluster"
    assert first["ingestion_adapter"] == "sec_form4"


def test_promotion_gate_promotes_better_candidate_and_promoter_rolls_back(tmp_path: Path):
    current = tmp_path / "models" / "production.json"
    candidate = tmp_path / "models" / "candidate.json"
    second = tmp_path / "models" / "candidate2.json"
    pool = tmp_path / "backtest_pool.json"
    registry = tmp_path / "models" / "version_registry.json"
    _model(current, training_count=20_000, accuracy=0.52)
    _model(candidate, training_count=25_000, accuracy=0.62)
    _model(second, training_count=30_000, accuracy=0.64)
    _pool(pool)

    gate = PromotionGate(
        current=current,
        candidate=candidate,
        pool=pool,
        version_registry=registry,
        min_training_count=1_000,
        min_hold_days=0,
        random_seed=7,
        run_id="test",
    ).run()

    assert gate.ok
    assert gate.summary["promoted"] is True
    assert gate.summary["wins"] == 3

    promoter = ModelPromoter(models_dir=tmp_path / "models")
    first = promoter.promote(candidate=candidate, reason="test", gate_result=gate.summary)
    second_entry = promoter.promote(candidate=second, reason="test second")

    assert first["version"] == "v0001"
    assert second_entry["version"] == "v0002"
    assert json.loads(current.read_text(encoding="utf-8"))["training_count"] == 30_000

    rollback = promoter.rollback(to_version="v0001")

    assert rollback["version"] == "v0001"
    assert json.loads(current.read_text(encoding="utf-8"))["training_count"] == 25_000


def test_promotion_gate_rejects_undertrained_candidate(tmp_path: Path):
    current = tmp_path / "models" / "production.json"
    candidate = tmp_path / "models" / "candidate.json"
    pool = tmp_path / "backtest_pool.json"
    _model(current, training_count=20_000, accuracy=0.52)
    _model(candidate, training_count=5, accuracy=0.9)
    _pool(pool)

    gate = PromotionGate(
        current=current,
        candidate=candidate,
        pool=pool,
        version_registry=tmp_path / "missing_registry.json",
        min_training_count=1_000,
        min_hold_days=0,
        random_seed=1,
        run_id="test",
    ).run()

    assert gate.ok
    assert gate.summary["promoted"] is False
    assert "training_count" in gate.summary["reason"]


def test_rollback_unknown_version_raises(tmp_path: Path):
    promoter = ModelPromoter(models_dir=tmp_path / "models")

    with pytest.raises(ValueError):
        promoter.rollback(to_version="v9999")
