from __future__ import annotations

import pytest

from ai_trader.config import AppSettings
from ai_trader.ingestion import hardware


def _fixed_machine(
    monkeypatch: pytest.MonkeyPatch,
    *,
    logical: int = 8,
    ram_available: float = 16.0,
) -> None:
    monkeypatch.setattr(hardware, "_physical_cores", lambda: max(1, logical // 2))
    monkeypatch.setattr(hardware, "_logical_cores", lambda: logical)
    monkeypatch.setattr(hardware, "_ram_gb", lambda: (32.0, ram_available))


def test_hardware_detect_uses_configured_network_tier(monkeypatch, tmp_path):
    _fixed_machine(monkeypatch, logical=8, ram_available=16.0)
    monkeypatch.setenv("AI_TRADER_INGESTION_NETWORK_MBPS", "250")

    profile = hardware.detect(probe_network=False, cache_path=tmp_path / "hardware.json")

    assert profile.network_mbps == 250
    assert profile.network_tier == "fast"
    assert profile.source_workers == 48
    assert profile.price_workers == 36
    assert profile.ticker_workers == 24


def test_hardware_detect_throttles_on_slow_network(monkeypatch, tmp_path):
    _fixed_machine(monkeypatch, logical=8, ram_available=16.0)
    monkeypatch.setenv("AI_TRADER_INGESTION_NETWORK_MBPS", "10")

    profile = hardware.detect(probe_network=False, cache_path=tmp_path / "hardware.json")

    assert profile.network_tier == "slow"
    assert profile.source_workers == 12
    assert profile.price_workers == 8
    assert profile.ticker_workers == 6


def test_hardware_detect_caps_workers_by_available_ram(monkeypatch, tmp_path):
    _fixed_machine(monkeypatch, logical=32, ram_available=0.25)
    monkeypatch.setenv("AI_TRADER_INGESTION_NETWORK_MBPS", "1000")

    profile = hardware.detect(probe_network=False, cache_path=tmp_path / "hardware.json")

    assert profile.network_tier == "very_fast"
    assert profile.source_workers == 4
    assert profile.price_workers == 4
    assert profile.ticker_workers == 4


def test_hardware_detect_reuses_cached_network_probe(monkeypatch, tmp_path):
    _fixed_machine(monkeypatch)
    cache_path = tmp_path / "hardware.json"
    monkeypatch.setattr(hardware, "_probe_network_mbps", lambda: 80.0)

    first = hardware.detect(probe_network=True, cache_path=cache_path)

    def _fail_probe() -> float:
        raise AssertionError("network probe should not run when cache is fresh")

    monkeypatch.setattr(hardware, "_probe_network_mbps", _fail_probe)
    second = hardware.detect(probe_network=True, cache_path=cache_path)

    assert first.network_mbps == 80.0
    assert second.network_mbps == 80.0
    assert second.network_tier == "normal"


def test_ingestion_worker_env_overrides_do_not_probe_hardware(monkeypatch):
    monkeypatch.setenv("AI_TRADER_INGESTION_SOURCE_WORKERS", "3")
    monkeypatch.setenv("AI_TRADER_INGESTION_PRICE_WORKERS", "4")
    monkeypatch.setenv("AI_TRADER_INGESTION_TICKER_WORKERS", "5")
    monkeypatch.setenv("AI_TRADER_INGESTION_HTTP_CONNECTIONS", "6")
    monkeypatch.setenv("AI_TRADER_INGESTION_WRITE_BUFFER", "7")

    def _fail_profile():
        raise AssertionError("hardware probe should not run for explicit env overrides")

    monkeypatch.setattr("ai_trader.config._hw_profile", _fail_profile)

    settings = AppSettings()

    assert settings.ingestion_source_workers == 3
    assert settings.ingestion_price_workers == 4
    assert settings.ingestion_ticker_workers == 5
    assert settings.ingestion_http_connections == 6
    assert settings.ingestion_write_buffer == 7
