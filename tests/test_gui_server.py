from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_trader.gui import server


def test_read_artifact_stays_within_workspace(tmp_path: Path):
    artifact = tmp_path / "logs" / "result.json"
    artifact.parent.mkdir()
    artifact.write_bytes(b'{"ok": true}\n')

    payload = server._read_artifact(tmp_path, "logs/result.json")

    assert payload["success"] is True
    assert payload["artifact"]["path"] == "logs/result.json"
    assert payload["content"] == '{"ok": true}\n'


def test_read_artifact_rejects_parent_escape(tmp_path: Path):
    with pytest.raises(ValueError, match="within"):
        server._read_artifact(tmp_path, "../outside.json")


def test_overview_payload_counts_workspace_files(monkeypatch, tmp_path: Path):
    examples = tmp_path / "logs" / "training_examples.jsonl"
    examples.parent.mkdir()
    examples.write_text('{"a":1}\n\n{"b":2}\n', encoding="utf-8")

    monkeypatch.setattr(
        server,
        "get_settings",
        lambda: SimpleNamespace(
            ingestion_profile_path=Path("logs/ingestion_profile.json"),
            ingestion_cache_dir=Path("data/cache"),
            ingestion_source_workers=3,
            ingestion_price_workers=4,
            ingestion_ticker_workers=5,
            ingestion_http_connections=6,
            ingestion_write_buffer=7,
            provider_status=lambda: {"polygon": True, "reddit": False},
        ),
    )
    monkeypatch.setattr(
        server,
        "detect_hardware",
        lambda probe_network=False: SimpleNamespace(
            physical_cores=2,
            logical_cores=4,
            ram_gb=16.0,
            ram_available_gb=10.0,
            network_mbps=None,
            network_tier="unknown",
        ),
    )

    payload = server._overview_payload(tmp_path)

    assert payload["metrics"]["training_examples"] == 2
    assert payload["providers"] == {"polygon": True, "reddit": False}
    assert payload["hardware"]["source_workers"] == 3
