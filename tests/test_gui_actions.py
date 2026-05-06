import sys

import pytest

from ai_trader.gui.actions import action_specs, build_command


def test_gui_actions_expose_console_tools():
    ids = {action["id"] for action in action_specs()}

    assert {
        "status",
        "analyze_news",
        "reason",
        "train_local",
        "backtest_run",
        "trade",
    }.issubset(ids)


def test_build_reason_command_with_rag_flag():
    command = build_command(
        "reason",
        {
            "bundle_file": "examples/sample_signal_bundle.json",
            "rag": True,
            "output": "logs/trade_plan.json",
        },
    )

    assert command[:3] == (sys.executable, "-m", "ai_trader.cli")
    assert command[3] == "reason"
    assert "--bundle-file" in command
    assert "examples/sample_signal_bundle.json" in command
    assert "--rag" in command
    assert "--output" in command


def test_build_backtest_command_infers_tickers_from_events_file():
    command = build_command(
        "backtest_run",
        {"start": "2022-01-01", "end": "2022-12-31"},
    )

    assert "--tickers" not in command
    assert "--events-file" in command
    assert "examples/sample_events.jsonl" in command


def test_build_loop_command_uses_events_file_for_ticker_discovery():
    command = build_command("build_loop", {})

    assert "--tickers" not in command
    assert "--events-file" in command
    assert "examples/sample_events.jsonl" in command


def test_build_command_rejects_missing_required_field():
    with pytest.raises(ValueError, match="Signal Bundle"):
        build_command("reason", {"bundle_file": ""})


def test_trade_command_defaults_to_dry_run():
    command = build_command("trade", {"plan_file": "logs/trade_plan.json"})

    assert "--dry-run" in command
    assert "--shares" not in command
    assert "--cash-fraction" in command
