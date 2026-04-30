from datetime import date
from pathlib import Path

import pytest

from ai_trader.config import AppSettings
from ai_trader.domain.signals import Signal, SignalBundle, SignalDirection
from ai_trader.intelligence.reasoner import FinalReasoner
from ai_trader.intelligence.trade_plan import TradePlan
from ai_trader.training import LocalCalibratorModel, LocalCalibratorTrainer, LocalTrainingExample
from ai_trader.training.data import load_training_examples


class StubLLM:
    def __init__(self, output: str) -> None:
        self.output = output

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        return self.output

    def chat(self, messages, *, model=None, temperature=0.2, max_tokens=None) -> str:
        raise NotImplementedError


def _bundle() -> SignalBundle:
    return SignalBundle(
        ticker="MSFT",
        as_of=date(2026, 4, 30),
        signals=(
            Signal(
                name="smart_money.congressional_trade",
                ticker="MSFT",
                direction=SignalDirection.LONG,
                strength=0.9,
                confidence=0.9,
                effective_date=date(2026, 4, 30),
                horizon_days=60,
            ),
        ),
    )


def _plan(*, conviction: float = 0.82, size_multiplier: float = 1.2) -> TradePlan:
    return TradePlan(
        ticker="MSFT",
        as_of=date(2026, 4, 30),
        direction=SignalDirection.LONG,
        conviction=conviction,
        size_multiplier=size_multiplier,
        holding_period_days=30,
        exit_trigger="Exit if the post-disclosure low breaks.",
    )


def test_load_training_examples_reads_jsonl():
    examples = load_training_examples(Path("examples/sample_training_examples.jsonl"))

    assert len(examples) == 2
    assert examples[0].signal_bundle.ticker == "MSFT"
    assert examples[0].pnl_pct == pytest.approx(-0.07)


def test_local_calibrator_saves_loads_and_caps_bad_local_setup(tmp_path: Path):
    example = LocalTrainingExample(signal_bundle=_bundle(), trade_plan=_plan(), pnl_pct=-0.12)
    model = LocalCalibratorTrainer().train((example,))
    path = tmp_path / "local_calibrator.json"

    model.save(path)
    loaded = LocalCalibratorModel.load(path)
    adjusted = loaded.apply(plan=_plan(conviction=0.82, size_multiplier=1.2), bundle=_bundle())

    assert loaded.training_count == 1
    assert loaded.predict_pnl(bundle=_bundle(), plan=_plan()) < 0
    assert adjusted.conviction == pytest.approx(0.2)
    assert adjusted.size_multiplier == pytest.approx(0.25)
    assert any("local calibrator expected pnl" in note for note in adjusted.guardrails)


def test_final_reasoner_applies_local_calibrator_after_guardrails():
    bundle = _bundle()
    model = LocalCalibratorTrainer().train(
        (LocalTrainingExample(signal_bundle=bundle, trade_plan=_plan(), pnl_pct=-0.10),)
    )
    llm = StubLLM(
        output="""
        {
          "ticker": "MSFT",
          "as_of": "2026-04-30",
          "direction": "long",
          "conviction": 0.82,
          "size_multiplier": 1.2,
          "holding_period_days": 30,
          "exit_trigger": "Exit if the post-disclosure low breaks."
        }
        """
    )

    plan = FinalReasoner(llm=llm, calibrator=model).reason(
        ticker="MSFT",
        as_of=date(2026, 4, 30),
        bundle=bundle,
    )

    assert plan.conviction == pytest.approx(0.2)
    assert plan.size_multiplier == pytest.approx(0.25)
    assert any("local calibrator capped conviction" in note for note in plan.guardrails)


def test_final_reasoner_loads_local_calibrator_from_settings(tmp_path: Path):
    bundle = _bundle()
    model = LocalCalibratorTrainer().train(
        (LocalTrainingExample(signal_bundle=bundle, trade_plan=_plan(), pnl_pct=-0.10),)
    )
    model_path = tmp_path / "local_calibrator.json"
    model.save(model_path)
    llm = StubLLM(
        output="""
        {
          "ticker": "MSFT",
          "as_of": "2026-04-30",
          "direction": "long",
          "conviction": 0.82,
          "size_multiplier": 1.2,
          "holding_period_days": 30,
          "exit_trigger": "Exit if the post-disclosure low breaks."
        }
        """
    )

    plan = FinalReasoner(
        llm=llm,
        settings=AppSettings(local_training_enabled=True, local_calibrator_path=model_path),
    ).reason(ticker="MSFT", as_of=date(2026, 4, 30), bundle=bundle)

    assert plan.conviction == pytest.approx(0.2)
    assert any("local calibrator expected pnl" in note for note in plan.guardrails)
