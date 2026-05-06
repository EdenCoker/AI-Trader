from datetime import date

import pytest

from ai_trader.domain.signals import Signal, SignalBundle, SignalDirection
from ai_trader.config import AppSettings
from ai_trader.intelligence.reasoner import FinalReasoner
from ai_trader.llm.errors import LLMError
from ai_trader.rag.index import Chunk, RetrievedChunk


class StubLLM:
    def __init__(self, output: str) -> None:
        self.output = output
        self.calls: list[str] = []

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append(prompt)
        return self.output

    def chat(self, messages, *, model=None, temperature=0.2, max_tokens=None) -> str:
        raise NotImplementedError


def test_final_reasoner_parses_trade_plan():
    bundle = SignalBundle(
        ticker="MSFT",
        as_of=date(2026, 4, 30),
        signals=(
            Signal(
                name="smart_money.congressional_trade",
                ticker="MSFT",
                direction=SignalDirection.LONG,
                strength=0.7,
                confidence=0.6,
                effective_date=date(2026, 4, 30),
                reasons=("x",),
            ),
        ),
    )

    llm = StubLLM(
        output="""
        {
          "ticker": "MSFT",
          "as_of": "2026-04-30",
          "direction": "long",
          "conviction": 0.62,
          "size_multiplier": 0.8,
          "holding_period_days": 45,
          "exit_trigger": "If price closes below the post-event low on above-average volume.",
          "thesis": ["smart-money aligned", "narrative supports follow-through"]
        }
        """
    )

    plan = FinalReasoner(llm=llm, settings=AppSettings(local_training_enabled=False)).reason(ticker="MSFT", as_of=date(2026, 4, 30), bundle=bundle)

    assert plan.direction is SignalDirection.LONG
    assert plan.conviction == pytest.approx(0.62)
    assert plan.size_multiplier == pytest.approx(0.8)
    assert len(llm.calls) == 1


def test_final_reasoner_extracts_wrapped_json():
    bundle = SignalBundle(ticker="AAPL", as_of=date(2026, 4, 30), signals=())
    llm = StubLLM(
        output="Here you go:\n"
        "{"
        '"ticker":"AAPL",'
        '"as_of":"2026-04-30",'
        '"direction":"neutral",'
        '"conviction":0.1,'
        '"size_multiplier":0.0,'
        '"holding_period_days":30,'
        '"exit_trigger":"No trade unless new info arrives.",'
        '"thesis":["insufficient edge"]'
        "}"
    )
    plan = FinalReasoner(llm=llm, settings=AppSettings(local_training_enabled=False)).reason(ticker="AAPL", as_of=date(2026, 4, 30), bundle=bundle)
    assert plan.direction is SignalDirection.NEUTRAL


def test_final_reasoner_raises_on_non_json():
    bundle = SignalBundle(ticker="AAPL", as_of=date(2026, 4, 30), signals=())
    llm = StubLLM(output="not json")
    with pytest.raises(LLMError):
        FinalReasoner(llm=llm).reason(ticker="AAPL", as_of=date(2026, 4, 30), bundle=bundle)


def test_final_reasoner_includes_rag_analogies_in_prompt():
    bundle = SignalBundle(ticker="MSFT", as_of=date(2026, 4, 30), signals=())

    class StubRAG:
        def retrieve(self, query: str, *, k: int = 3):
            return (
                RetrievedChunk(
                    score=0.91,
                    chunk=Chunk(
                        chunk_id="doc#0",
                        text="When signals disagree, reduce size and demand better odds.",
                        metadata={"source": "macro_trend_notes.txt"},
                    ),
                ),
            )

    llm = StubLLM(
        output="""
        {
          "ticker": "MSFT",
          "as_of": "2026-04-30",
          "direction": "neutral",
          "conviction": 0.1,
          "size_multiplier": 0.0,
          "holding_period_days": 30,
          "exit_trigger": "No trade unless new info arrives."
        }
        """
    )

    settings = AppSettings(rag_enabled=True, llm_model="gpt-test")
    FinalReasoner(llm=llm, settings=settings, rag=StubRAG()).reason(
        ticker="MSFT", as_of=date(2026, 4, 30), bundle=bundle
    )
    assert "Retrieved Analogies" in llm.calls[0]
    assert "macro_trend_notes.txt" in llm.calls[0]


def test_final_reasoner_guardrails_neutralize_no_evidence_trade():
    bundle = SignalBundle(ticker="AAPL", as_of=date(2026, 4, 30), signals=())
    llm = StubLLM(
        output="""
        {
          "ticker": "AAPL",
          "as_of": "2026-04-30",
          "direction": "long",
          "conviction": 0.95,
          "size_multiplier": 2.0,
          "holding_period_days": 60,
          "exit_trigger": "No clear invalidation."
        }
        """
    )

    plan = FinalReasoner(llm=llm).reason(ticker="AAPL", as_of=date(2026, 4, 30), bundle=bundle)

    assert plan.direction is SignalDirection.NEUTRAL
    assert plan.conviction == pytest.approx(0.15)
    assert plan.size_multiplier == 0.0
    assert any("no signal" in note for note in plan.guardrails)


def test_final_reasoner_guardrails_neutralize_bundle_contradiction():
    bundle = SignalBundle(
        ticker="MSFT",
        as_of=date(2026, 4, 30),
        signals=(
            Signal(
                name="smart_money.13f_change",
                ticker="MSFT",
                direction=SignalDirection.LONG,
                strength=0.8,
                confidence=0.8,
                effective_date=date(2026, 4, 30),
            ),
        ),
    )
    llm = StubLLM(
        output="""
        {
          "ticker": "MSFT",
          "as_of": "2026-04-30",
          "direction": "short",
          "conviction": 0.9,
          "size_multiplier": 2.0,
          "holding_period_days": 20,
          "exit_trigger": "Contradictory call."
        }
        """
    )

    plan = FinalReasoner(llm=llm).reason(ticker="MSFT", as_of=date(2026, 4, 30), bundle=bundle)

    assert plan.direction is SignalDirection.NEUTRAL
    assert plan.conviction <= 0.25
    assert plan.size_multiplier == 0.0
    assert any("contradicted" in note for note in plan.guardrails)
