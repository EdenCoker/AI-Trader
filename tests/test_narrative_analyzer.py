from datetime import date

import pytest

from ai_trader.config import AppSettings
from ai_trader.intelligence.models import PsychologyStage
from ai_trader.intelligence.narrative import NarrativeAnalyzer


class StubLLM:
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = list(outputs)
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
        if not self._outputs:
            raise RuntimeError("No more stub outputs")
        return self._outputs.pop(0)

    def chat(self, messages, *, model=None, temperature=0.2, max_tokens=None) -> str:
        raise NotImplementedError


def test_narrative_analyzer_runs_three_stage_chain():
    llm = StubLLM(
        outputs=[
            """
            {
              "consensus_view": "Market expected modest upside but was cautious.",
              "key_expectations": ["beat revenue", "raise guidance", "AI narrative holds"],
              "implied_positioning": "Crowd leaning long but not euphoric.",
              "confidence": 0.6
            }
            """,
            """
            {
              "direction": "positive",
              "surprise_score": 0.7,
              "priced_in_fraction": 0.4,
              "novelty": 0.8,
              "what_changed": ["guidance raised", "margin expansion"],
              "pricing_context": "Recent run-up suggests partial pricing."
            }
            """,
            """
            {
              "psychology_stage": "Hope",
              "immediate_reaction": "Chase up on headline and guidance.",
              "follow_through_1w": "Likely grind higher unless macro risk-off hits.",
              "volatility_risk": 0.55,
              "contrarian_risk": 0.35,
              "watch_for": ["post-earnings drift", "analyst downgrades"]
            }
            """,
        ]
    )
    analyzer = NarrativeAnalyzer(llm=llm, settings=AppSettings(llm_model="gpt-test"))
    result = analyzer.analyze(
        ticker="AAPL",
        as_of=date(2026, 4, 30),
        headline="Apple reports earnings",
        body="Earnings were strong and guidance was raised.",
    )

    assert result.calibration.confidence == pytest.approx(0.6)
    assert result.surprise.direction == "positive"
    assert result.behavior.psychology_stage is PsychologyStage.HOPE
    assert len(llm.calls) == 3


def test_narrative_analyzer_extracts_wrapped_json():
    llm = StubLLM(
        outputs=[
            "Sure, here's the JSON:\n"
            "{"
            '"consensus_view":"x",'
            '"key_expectations":["a"],'
            '"implied_positioning":"b",'
            '"confidence":0.5'
            "}\nThanks!",
            "{"
            '"direction":"mixed",'
            '"surprise_score":0.2,'
            '"priced_in_fraction":0.9,'
            '"novelty":0.1,'
            '"what_changed":["none"],'
            '"pricing_context":"c"'
            "}",
            "{"
            '"psychology_stage":"Anxiety",'
            '"immediate_reaction":"chop",'
            '"follow_through_1w":"range",'
            '"volatility_risk":0.9,'
            '"contrarian_risk":0.2,'
            '"watch_for":["liquidity"]'
            "}",
        ]
    )
    analyzer = NarrativeAnalyzer(llm=llm, settings=AppSettings(llm_model="gpt-test"))
    result = analyzer.analyze(
        ticker="MSFT",
        as_of=date(2026, 4, 30),
        headline="x",
        body="y",
    )

    assert result.calibration.consensus_view == "x"
    assert result.behavior.psychology_stage is PsychologyStage.ANXIETY

