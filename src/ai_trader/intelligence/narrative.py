from __future__ import annotations

import json
from datetime import date
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from ai_trader.config import AppSettings, get_settings
from ai_trader.llm import get_llm_client
from ai_trader.llm.contracts import LLMClient
from ai_trader.llm.errors import LLMError
from ai_trader.intelligence.models import (
    BehaviorPrediction,
    ExpectationCalibration,
    NarrativeIntelligence,
    SurpriseAssessment,
)


SYSTEM_PROMPT = (
    "You are a disciplined market-narrative analyst. "
    "You produce structured, testable outputs. "
    "When information is missing, state uncertainty explicitly and avoid hallucinating numbers. "
    "Return only the requested JSON."
)


class NarrativeAnalyzer:
    """Phase 2: News & Narrative Intelligence (3-stage chain)."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        *,
        settings: AppSettings | None = None,
        model: str | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm = llm or get_llm_client(self._settings)
        self._model = model or self._settings.llm_model

    def analyze(
        self,
        *,
        ticker: str,
        as_of: date,
        headline: str,
        body: str,
        market_context: str | None = None,
        analyst_context: str | None = None,
    ) -> NarrativeIntelligence:
        calibration = self._calibrate(
            ticker=ticker,
            as_of=as_of,
            headline=headline,
            body=body,
            market_context=market_context,
            analyst_context=analyst_context,
        )
        surprise = self._surprise(
            ticker=ticker,
            as_of=as_of,
            headline=headline,
            body=body,
            calibration=calibration,
            market_context=market_context,
        )
        behavior = self._behavior(
            ticker=ticker,
            as_of=as_of,
            headline=headline,
            body=body,
            calibration=calibration,
            surprise=surprise,
            market_context=market_context,
        )
        return NarrativeIntelligence(calibration=calibration, surprise=surprise, behavior=behavior)

    def _calibrate(
        self,
        *,
        ticker: str,
        as_of: date,
        headline: str,
        body: str,
        market_context: str | None,
        analyst_context: str | None,
    ) -> ExpectationCalibration:
        parser = PydanticOutputParser(pydantic_object=ExpectationCalibration)
        prompt = PromptTemplate(
            template=(
                "TASK: Calibrate this news vs what the market/analysts likely expected.\n"
                "Ticker: {ticker}\n"
                "As-of date: {as_of}\n"
                "Headline: {headline}\n"
                "Body: {body}\n"
                "Market context (optional): {market_context}\n"
                "Analyst expectations context (optional): {analyst_context}\n\n"
                "Produce:\n"
                "- consensus_view: 1-3 sentences\n"
                "- key_expectations: 3-8 bullets (short strings)\n"
                "- implied_positioning: how positioned/primed the crowd likely was\n"
                "- confidence: 0-1\n\n"
                "{format_instructions}"
            ),
            input_variables=[
                "ticker",
                "as_of",
                "headline",
                "body",
                "market_context",
                "analyst_context",
            ],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        text = self._invoke(
            prompt.format(
                ticker=ticker,
                as_of=as_of.isoformat(),
                headline=headline,
                body=body,
                market_context=market_context or "",
                analyst_context=analyst_context or "",
            )
        )
        return _parse(parser, text, "calibration")

    def _surprise(
        self,
        *,
        ticker: str,
        as_of: date,
        headline: str,
        body: str,
        calibration: ExpectationCalibration,
        market_context: str | None,
    ) -> SurpriseAssessment:
        parser = PydanticOutputParser(pydantic_object=SurpriseAssessment)
        prompt = PromptTemplate(
            template=(
                "TASK: Score the 'surprise' and whether it was already priced in.\n"
                "Ticker: {ticker}\n"
                "As-of date: {as_of}\n"
                "Headline: {headline}\n"
                "Body: {body}\n"
                "Market context (optional): {market_context}\n\n"
                "Calibration JSON:\n{calibration_json}\n\n"
                "Rules:\n"
                "- priced_in_fraction: 0 means not priced in; 1 means fully priced in.\n"
                "- surprise_score: magnitude of informational surprise 0-1.\n"
                "- novelty: 0 means rehash; 1 means genuinely new.\n"
                "- direction: 'positive', 'negative', or 'mixed'.\n\n"
                "{format_instructions}"
            ),
            input_variables=[
                "ticker",
                "as_of",
                "headline",
                "body",
                "market_context",
                "calibration_json",
            ],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        text = self._invoke(
            prompt.format(
                ticker=ticker,
                as_of=as_of.isoformat(),
                headline=headline,
                body=body,
                market_context=market_context or "",
                calibration_json=calibration.model_dump_json(indent=2),
            )
        )
        return _parse(parser, text, "surprise")

    def _behavior(
        self,
        *,
        ticker: str,
        as_of: date,
        headline: str,
        body: str,
        calibration: ExpectationCalibration,
        surprise: SurpriseAssessment,
        market_context: str | None,
    ) -> BehaviorPrediction:
        parser = PydanticOutputParser(pydantic_object=BehaviorPrediction)
        prompt = PromptTemplate(
            template=(
                "TASK: Predict the behavioral/emotional reaction path.\n"
                "Ticker: {ticker}\n"
                "As-of date: {as_of}\n"
                "Headline: {headline}\n"
                "Body: {body}\n"
                "Market context (optional): {market_context}\n\n"
                "Calibration JSON:\n{calibration_json}\n\n"
                "Surprise JSON:\n{surprise_json}\n\n"
                "Output should be practical: what happens next, where does it fail, what to watch.\n\n"
                "{format_instructions}"
            ),
            input_variables=[
                "ticker",
                "as_of",
                "headline",
                "body",
                "market_context",
                "calibration_json",
                "surprise_json",
            ],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        text = self._invoke(
            prompt.format(
                ticker=ticker,
                as_of=as_of.isoformat(),
                headline=headline,
                body=body,
                market_context=market_context or "",
                calibration_json=calibration.model_dump_json(indent=2),
                surprise_json=surprise.model_dump_json(indent=2),
            )
        )
        return _parse(parser, text, "behavior")

    def _invoke(self, prompt: str) -> str:
        try:
            return self._llm.complete(prompt, system=SYSTEM_PROMPT, model=self._model, temperature=0.2)
        except Exception as exc:
            raise LLMError(f"Narrative analyzer LLM call failed: {exc}") from exc


def _parse(parser: PydanticOutputParser, text: str, stage: str):
    try:
        return parser.parse(text)
    except Exception:
        # Try a minimal JSON-only fallback, since many models wrap JSON in prose.
        extracted = _extract_first_json_object(text)
        if extracted is not None:
            try:
                return parser.pydantic_object.model_validate_json(extracted)
            except Exception as exc:  # pragma: no cover - depends on model formatting
                raise LLMError(f"Failed to parse {stage} JSON: {exc}") from exc
        raise LLMError(f"Failed to parse {stage} output as structured JSON.")


def _extract_first_json_object(text: str) -> str | None:
    # Naive but effective: find the first '{' and match braces.
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : idx + 1]
                try:
                    json.loads(candidate)
                except Exception:
                    return None
                return candidate
    return None

