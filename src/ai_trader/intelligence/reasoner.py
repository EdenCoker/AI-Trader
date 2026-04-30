from __future__ import annotations

import json
from datetime import date

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from ai_trader.config import AppSettings, get_settings
from ai_trader.domain.signals import SignalBundle
from ai_trader.intelligence.guardrails import ReasoningGuardrails
from ai_trader.intelligence.models import NarrativeIntelligence
from ai_trader.intelligence.trade_plan import TradePlan
from ai_trader.llm import get_llm_client
from ai_trader.llm.contracts import LLMClient
from ai_trader.llm.errors import LLMError
from ai_trader.rag.trader_rag import TraderRAG, format_retrieved, get_trader_rag


SYSTEM_PROMPT = (
    "You are a trading system 'final reasoner'. "
    "You fuse multiple signals and narrative context into an auditable trade plan. "
    "Be conservative when inputs disagree or are stale. "
    "Return only the requested JSON."
)


class FinalReasoner:
    """Phase 5-ish: Fuse signals into a structured TradePlan."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        *,
        settings: AppSettings | None = None,
        model: str | None = None,
        rag: TraderRAG | None = None,
        guardrails: ReasoningGuardrails | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm = llm or get_llm_client(self._settings)
        self._model = model or self._settings.llm_model
        self._rag = rag
        self._guardrails = guardrails or ReasoningGuardrails()

    def reason(
        self,
        *,
        ticker: str,
        as_of: date,
        bundle: SignalBundle,
        narrative: NarrativeIntelligence | None = None,
        position_context: str | None = None,
    ) -> TradePlan:
        parser = PydanticOutputParser(pydantic_object=TradePlan)
        narrative_json = narrative.model_dump_json(indent=2) if narrative is not None else ""
        position_text = position_context or ""
        analogies_text = ""
        if self._settings.rag_enabled:
            rag = self._rag or get_trader_rag(self._settings)
            retrieved = rag.retrieve(self._rag_query(bundle, narrative), k=3)
            analogies_text = format_retrieved(retrieved)

        prompt = PromptTemplate(
            template=(
                "TASK: Produce a trade plan for the ticker using only the provided signals/context.\n"
                "Ticker: {ticker}\n"
                "As-of date: {as_of}\n\n"
                "SignalBundle JSON:\n{bundle_json}\n\n"
                "NarrativeIntelligence JSON (optional):\n{narrative_json}\n\n"
                "Retrieved Analogies (optional):\n{analogies_text}\n\n"
                "Current position context (optional): {position_context}\n\n"
                "Rules:\n"
                "- If evidence is weak or contradictory, use direction='neutral' with low conviction.\n"
                "- conviction is 0-1.\n"
                "- size_multiplier is 0-2 (0 means no trade).\n"
                "- holding_period_days should reflect the signal horizons.\n"
                "- exit_trigger must be specific and falsifiable.\n\n"
                "{format_instructions}"
            ),
            input_variables=[
                "ticker",
                "as_of",
                "bundle_json",
                "narrative_json",
                "analogies_text",
                "position_context",
            ],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        text = self._invoke(
            prompt.format(
                ticker=ticker,
                as_of=as_of.isoformat(),
                bundle_json=bundle.model_dump_json(indent=2),
                narrative_json=narrative_json,
                analogies_text=analogies_text,
                position_context=position_text,
            )
        )
        plan = _parse(parser, text)
        return self._guardrails.apply(plan=plan, bundle=bundle, narrative=narrative)

    def _invoke(self, prompt: str) -> str:
        try:
            return self._llm.complete(prompt, system=SYSTEM_PROMPT, model=self._model, temperature=0.2)
        except Exception as exc:
            raise LLMError(f"Final reasoner LLM call failed: {exc}") from exc

    @staticmethod
    def _rag_query(bundle: SignalBundle, narrative: NarrativeIntelligence | None) -> str:
        signals = ", ".join(f"{s.name}:{s.direction.value}:{s.strength:.2f}" for s in bundle.signals)
        narrative_hint = ""
        if narrative is not None:
            narrative_hint = (
                f"Surprise={narrative.surprise.direction} "
                f"score={narrative.surprise.surprise_score:.2f} "
                f"psychology={narrative.behavior.psychology_stage.value}"
            )
        return f"{bundle.ticker} trade decision analogies. Signals: {signals}. {narrative_hint}".strip()


def _parse(parser: PydanticOutputParser, text: str) -> TradePlan:
    try:
        return parser.parse(text)
    except Exception:
        extracted = _extract_first_json_object(text)
        if extracted is not None:
            try:
                return parser.pydantic_object.model_validate_json(extracted)
            except Exception as exc:  # pragma: no cover
                raise LLMError(f"Failed to parse TradePlan JSON: {exc}") from exc
        raise LLMError("Failed to parse trade plan output as structured JSON.")


def _extract_first_json_object(text: str) -> str | None:
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
