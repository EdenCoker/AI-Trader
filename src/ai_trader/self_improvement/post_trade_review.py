from __future__ import annotations

import json
from typing import Literal

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, ConfigDict, Field

from ai_trader.config import AppSettings, get_settings
from ai_trader.llm import get_llm_client
from ai_trader.llm.contracts import LLMClient
from ai_trader.llm.errors import LLMError
from ai_trader.self_improvement.proposal import (
    PromptProposal,
    Proposal,
    TradeOutcome,
    WeightProposal,
    should_review,
)


SYSTEM_PROMPT = (
    "You are a conservative post-trade reviewer. "
    "Diagnose reasoning failures and suggest only minimal, auditable changes. "
    "Never suggest weakening look-ahead, disclosure-date, filing-date, or live-trading safety controls. "
    "Return only JSON matching the requested schema."
)


class CritiqueOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    reasoning_error: str
    missed_evidence: tuple[str, ...] = ()
    severity: float = Field(ge=0, le=1)


class RootCauseOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    root_cause: str
    target_kind: Literal["prompt", "weight"]
    target: str
    current_value: str


class ProposalOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    proposal_kind: Literal["prompt", "weight"]
    target: str
    current_text: str = ""
    proposed_text: str = ""
    current_value: float | None = None
    proposed_value: float | None = None
    delta_pct: float = 0.0
    rationale: str
    expected_improvement: str = ""


class PostTradeReviewer:
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

    def review(self, outcome: TradeOutcome) -> Proposal | None:
        if not should_review(outcome):
            return None

        critique = self._critique(outcome)
        root_cause = self._root_cause(outcome, critique)
        proposal = self._proposal(outcome, critique, root_cause)

        if proposal.proposal_kind == "weight":
            if proposal.current_value is None or proposal.proposed_value is None:
                raise LLMError("Weight proposal missing numeric values")
            return WeightProposal(
                source_outcome=outcome,
                target_field=proposal.target,
                current_value=proposal.current_value,
                proposed_value=proposal.proposed_value,
                delta_pct=proposal.delta_pct,
                rationale=proposal.rationale,
            )

        return PromptProposal(
            source_outcome=outcome,
            target_module=proposal.target,
            current_text=proposal.current_text,
            proposed_text=proposal.proposed_text,
            rationale=proposal.rationale,
            expected_improvement=proposal.expected_improvement,
        )

    def _critique(self, outcome: TradeOutcome) -> CritiqueOutput:
        parser = PydanticOutputParser(pydantic_object=CritiqueOutput)
        prompt = PromptTemplate(
            template=(
                "Here is the original thesis and actual outcome. What was wrong with the reasoning?\n"
                "Outcome JSON:\n{outcome_json}\n\n{format_instructions}"
            ),
            input_variables=["outcome_json"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        return _parse(parser, self._invoke(prompt.format(outcome_json=outcome.model_dump_json(indent=2))))

    def _root_cause(self, outcome: TradeOutcome, critique: CritiqueOutput) -> RootCauseOutput:
        parser = PydanticOutputParser(pydantic_object=RootCauseOutput)
        prompt = PromptTemplate(
            template=(
                "Which specific prompt text or weight caused the flawed reasoning?\n"
                "Outcome JSON:\n{outcome_json}\n\nCritique JSON:\n{critique_json}\n\n"
                "{format_instructions}"
            ),
            input_variables=["outcome_json", "critique_json"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        return _parse(
            parser,
            self._invoke(
                prompt.format(
                    outcome_json=outcome.model_dump_json(indent=2),
                    critique_json=critique.model_dump_json(indent=2),
                )
            ),
        )

    def _proposal(
        self,
        outcome: TradeOutcome,
        critique: CritiqueOutput,
        root_cause: RootCauseOutput,
    ) -> ProposalOutput:
        parser = PydanticOutputParser(pydantic_object=ProposalOutput)
        prompt = PromptTemplate(
            template=(
                "Write one concrete, minimal change that addresses the root cause.\n"
                "Outcome JSON:\n{outcome_json}\n\nCritique JSON:\n{critique_json}\n\n"
                "Root cause JSON:\n{root_cause_json}\n\n"
                "{format_instructions}"
            ),
            input_variables=["outcome_json", "critique_json", "root_cause_json"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        return _parse(
            parser,
            self._invoke(
                prompt.format(
                    outcome_json=outcome.model_dump_json(indent=2),
                    critique_json=critique.model_dump_json(indent=2),
                    root_cause_json=root_cause.model_dump_json(indent=2),
                )
            ),
        )

    def _invoke(self, prompt: str) -> str:
        try:
            return self._llm.complete(prompt, system=SYSTEM_PROMPT, model=self._model, temperature=0.1)
        except Exception as exc:
            raise LLMError(f"Post-trade review LLM call failed: {exc}") from exc


def _parse(parser: PydanticOutputParser, text: str):
    try:
        return parser.parse(text)
    except Exception:
        extracted = _extract_first_json_object(text)
        if extracted is None:
            raise LLMError("Post-trade review output was not valid JSON.")
        try:
            return parser.pydantic_object.model_validate_json(extracted)
        except Exception as exc:
            raise LLMError(f"Post-trade review output failed schema validation: {exc}") from exc


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : idx + 1]
                try:
                    json.loads(candidate)
                except Exception:
                    return None
                return candidate
    return None

