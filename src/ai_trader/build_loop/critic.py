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
from ai_trader.self_improvement.proposal import Proposal, SAFETY_PATTERNS


class CritiqueResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    approved: bool
    confidence: float = Field(ge=0, le=1)
    concerns: tuple[str, ...] = ()
    overfitting_risk: Literal["low", "medium", "high"]


class SelfCritic:
    SAFETY_PATTERNS = SAFETY_PATTERNS

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

    def critique(self, proposal: Proposal) -> CritiqueResult:
        text = proposal.model_dump_json()
        safety_hit = self._safety_hit(text)
        if safety_hit is not None:
            return CritiqueResult(
                approved=False,
                confidence=1.0,
                concerns=(f"Safety pattern rejected before LLM call: {safety_hit}",),
                overfitting_risk="high",
            )

        parser = PydanticOutputParser(pydantic_object=CritiqueResult)
        prompt = PromptTemplate(
            template=(
                "Adversarially review this AI-generated trading-system proposal. "
                "Reject fragile, broad, unsafe, or overfit changes.\n"
                "Proposal JSON:\n{proposal_json}\n\n{format_instructions}"
            ),
            input_variables=["proposal_json"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        result = _parse(
            parser,
            self._invoke(prompt.format(proposal_json=proposal.model_dump_json(indent=2))),
        )
        if result.overfitting_risk == "high":
            return result.model_copy(update={"approved": False})
        return result

    def _safety_hit(self, text: str) -> str | None:
        lowered = text.casefold()
        for pattern in self.SAFETY_PATTERNS:
            if pattern.casefold() in lowered:
                return pattern
        return None

    def _invoke(self, prompt: str) -> str:
        try:
            return self._llm.complete(prompt, system="Return only JSON.", model=self._model, temperature=0.3)
        except Exception as exc:
            raise LLMError(f"SelfCritic LLM call failed: {exc}") from exc


def _parse(parser: PydanticOutputParser, text: str) -> CritiqueResult:
    try:
        return parser.parse(text)
    except Exception:
        extracted = _extract_first_json_object(text)
        if extracted is None:
            raise LLMError("SelfCritic output was not JSON")
        return parser.pydantic_object.model_validate_json(extracted)


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

