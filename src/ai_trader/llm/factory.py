from __future__ import annotations

from ai_trader.config import AppSettings, get_settings
from ai_trader.llm.contracts import LLMClient
from ai_trader.llm.errors import LLMConfigurationError
from ai_trader.llm.ollama import OllamaClient
from ai_trader.llm.openai_compatible import OpenAICompatibleClient


def get_llm_client(settings: AppSettings | None = None) -> LLMClient:
    settings = settings or get_settings()
    backend = (settings.llm_backend or "").casefold().strip()
    if backend in {"ollama"}:
        return OllamaClient(
            base_url=settings.ollama_base_url,
            default_model=settings.llm_model,
            timeout_s=settings.llm_timeout_s,
        )

    if backend in {"openai"}:
        if settings.openai_api_key is None:
            raise LLMConfigurationError(
                "OPENAI_API_KEY is required when AI_TRADER_LLM_BACKEND=openai"
            )
        return OpenAICompatibleClient(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key.get_secret_value(),
            default_model=settings.llm_model,
            timeout_s=settings.llm_timeout_s,
        )

    if backend in {"openai_compatible", "openai-compatible", "openai-compatible-server"}:
        api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        return OpenAICompatibleClient(
            base_url=settings.openai_base_url,
            api_key=api_key,
            default_model=settings.llm_model,
            timeout_s=settings.llm_timeout_s,
        )

    raise LLMConfigurationError(
        "Unsupported AI_TRADER_LLM_BACKEND. Use one of: openai, openai_compatible, ollama"
    )

