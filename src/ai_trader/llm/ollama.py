from __future__ import annotations

from typing import Sequence

import httpx

from ai_trader.llm.contracts import ChatMessage
from ai_trader.llm.errors import LLMError


class OllamaClient:
    """Client for a local Ollama server (`ollama serve`)."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        default_model: str = "llama3.1",
        timeout_s: float = 60.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        # Use a short connect timeout but no read timeout: model generation time
        # is unbounded and varies with model size, prompt length, and hardware.
        self._client = http_client or httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None)
        )

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        think: bool = False,
    ) -> str:
        messages: list[ChatMessage] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, model=model, temperature=temperature, max_tokens=max_tokens, think=think)

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        think: bool = False,
    ) -> str:
        options: dict[str, object] = {"temperature": temperature, "num_ctx": 8192}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        model_name = model or self._default_model
        payload: dict[str, object] = {
            "model": model_name,
            "messages": list(messages),
            "stream": False,
            "options": options,
        }
        # Only send "think" for models that support extended thinking (qwen3 family).
        # Sending it to non-thinking models (e.g. dolphin3/llama) causes a 500 error.
        if "qwen3" in model_name.lower():
            payload["think"] = think

        url = f"{self._base_url}/api/chat"
        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        except ValueError as exc:
            raise LLMError("Ollama response was not valid JSON") from exc

        content = _extract_ollama_content(data)
        if content is None:
            raise LLMError("Ollama response missing assistant content")
        return content


def _extract_ollama_content(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    message = payload.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if not isinstance(content, str):
        return None
    return content

