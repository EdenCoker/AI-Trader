from __future__ import annotations

from typing import Sequence

import httpx

from ai_trader.llm.contracts import ChatMessage
from ai_trader.llm.errors import LLMError


class OpenAICompatibleClient:
    """Minimal OpenAI Chat Completions client.

    Works with OpenAI and with local servers that implement the OpenAI-compatible
    `/v1/chat/completions` endpoint (vLLM, llama.cpp server, etc.).
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        default_model: str,
        timeout_s: float = 60.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._default_model = default_model
        self._client = http_client or httpx.Client(timeout=timeout_s)

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        messages: list[ChatMessage] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        payload: dict[str, object] = {
            "model": model or self._default_model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        url = f"{self._base_url}/chat/completions"
        try:
            response = self._client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise LLMError(f"OpenAI-compatible request failed: {exc}") from exc
        except ValueError as exc:  # json decoding
            raise LLMError("OpenAI-compatible response was not valid JSON") from exc

        content = _extract_chat_content(data)
        if content is None:
            raise LLMError("OpenAI-compatible response missing assistant content")
        return content


def _extract_chat_content(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    first = choices[0]
    if isinstance(first, dict):
        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content

        # Some servers return a legacy `text` field.
        text = first.get("text")
        if isinstance(text, str):
            return text

    return None

