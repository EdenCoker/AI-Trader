from __future__ import annotations

from typing import Protocol, Sequence

import httpx

from ai_trader.config import AppSettings, get_settings
from ai_trader.llm.errors import LLMConfigurationError, LLMError


class EmbeddingsBackend(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class LocalSentenceTransformersEmbeddings:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "Local embeddings require `sentence-transformers`. Install with extras: `.[rag]`."
                ) from exc
            self._model = SentenceTransformer(self._model_name)
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]


class OpenAIEmbeddings:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 60.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._client = http_client or httpx.Client(timeout=timeout_s)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        url = f"{self._base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {"model": self._model, "input": list(texts)}
        try:
            response = self._client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise LLMError(f"OpenAI embeddings request failed: {exc}") from exc
        except ValueError as exc:
            raise LLMError("OpenAI embeddings response was not valid JSON") from exc

        vectors: list[list[float]] = []
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise LLMError("OpenAI embeddings response missing 'data'")
        for item in items:
            if not isinstance(item, dict):
                raise LLMError("OpenAI embeddings response malformed")
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                raise LLMError("OpenAI embeddings item missing embedding")
            vectors.append([float(x) for x in embedding])
        return vectors


def get_embeddings_backend(settings: AppSettings | None = None) -> EmbeddingsBackend:
    settings = settings or get_settings()
    backend = (settings.embeddings_backend or "").casefold().strip()
    if backend in {"local", "sentence-transformers", "sentence_transformers"}:
        return LocalSentenceTransformersEmbeddings(settings.local_embedding_model)
    if backend in {"openai"}:
        if settings.openai_api_key is None:
            raise LLMConfigurationError(
                "OPENAI_API_KEY is required when AI_TRADER_EMBEDDINGS_BACKEND=openai"
            )
        return OpenAIEmbeddings(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.openai_embedding_model,
            timeout_s=settings.llm_timeout_s,
        )
    raise LLMConfigurationError("Unsupported embeddings backend. Use: local, openai")
