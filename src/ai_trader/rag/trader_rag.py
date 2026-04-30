from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ai_trader.config import AppSettings, get_settings
from ai_trader.llm.errors import LLMConfigurationError
from ai_trader.rag.embeddings import (
    EmbeddingsBackend,
    LocalSentenceTransformersEmbeddings,
    get_embeddings_backend,
)
from ai_trader.rag.index import LocalVectorIndex, RetrievedChunk


class TraderRAG:
    def __init__(
        self,
        *,
        index_dir: Path,
        corpus_dir: Path,
        embeddings: EmbeddingsBackend,
    ) -> None:
        self._index_dir = index_dir
        self._corpus_dir = corpus_dir
        self._embeddings = embeddings
        self._index: LocalVectorIndex | None = None

    @classmethod
    def from_settings(cls, settings: AppSettings | None = None) -> TraderRAG:
        settings = settings or get_settings()
        return cls(
            index_dir=settings.rag_index_dir,
            corpus_dir=settings.rag_corpus_dir,
            embeddings=get_embeddings_backend(settings),
        )

    def load(self) -> None:
        self._index = LocalVectorIndex.load(self._index_dir)

    def build(self) -> None:
        self._index = LocalVectorIndex.build(corpus_dir=self._corpus_dir, embeddings=self._embeddings)
        self._index.save(self._index_dir)

    def ensure_loaded(self) -> None:
        if self._index is not None:
            return
        if (self._index_dir / "chunks.jsonl").exists() and (self._index_dir / "vectors.npy").exists():
            self.load()
            return
        raise LLMConfigurationError(
            f"RAG index not found at {self._index_dir}. Run: ai-trader rag-index"
        )

    def retrieve(self, query: str, *, k: int = 3) -> tuple[RetrievedChunk, ...]:
        self.ensure_loaded()
        assert self._index is not None
        return self._index.query(query, embeddings=self._embeddings, k=k)

    def stats(self) -> dict[str, int]:
        self.ensure_loaded()
        assert self._index is not None
        return {"chunks": self._index.size, "dimension": self._index.dimension}


def format_retrieved(chunks: tuple[RetrievedChunk, ...], *, max_chars: int = 1200) -> str:
    parts: list[str] = []
    for item in chunks:
        meta = item.chunk.metadata or {}
        source = meta.get("source") or meta.get("path") or item.chunk.chunk_id
        snippet = item.chunk.text.strip()
        if len(snippet) > max_chars:
            snippet = snippet[: max_chars - 3].rstrip() + "..."
        parts.append(f"[score={item.score:.3f}] {source}\n{snippet}")
    return "\n\n".join(parts)


def get_trader_rag(settings: AppSettings | None = None) -> TraderRAG:
    settings = settings or get_settings()
    backend = (settings.embeddings_backend or "").casefold().strip()
    if backend in {"local", "sentence-transformers", "sentence_transformers"}:
        return _cached_local_rag(
            str(settings.rag_index_dir),
            str(settings.rag_corpus_dir),
            settings.local_embedding_model,
        )
    return TraderRAG.from_settings(settings)


@lru_cache(maxsize=4)
def _cached_local_rag(index_dir: str, corpus_dir: str, model_name: str) -> TraderRAG:
    return TraderRAG(
        index_dir=Path(index_dir),
        corpus_dir=Path(corpus_dir),
        embeddings=LocalSentenceTransformersEmbeddings(model_name),
    )
