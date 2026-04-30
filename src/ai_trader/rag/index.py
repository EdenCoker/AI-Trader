from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ai_trader.rag.embeddings import EmbeddingsBackend


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    score: float
    chunk: Chunk


class LocalVectorIndex:
    """Simple on-disk vector index using NumPy arrays and cosine similarity."""

    def __init__(self, *, chunks: list[Chunk], matrix: np.ndarray) -> None:
        if matrix.ndim != 2:
            raise ValueError("matrix must be 2D [n, d]")
        if len(chunks) != matrix.shape[0]:
            raise ValueError("chunk count must match matrix rows")
        self._chunks = chunks
        self._matrix = matrix.astype(np.float32, copy=False)
        self._matrix = _l2_normalize_rows(self._matrix)

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def dimension(self) -> int:
        return int(self._matrix.shape[1])

    @classmethod
    def build(
        cls,
        *,
        corpus_dir: Path,
        embeddings: EmbeddingsBackend,
        chunk_chars: int = 1600,
        overlap_chars: int = 200,
        glob: str = "**/*.txt",
    ) -> LocalVectorIndex:
        files = sorted(corpus_dir.glob(glob))
        chunks: list[Chunk] = []
        texts: list[str] = []
        for file_path in files:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            for idx, chunk_text in enumerate(_chunk_text(content, chunk_chars, overlap_chars)):
                chunk_id = f"{file_path.as_posix()}#{idx}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        metadata={"source": file_path.name, "path": file_path.as_posix()},
                    )
                )
                texts.append(chunk_text)

        if not chunks:
            raise ValueError(f"No .txt files found under {corpus_dir}")

        vectors = embeddings.embed(texts)
        matrix = np.asarray(vectors, dtype=np.float32)
        return cls(chunks=chunks, matrix=matrix)

    @classmethod
    def load(cls, index_dir: Path) -> LocalVectorIndex:
        index_dir.mkdir(parents=True, exist_ok=True)
        chunks_path = index_dir / "chunks.jsonl"
        matrix_path = index_dir / "vectors.npy"
        if not chunks_path.exists() or not matrix_path.exists():
            raise FileNotFoundError(f"Index not found in {index_dir}")

        chunks: list[Chunk] = []
        with chunks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                chunks.append(Chunk.model_validate_json(line))
        matrix = np.load(matrix_path)
        return cls(chunks=chunks, matrix=matrix)

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        chunks_path = index_dir / "chunks.jsonl"
        matrix_path = index_dir / "vectors.npy"

        with chunks_path.open("w", encoding="utf-8") as handle:
            for chunk in self._chunks:
                handle.write(chunk.model_dump_json())
                handle.write("\n")
        np.save(matrix_path, self._matrix)

    def query(self, query: str, *, embeddings: EmbeddingsBackend, k: int = 3) -> tuple[RetrievedChunk, ...]:
        if self.size == 0:
            return ()
        query_vec = np.asarray(embeddings.embed([query])[0], dtype=np.float32)
        query_vec = _l2_normalize(query_vec)
        scores = self._matrix @ query_vec
        k = min(max(1, k), self.size)
        top_idx = np.argpartition(-scores, kth=k - 1)[:k]
        ranked = sorted(((float(scores[i]), i) for i in top_idx), reverse=True)
        return tuple(RetrievedChunk(score=score, chunk=self._chunks[i]) for score, i in ranked)


def _chunk_text(text: str, chunk_chars: int, overlap_chars: int) -> Iterable[str]:
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be > 0")
    if overlap_chars < 0 or overlap_chars >= chunk_chars:
        raise ValueError("overlap_chars must be >=0 and < chunk_chars")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap_chars)
    return chunks


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector
    return vector / norm


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
