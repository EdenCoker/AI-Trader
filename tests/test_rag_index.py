from pathlib import Path

import numpy as np
import pytest

from ai_trader.rag.index import Chunk, LocalVectorIndex


class StubEmbeddings:
    def embed(self, texts):
        vectors = []
        for text in texts:
            lower = text.casefold()
            vectors.append(
                [
                    float(lower.count("value")),
                    float(lower.count("macro")),
                    float(lower.count("reflex")),
                ]
            )
        return vectors


def test_local_vector_index_build_query_save_load(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("value value quality", encoding="utf-8")
    (corpus / "b.txt").write_text("macro trend reflex", encoding="utf-8")

    embeddings = StubEmbeddings()
    index = LocalVectorIndex.build(corpus_dir=corpus, embeddings=embeddings, chunk_chars=100, overlap_chars=0)
    assert index.size >= 2
    assert index.dimension == 3

    results = index.query("macro", embeddings=embeddings, k=1)
    assert results[0].chunk.metadata["source"] == "b.txt"

    index_dir = tmp_path / "index"
    index.save(index_dir)
    loaded = LocalVectorIndex.load(index_dir)
    assert loaded.size == index.size

    results2 = loaded.query("value", embeddings=embeddings, k=1)
    assert results2[0].chunk.metadata["source"] == "a.txt"


def test_index_normalizes_vectors():
    chunks = [
        Chunk(chunk_id="a#0", text="a", metadata={}),
        Chunk(chunk_id="b#0", text="b", metadata={}),
    ]
    matrix = np.array([[3.0, 0.0], [0.0, 4.0]], dtype=np.float32)
    index = LocalVectorIndex(chunks=chunks, matrix=matrix)

    class QueryEmbeddings:
        def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    results = index.query("x", embeddings=QueryEmbeddings(), k=1)
    assert results[0].chunk.chunk_id == "a#0"
    assert results[0].score == pytest.approx(1.0)
