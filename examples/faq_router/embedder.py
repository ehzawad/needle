"""Embedding wrapper that applies the correct prompt convention per model.

Getting the query/document asymmetry wrong silently shifts the embedding
geometry and invalidates any calibrated OOS threshold, so this is centralized
in one place. Conventions:

- ``*-instruct`` E5 checkpoints: one-sentence task instruction on the QUERY
  (``Instruct: <task>\\nQuery: <text>``), raw text on the document/prototype side.
- non-instruct E5 (``e5-small-v2`` etc.): ``query: <text>`` and ``passage: <text>``.
- BGE English: a fixed retrieval instruction on the query, raw text on documents.
- anything else: raw text on both sides.

All outputs are L2-normalized, so cosine similarity is a plain dot product.
"""
from __future__ import annotations

import numpy as np

DEFAULT_QUERY_TASK = "Given a user utterance, retrieve the FAQ intent that expresses the same request."


class Embedder:
    def __init__(self, model_name: str = "intfloat/e5-small-v2", device=None,
                 task: str = DEFAULT_QUERY_TASK, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.task = task
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)

        n = model_name.lower()
        if "instruct" in n:
            self._q = lambda t: f"Instruct: {self.task}\nQuery: {t}"
            self._d = lambda t: t
        elif "bge" in n:
            self._q = lambda t: f"Represent this sentence for searching relevant passages: {t}"
            self._d = lambda t: t
        elif "e5" in n:
            self._q = lambda t: f"query: {t}"
            self._d = lambda t: f"passage: {t}"
        else:
            self._q = lambda t: t
            self._d = lambda t: t

    def encode_queries(self, texts) -> np.ndarray:
        return self._encode([self._q(t) for t in texts])

    def encode_docs(self, texts) -> np.ndarray:
        return self._encode([self._d(t) for t in texts])

    def _encode(self, texts) -> np.ndarray:
        emb = self.model.encode(
            texts, batch_size=self.batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        return emb.astype(np.float32)
