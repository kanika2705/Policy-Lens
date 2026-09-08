"""
retriever.py
------------
A small wrapper around FAISS that stores chunk embeddings and lets us find
the most relevant chunks for a given question. This is the "search engine"
part of the RAG (Retrieval-Augmented Generation) pipeline.

How it works, conceptually:
1. Every chunk of the policy gets converted to an embedding (a vector).
2. All those vectors go into a FAISS index -- basically a fast structure for
   finding "which stored vectors are closest to this new vector".
3. When the user asks a question, we embed the question the same way, and
   ask FAISS for the K closest chunk vectors. "Closest" here roughly means
   "most similar in meaning".

We use cosine similarity (via normalized vectors + an inner-product index),
which is the standard choice for text embeddings.
"""

import numpy as np
import faiss

from src.embeddings import embed_texts, embed_query


class VectorStore:
    """Holds one policy document's chunks + their embeddings, searchable."""

    def __init__(self):
        self.index = None
        self.chunks = []  # parallel list: chunks[i] matches vector i in the index

    def build(self, chunks: list[dict]):
        """
        Embed every chunk's text and load the vectors into a FAISS index.
        Call this once per uploaded document.
        """
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        vectors = np.array(embed_texts(texts), dtype="float32")

        # Normalize vectors to unit length so that inner product == cosine similarity.
        faiss.normalize_L2(vectors)

        dimension = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # IP = inner product
        self.index.add(vectors)

    def search(self, query: str, k: int = 4) -> list[dict]:
        """
        Return the top-k chunks most relevant to `query`, each annotated
        with a similarity score (higher = more relevant).
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vector = np.array([embed_query(query)], dtype="float32")
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS pads with -1 if fewer than k results exist
                continue
            chunk = dict(self.chunks[idx])  # copy so we don't mutate the original
            chunk["score"] = float(score)
            results.append(chunk)
        return results
