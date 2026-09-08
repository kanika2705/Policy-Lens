"""
embeddings.py
-------------
Turns text into "embeddings" -- lists of numbers (vectors) that capture the
*meaning* of the text, not just its exact words.

Why we need this:
Two sentences can mean almost the same thing while sharing almost no words
("you must attend at least 75% of classes" vs "minimum attendance is 75
percent"). Plain keyword search would miss that they're related. Embeddings
place text into a "meaning space" where similar-meaning text ends up close
together, geometrically. That's what lets us later find "which chunks of the
policy are actually relevant to this question" even when the user's wording
doesn't match the document's wording.

We use a local Sentence-Transformers model, `all-MiniLM-L6-v2`, instead of a
paid API. It runs entirely on your own CPU (no GPU needed), downloads once
(~80MB, first run only), and needs no API key or billing. It's smaller than
OpenAI's embedding models, but plenty accurate for finding relevant chunks
in a policy document.
"""

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# We load the model once and reuse it for every call, instead of reloading it
# every time (that would be slow). This module-level variable acts as a
# simple cache: the first call loads the model, every call after that reuses it.
_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        # First call: this downloads the model from Hugging Face the very
        # first time it runs (needs internet once), then caches it locally
        # (usually in ~/.cache/torch or ~/.cache/huggingface) for every run
        # after that -- fully offline from then on.
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of strings into a list of embedding vectors.
    Batched into one call for efficiency (faster than one call per chunk).
    """
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    """Convenience wrapper for embedding a single piece of text (a question)."""
    return embed_texts([query])[0]
