"""
retrieve_similar_cases.py — Step 3: Retrieval

Public API
----------
    from retrieve_similar_cases import retrieve_similar_cases

    cases = retrieve_similar_cases("My Spotify keeps crashing on shuffle", k=3)
    # [
    #   {"customer_text": "...", "brand_text": "...", "score": 0.92},
    #   ...
    # ]

Design
------
- Embeds customer_text from data/spotify_pairs_clean.csv using
  sentence-transformers/all-MiniLM-L6-v2 (22 MB, runs fully locally, no API cost).
- Builds a FAISS IndexFlatIP index over L2-normalised embeddings so inner-product
  == cosine similarity.
- Caches the raw embeddings to data/spotify_embeddings.npy and the FAISS index to
  data/spotify_faiss.index so the 33 k-pair encoding (≈20 s on CPU) only runs once.
- Everything is lazy-initialised: the first call to retrieve_similar_cases() triggers
  the build; subsequent calls within the same process are instant.
- The index excludes the query itself from results (not needed here since queries are
  new messages, not from the dataset, but the deduplication guard is free).

Why local embeddings?
- all-MiniLM-L6-v2 quality is well-suited to short tweet-length texts.
- Embedding 33 k pairs via Claude API would cost ~$6 (exceeds the whole budget).
- Inference: ~20 s on CPU for the full corpus, <5 ms per query.

Files written to disk (gitignored):
    data/spotify_embeddings.npy      — float32 (33147, 384) embedding matrix
    data/spotify_faiss.index         — FAISS binary index
"""

import logging
import os
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
PAIRS_CSV = DATA_DIR / "spotify_pairs_clean.csv"
EMBEDDINGS_NPY = DATA_DIR / "spotify_embeddings.npy"
FAISS_INDEX = DATA_DIR / "spotify_faiss.index"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension

# ---------------------------------------------------------------------------
# Module-level singletons — built on first call, reused thereafter
# ---------------------------------------------------------------------------

_df: pd.DataFrame | None = None       # the full pairs dataframe
_index: faiss.Index | None = None     # FAISS index
_model: SentenceTransformer | None = None  # embedding model


def _get_model() -> SentenceTransformer:
    """Return (and cache) the sentence-transformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _build_or_load_index() -> tuple[pd.DataFrame, faiss.Index]:
    """
    Load the CSV, build embeddings (or load from cache), and return the
    dataframe + FAISS index. Called at most once per process.
    """
    global _df, _index

    if _df is not None and _index is not None:
        return _df, _index

    # 1. Load pairs CSV
    if not PAIRS_CSV.exists():
        raise FileNotFoundError(
            f"{PAIRS_CSV} not found. Run build_dataset.py first."
        )
    logger.info("Loading pairs from %s …", PAIRS_CSV)
    df = pd.read_csv(PAIRS_CSV, dtype=str).fillna("")
    # Normalise column names defensively
    df.columns = [c.strip() for c in df.columns]
    logger.info("Loaded %d pairs", len(df))

    texts = df["customer_text"].tolist()

    # 2. Load or compute embeddings
    if EMBEDDINGS_NPY.exists() and FAISS_INDEX.exists():
        logger.info("Loading cached embeddings from %s …", EMBEDDINGS_NPY)
        embeddings = np.load(str(EMBEDDINGS_NPY))
        logger.info("Loading cached FAISS index from %s …", FAISS_INDEX)
        index = faiss.read_index(str(FAISS_INDEX))
    else:
        logger.info(
            "Encoding %d texts with %s — this takes ~20 s on CPU and runs once …",
            len(texts), MODEL_NAME,
        )
        model = _get_model()
        embeddings = model.encode(
            texts,
            batch_size=256,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalise so IP == cosine similarity
        )
        embeddings = embeddings.astype(np.float32)

        # Save embeddings cache
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        np.save(str(EMBEDDINGS_NPY), embeddings)
        logger.info("Saved embeddings to %s", EMBEDDINGS_NPY)

        # Build FAISS IndexFlatIP (exact cosine search after L2 normalisation)
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(embeddings)
        faiss.write_index(index, str(FAISS_INDEX))
        logger.info("Saved FAISS index to %s (%d vectors)", FAISS_INDEX, index.ntotal)

    _df = df
    _index = index
    return df, index


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_similar_cases(customer_text: str, k: int = 3) -> list[dict]:
    """
    Find the k most similar resolved support cases to a customer message.

    Parameters
    ----------
    customer_text : str
        The incoming customer tweet (raw text, @mentions OK).
    k : int
        Number of results to return (default 3).

    Returns
    -------
    list of dict, each with keys:
        customer_text : str   — historical customer message
        brand_text    : str   — SpotifyCares reply to that message
        score         : float — cosine similarity [0, 1]; higher = more similar
    """
    if not customer_text or not customer_text.strip():
        return []

    df, index = _build_or_load_index()

    # Embed and normalise the query (model already cached after first call)
    model = _get_model()
    query_vec = model.encode(
        [customer_text.strip()],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    # FAISS search — returns (scores, indices) each shape (1, k)
    fetch_k = min(k + 1, index.ntotal)   # +1 in case query itself is in index
    scores, indices = index.search(query_vec, fetch_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:          # FAISS returns -1 for padding when corpus < k
            continue
        row = df.iloc[idx]
        results.append({
            "customer_text": row["customer_text"],
            "brand_text":    row["brand_text"],
            "score":         float(score),
        })
        if len(results) >= k:
            break

    return results
