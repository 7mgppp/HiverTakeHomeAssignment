"""
retrieve_similar_cases.py — Step 3: Retrieval

Public API
----------
    from retrieve_similar_cases import retrieve_similar_cases

    cases = retrieve_similar_cases("My Spotify keeps crashing on shuffle", k=3)
    # [
    #   {
    #     "customer_text":  "...",
    #     "brand_text":     "...",
    #     "score":          0.92,
    #     "is_deflection":  False,
    #   },
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

Post-FAISS filtering (applied after raw candidate retrieval):
  1. Deduplication — if the same customer_text appears more than once (multi-turn
     threads reconstructed into multiple pairs), only the highest-scoring copy is kept.
  2. Deflection down-weighting — brand replies matching DM-deflection patterns (same
     regex list as build_dataset.py Stage 1) receive a score penalty of DEFLECTION_PENALTY.
     They are NOT removed entirely: if the top-k candidates are all deflections the caller
     still gets results, but 'is_deflection=True' signals to Step 4 that grounding is thin.
  To ensure k non-deflection results can be found when they exist, we fetch
  OVERSAMPLE * k raw candidates from FAISS before applying the filters.

Why local embeddings?
- all-MiniLM-L6-v2 quality is well-suited to short tweet-length texts.
- Embedding 33 k pairs via Claude API would cost ~$6 (exceeds the whole budget).
- Inference: ~20 s on CPU for the full corpus, <5 ms per query.

Files written to disk (gitignored):
    data/spotify_embeddings.npy      — float32 (33147, 384) embedding matrix
    data/spotify_faiss.index         — FAISS binary index
"""

import logging
import re
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

# Full index paths
PAIRS_CSV_FULL = DATA_DIR / "spotify_pairs_clean.csv"
EMBEDDINGS_NPY_FULL = DATA_DIR / "spotify_embeddings.npy"
FAISS_INDEX_FULL = DATA_DIR / "spotify_faiss.index"

# Held-out index paths (excludes golden-set source queries)
PAIRS_CSV_HELDOUT = DATA_DIR / "spotify_pairs_heldout.csv"
EMBEDDINGS_NPY_HELDOUT = DATA_DIR / "spotify_embeddings_heldout.npy"
FAISS_INDEX_HELDOUT = DATA_DIR / "spotify_faiss_heldout.index"

DEFAULT_INDEX_MODE = os.environ.get("FAISS_INDEX_MODE", "heldout").lower()

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension

# How many raw FAISS candidates to fetch per k requested.
# Gives the filter/dedup pipeline room to discard duplicates and deflections
# while still returning k results.
OVERSAMPLE = 8

# Score multiplier applied to deflection replies.
# 0.5 halves the cosine score, pushing them below any substantive reply
# at similarity > 0.5× the deflection's raw score.
DEFLECTION_PENALTY = 0.5

# ---------------------------------------------------------------------------
# DM-deflection detection — same patterns as build_dataset.py Stage 1
# ---------------------------------------------------------------------------

# Extended slightly vs. build_dataset.py to also catch patterns that survived
# because they were combined with other text (e.g. "Let's carry on in DMs")
_DEFLECTION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        # Core DM-mention patterns
        r"\bdm us\b",
        r"\bsend us a dm\b",
        r"\bshoot us a dm\b",
        r"\bdirect message\b",
        r"\bfollow.*\bdm\b",
        r"\bplease dm\b",
        r"\bcheck your dms?\b",
        r"\bslide into our dms\b",
        # "sent a/you a DM" variants (word order varies)
        r"\bsent (you )?a dm\b",
        r"\bsent a dm (your way|over|there)\b",
        r"\bsent.*\bdm\b.{0,20}(your way|over|there)\b",
        # "via DM" / "over DM" / "in a DM"
        r"\b(via|over|in( a)?)\s+dms?\b",
        # "responded to your DM" (with or without we've/we just)
        r"\bresponded to your dm\b",
        r"\breply(ing)? (to|via|in|over) (your )?dm\b",
        r"\breplied.*\bdm\b",
        # "chatting / helping / continuing there / backstage / in DMs"
        r"\bchat(ting)? (there|backstage|in dms?)\b",
        r"\bcarry on (chatting|helping( out)?)\b",
        r"\bcontinue (chatting|helping|there)\b",
        r"\bcontinuing (to help|chatting|there)\b",
        r"\bhelp(ing)? out (there|backstage)\b",
        r"\bcarry on (helping out|chatting) (there|backstage)\b",
        # "let's carry on / continue there / backstage"
        r"\blet'?s (carry on|continue|chat) (there|in|via).{0,20}dm\b",
        r"\blet'?s (carry on|continue|chat) (there|backstage)\b",
    ]
]


def is_deflection(text: str) -> bool:
    """
    True if the brand reply is primarily a 'DM us' / 'check your DMs' deflection.
    Reuses + extends the Stage 1 filter from build_dataset.py.
    """
    return any(p.search(text) for p in _DEFLECTION_PATTERNS)


# ---------------------------------------------------------------------------
# Module-level singletons & cache
# ---------------------------------------------------------------------------

_index_cache: dict[str, tuple[pd.DataFrame, faiss.Index]] = {}
_model: SentenceTransformer | None = None  # embedding model


def _get_model() -> SentenceTransformer:
    """Return (and cache) the sentence-transformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _build_or_load_index(mode: str = "heldout") -> tuple[pd.DataFrame, faiss.Index]:
    """
    Load the CSV, build embeddings (or load from cache), and return the
    dataframe + FAISS index for the requested mode ('heldout' or 'full').
    """
    if mode in _index_cache:
        return _index_cache[mode]

    if mode == "full":
        pairs_csv = PAIRS_CSV_FULL
        embeddings_npy = EMBEDDINGS_NPY_FULL
        faiss_index_path = FAISS_INDEX_FULL
    else:
        pairs_csv = PAIRS_CSV_HELDOUT if PAIRS_CSV_HELDOUT.exists() else PAIRS_CSV_FULL
        embeddings_npy = EMBEDDINGS_NPY_HELDOUT if EMBEDDINGS_NPY_HELDOUT.exists() else EMBEDDINGS_NPY_FULL
        faiss_index_path = FAISS_INDEX_HELDOUT if FAISS_INDEX_HELDOUT.exists() else FAISS_INDEX_FULL

    if not pairs_csv.exists():
        raise FileNotFoundError(f"{pairs_csv} not found.")

    logger.info("Loading pairs (%s mode) from %s …", mode, pairs_csv)
    df = pd.read_csv(pairs_csv, dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]
    logger.info("Loaded %d pairs (%s mode)", len(df), mode)

    texts = df["customer_text"].tolist()

    if embeddings_npy.exists() and faiss_index_path.exists():
        logger.info("Loading cached embeddings from %s …", embeddings_npy)
        embeddings = np.load(str(embeddings_npy))
        logger.info("Loading cached FAISS index from %s …", faiss_index_path)
        index = faiss.read_index(str(faiss_index_path))
    else:
        logger.info("Encoding %d texts with %s (%s mode) …", len(texts), MODEL_NAME, mode)
        model = _get_model()
        embeddings = model.encode(
            texts,
            batch_size=256,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        np.save(str(embeddings_npy), embeddings)
        logger.info("Saved embeddings to %s", embeddings_npy)

        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(embeddings)
        faiss.write_index(index, str(faiss_index_path))
        logger.info("Saved FAISS index to %s (%d vectors)", faiss_index_path, index.ntotal)

    _index_cache[mode] = (df, index)
    return df, index


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_similar_cases(customer_text: str, k: int = 3, index_mode: str | None = None) -> list[dict]:
    """
    Find the k most similar resolved support cases to a customer message.

    Post-FAISS processing:
      - Deduplicates on customer_text (keeps highest-scoring copy).
      - Down-weights DM-deflection brand replies by DEFLECTION_PENALTY so
        substantive resolutions sort first.
      - Returns up to k results; fewer if the corpus has no matches.

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
        score         : float — adjusted cosine similarity (after deflection penalty)
        raw_score     : float — original cosine similarity from FAISS
        is_deflection : bool  — True when brand_text is a DM-deflection reply
    """
    if not customer_text or not customer_text.strip():
        return []

    mode = (index_mode or DEFAULT_INDEX_MODE).lower()
    df, index = _build_or_load_index(mode)

    # Embed and normalise the query (model already cached after first call)
    model = _get_model()
    query_vec = model.encode(
        [customer_text.strip()],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    # Fetch OVERSAMPLE * k raw candidates — gives the filter pipeline room to work
    fetch_k = min(k * OVERSAMPLE, index.ntotal)
    scores, indices = index.search(query_vec, fetch_k)

    # -------------------------------------------------------------------------
    # Post-processing: dedup → flag deflections → re-sort → take k
    # -------------------------------------------------------------------------
    seen_customer_texts: set[str] = set()
    candidates: list[dict] = []

    for raw_score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        row = df.iloc[idx]
        cust = row["customer_text"]
        brand = row["brand_text"]

        # 1. Deduplicate by normalised customer_text (strip, lower, collapse whitespace)
        cust_key = " ".join(cust.lower().split())
        if cust_key in seen_customer_texts:
            continue
        seen_customer_texts.add(cust_key)

        # 2. Flag and penalise deflection replies
        defl = is_deflection(brand)
        adjusted_score = float(raw_score) * (DEFLECTION_PENALTY if defl else 1.0)

        candidates.append({
            "customer_text": cust,
            "brand_text":    brand,
            "score":         round(adjusted_score, 4),
            "raw_score":     round(float(raw_score), 4),
            "is_deflection": defl,
        })

    # 3. Re-sort by adjusted score descending, then take top k
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:k]

