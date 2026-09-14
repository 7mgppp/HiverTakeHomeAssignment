"""
build_heldout_index.py — Build Held-out FAISS Index

Excludes any pair from data/spotify_pairs_clean.csv whose customer_text matches
(exact match or same first 50 characters of normalized text) any query in data/golden_set_labeled.csv.

Outputs:
- data/spotify_pairs_heldout.csv
- data/spotify_embeddings_heldout.npy
- data/spotify_faiss_heldout.index
"""

import logging
import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
PAIRS_CSV = DATA_DIR / "spotify_pairs_clean.csv"
GOLDEN_CSV = DATA_DIR / "golden_set_labeled.csv"

HELDOUT_PAIRS_CSV = DATA_DIR / "spotify_pairs_heldout.csv"
HELDOUT_EMBEDDINGS_NPY = DATA_DIR / "spotify_embeddings_heldout.npy"
HELDOUT_FAISS_INDEX = DATA_DIR / "spotify_faiss_heldout.index"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def normalize_text(text: str) -> str:
    """Normalize text by stripping mentions, punctuation, and extra whitespace."""
    if not isinstance(text, str):
        return ""
    t = re.sub(r"@\w+", "", text.lower())
    t = re.sub(r"[^\w\s]", "", t)
    return " ".join(t.split())


def build_heldout_dataset_and_index():
    logger.info("Loading labeled golden set from %s...", GOLDEN_CSV)
    golden_df = pd.read_csv(GOLDEN_CSV, dtype=str).fillna("")
    logger.info("Loaded %d golden-set rows.", len(golden_df))

    golden_texts = [normalize_text(t) for t in golden_df["customer_text"] if t.strip()]
    golden_exact_set = set(golden_texts)
    golden_prefixes = [t[:50] for t in golden_texts if len(t) >= 10]

    logger.info("Loading full cleaned corpus from %s...", PAIRS_CSV)
    pairs_df = pd.read_csv(PAIRS_CSV, dtype=str).fillna("")
    initial_count = len(pairs_df)
    logger.info("Initial corpus size: %d rows.", initial_count)

    # Filter out golden-set leakage
    excluded_mask = []
    for _, row in pairs_df.iterrows():
        c_norm = normalize_text(row["customer_text"])
        is_exact = c_norm in golden_exact_set
        is_prefix = any(c_norm[:50] == gp for gp in golden_prefixes if gp)
        excluded_mask.append(is_exact or is_prefix)

    excluded_count = sum(excluded_mask)
    logger.info("Identified %d leaked / matching rows to exclude.", excluded_count)

    heldout_df = pairs_df[~np.array(excluded_mask)].reset_index(drop=True)
    logger.info("Held-out corpus size: %d rows (retained %.2f%%).", len(heldout_df), 100 * len(heldout_df) / initial_count)

    # Save heldout CSV
    heldout_df.to_csv(HELDOUT_PAIRS_CSV, index=False)
    logger.info("Saved held-out pairs to %s", HELDOUT_PAIRS_CSV)

    # Generate Embeddings
    logger.info("Encoding %d customer texts with %s...", len(heldout_df), MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    texts = heldout_df["customer_text"].tolist()
    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    np.save(str(HELDOUT_EMBEDDINGS_NPY), embeddings)
    logger.info("Saved held-out embeddings to %s", HELDOUT_EMBEDDINGS_NPY)

    # Build FAISS Index
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    faiss.write_index(index, str(HELDOUT_FAISS_INDEX))
    logger.info("Saved held-out FAISS index to %s (ntotal=%d)", HELDOUT_FAISS_INDEX, index.ntotal)


if __name__ == "__main__":
    build_heldout_dataset_and_index()
