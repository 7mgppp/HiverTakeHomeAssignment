"""
build_dataset.py — Step 1 Data Pipeline for SpotifyCares AI Support Agent

Reconstructs (customer_message, brand_reply) pairs from the Kaggle
"Customer Support on Twitter" dataset (twcs.csv), applies a four-stage
filter funnel, and writes:

  data/spotify_pairs_clean.csv   — full cleaned set (~33k rows expected)
  data/spotify_pairs_sample.csv  — 2,000-row reproducible subsample (seed=42)

Usage:
    python build_dataset.py [--path twcs.csv]

Dependencies: pandas (already in your env via profile_brand_data.py)
"""

import argparse
import os
import re

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BRAND = "SpotifyCares"
RANDOM_SEED = 42
SAMPLE_SIZE = 2_000
OUTPUT_DIR = "data"
FULL_OUTPUT = os.path.join(OUTPUT_DIR, "spotify_pairs_clean.csv")
SAMPLE_OUTPUT = os.path.join(OUTPUT_DIR, "spotify_pairs_sample.csv")

# From profile_brand_data.py — extended with a few extra surface forms
DEFLECTION_PATTERNS = [
    r"\bdm us\b",
    r"\bsend us a dm\b",
    r"\bdirect message\b",
    r"\bfollow.*dm\b",
    r"\bplease dm\b",
    r"\bwe've sent you a dm\b",
]

# URL pattern used in the new link-out filter
URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def is_deflection(text: str) -> bool:
    """True if the brand reply is a pure 'DM us' deflection."""
    text = text.lower()
    return any(re.search(p, text) for p in DEFLECTION_PATTERNS)


def is_probably_english(text: str) -> bool:
    """
    Crude heuristic: message is probably English when >85% of its
    alphabetic characters are ASCII letters.
    """
    letters = sum(c.isalpha() for c in text)
    ascii_letters = sum(c.isalpha() and ord(c) < 128 for c in text)
    return letters == 0 or (ascii_letters / max(letters, 1)) > 0.85


def is_link_only_reply(text: str) -> bool:
    """
    True when the brand reply is a bare redirect with no real content:
      - shorter than 40 characters, AND
      - contains a URL, AND
      - after stripping the URL(s) there is almost no remaining text
        (<=10 non-whitespace chars left).
    """
    if len(text) >= 40:
        return False
    if not URL_PATTERN.search(text):
        return False
    stripped = URL_PATTERN.sub("", text).strip()
    return len(stripped.replace(" ", "")) <= 10


# ---------------------------------------------------------------------------
# Pair reconstruction
# ---------------------------------------------------------------------------

def build_pairs(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    """
    Walk every brand reply, look up the parent tweet, yield a pair.
    Mirrors the logic in profile_brand_data.py but also captures
    tweet_ids and created_at for the output schema.
    """
    brand_tweets = df[df["author_id"] == brand]
    print(f"Tweets authored by {brand}: {len(brand_tweets):,}")

    # Index the whole dataset once for O(1) parent lookups
    df_indexed = df.set_index("tweet_id")

    pairs = []
    for _, brand_row in brand_tweets.iterrows():
        parent_id = brand_row.get("in_response_to_tweet_id")
        if pd.isna(parent_id) or parent_id == "":
            continue
        if parent_id not in df_indexed.index:
            continue

        customer_row = df_indexed.loc[parent_id]
        # Guard against duplicate tweet_ids returning a DataFrame
        if isinstance(customer_row, pd.DataFrame):
            customer_row = customer_row.iloc[0]

        if customer_row["author_id"] == brand:
            continue  # brand replying to itself

        pairs.append({
            "customer_tweet_id": parent_id,
            "brand_tweet_id": brand_row["tweet_id"],
            "customer_id": customer_row["author_id"],
            "customer_text": customer_row["text"],
            "brand_text": brand_row["text"],
            # created_at comes from the customer tweet; fall back to brand tweet
            "created_at": customer_row.get("created_at", brand_row.get("created_at", "")),
        })

    return pd.DataFrame(pairs)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(csv_path: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, dtype=str)
    df["text"] = df["text"].fillna("")
    print(f"Total rows in dataset: {len(df):,}\n")

    # Reconstruct pairs
    print("Reconstructing (customer -> brand reply) pairs ...")
    pairs_df = build_pairs(df, BRAND)

    raw_count = len(pairs_df)
    print(f"Raw pairs found: {raw_count:,}\n")

    if pairs_df.empty:
        print("No pairs found — check that BRAND handle is correct.")
        return

    # Filter funnel
    print("=" * 62)
    print("FILTER FUNNEL")
    print("=" * 62)
    print(f"  Stage 0 — Raw pairs:                            {raw_count:>7,}")

    # Filter 1: pure DM-deflection brand replies
    mask_deflect = pairs_df["brand_text"].apply(is_deflection)
    pairs_df = pairs_df[~mask_deflect].copy()
    count_after_deflect = len(pairs_df)
    print(f"  Stage 1 — After drop DM-deflection replies:     {count_after_deflect:>7,}  "
          f"(removed {raw_count - count_after_deflect:,})")

    # Filter 2: non-English customer messages
    mask_english = pairs_df["customer_text"].apply(is_probably_english)
    pairs_df = pairs_df[mask_english].copy()
    count_after_english = len(pairs_df)
    print(f"  Stage 2 — After drop non-English messages:      {count_after_english:>7,}  "
          f"(removed {count_after_deflect - count_after_english:,})")

    # Filter 3: very short customer messages (<15 chars)
    mask_len = pairs_df["customer_text"].str.len() > 15
    pairs_df = pairs_df[mask_len].copy()
    count_after_len = len(pairs_df)
    print(f"  Stage 3 — After drop short messages (<15 ch):   {count_after_len:>7,}  "
          f"(removed {count_after_english - count_after_len:,})")

    # Filter 4 (NEW): bare link-out brand replies (<40 chars + URL + no substance)
    mask_linkout = pairs_df["brand_text"].apply(is_link_only_reply)
    pairs_df = pairs_df[~mask_linkout].copy()
    count_after_linkout = len(pairs_df)
    print(f"  Stage 4 — After drop bare link-only replies:    {count_after_linkout:>7,}  "
          f"(removed {count_after_len - count_after_linkout:,})")

    print("=" * 62)
    print(f"  FINAL USABLE PAIRS: {count_after_linkout:,}")
    print("=" * 62)

    # Assign pair_id and finalise column order
    pairs_df = pairs_df.reset_index(drop=True)
    pairs_df.insert(0, "pair_id", pairs_df.index + 1)

    final_cols = [
        "pair_id", "customer_id", "customer_text", "brand_text",
        "customer_tweet_id", "brand_tweet_id", "created_at",
    ]
    pairs_df = pairs_df[final_cols]

    # Save full cleaned dataset
    pairs_df.to_csv(FULL_OUTPUT, index=False)
    full_size_mb = os.path.getsize(FULL_OUTPUT) / (1024 * 1024)
    print(f"\nSaved full cleaned dataset  -> {FULL_OUTPUT}  ({full_size_mb:.1f} MB)")

    # Save 2,000-row reproducible sample
    n_sample = min(SAMPLE_SIZE, len(pairs_df))
    sample_df = pairs_df.sample(n=n_sample, random_state=RANDOM_SEED)
    sample_df = sample_df.reset_index(drop=True)
    sample_df.to_csv(SAMPLE_OUTPUT, index=False)
    sample_size_kb = os.path.getsize(SAMPLE_OUTPUT) / 1024
    print(f"Saved {n_sample}-row sample          -> {SAMPLE_OUTPUT}  ({sample_size_kb:.0f} KB)")

    # Sanity check: 3 random example pairs
    print("\n" + "=" * 62)
    print("SANITY CHECK — 3 random example pairs (seed=42)")
    print("=" * 62)
    for i, (_, row) in enumerate(
        pairs_df.sample(3, random_state=RANDOM_SEED).iterrows(), start=1
    ):
        print(f"\n  [{i}] pair_id={row['pair_id']}  customer={row['customer_id']}")
        print(f"      CUSTOMER: {repr(row['customer_text'][:140])}")
        print(f"      SPOTIFY:  {repr(row['brand_text'][:140])}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build SpotifyCares support pair dataset from twcs.csv"
    )
    parser.add_argument(
        "--path", default="twcs.csv",
        help="Path to twcs.csv (default: twcs.csv in current directory)"
    )
    args = parser.parse_args()
    main(args.path)
