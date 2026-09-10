"""
Profile the Customer Support on Twitter dataset for a given brand.

Counts how many CLEAN (customer_problem -> brand_resolution) pairs you'd
actually get to work with, after filtering out junk. Run this for each
candidate brand before committing.

Usage:
    1. Download twcs.csv from:
       https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
    2. Place it in the same folder as this script (or pass --path)
    3. python profile_brand_data.py --brand XboxSupport
    4. python profile_brand_data.py --brand SpotifyCares
"""

import argparse
import re
import pandas as pd

DEFLECTION_PATTERNS = [
    r"\bdm us\b", r"\bsend us a dm\b", r"\bdirect message\b",
    r"\bfollow.*dm\b", r"\bplease dm\b", r"\bwe've sent you a dm\b",
]

def is_deflection(text: str) -> bool:
    text = text.lower()
    return any(re.search(p, text) for p in DEFLECTION_PATTERNS)

def is_probably_english(text: str) -> bool:
    # crude heuristic: mostly ASCII letters
    letters = sum(c.isalpha() for c in text)
    ascii_letters = sum(c.isalpha() and ord(c) < 128 for c in text)
    return letters == 0 or (ascii_letters / max(letters, 1)) > 0.85

def main(path: str, brand: str):
    print(f"Loading {path} ...")
    df = pd.read_csv(path, dtype=str)
    df["text"] = df["text"].fillna("")

    print(f"Total rows in dataset: {len(df):,}")

    # Brand's own tweets (their replies)
    brand_tweets = df[df["author_id"] == brand]
    print(f"\nTweets authored by {brand}: {len(brand_tweets):,}")

    # Index all tweets by tweet_id for fast lookup
    df_indexed = df.set_index("tweet_id")

    pairs = []
    for _, brand_row in brand_tweets.iterrows():
        parent_id = brand_row.get("in_response_to_tweet_id")
        if pd.isna(parent_id) or parent_id == "":
            continue  # brand tweet wasn't a reply to anyone
        if parent_id not in df_indexed.index:
            continue
        customer_row = df_indexed.loc[parent_id]
        # in_response_to could return a DataFrame if duplicate ids; guard
        if isinstance(customer_row, pd.DataFrame):
            customer_row = customer_row.iloc[0]
        if customer_row["author_id"] == brand:
            continue  # brand replying to itself, skip
        pairs.append({
            "customer_id": customer_row["author_id"],
            "customer_text": customer_row["text"],
            "brand_text": brand_row["text"],
        })

    raw_pair_count = len(pairs)
    print(f"Raw (customer -> brand reply) pairs found: {raw_pair_count:,}")

    pairs_df = pd.DataFrame(pairs)
    if pairs_df.empty:
        print("No pairs found — check brand handle spelling.")
        return

    # Filter 1: drop deflection-only replies ("please DM us")
    pairs_df["is_deflection"] = pairs_df["brand_text"].apply(is_deflection)
    after_deflection = pairs_df[~pairs_df["is_deflection"]]
    print(f"After removing pure 'DM us' deflections: {len(after_deflection):,}")

    # Filter 2: drop non-English (crude heuristic)
    after_deflection = after_deflection.copy()
    after_deflection["is_english"] = after_deflection["customer_text"].apply(is_probably_english)
    after_english = after_deflection[after_deflection["is_english"]]
    print(f"After removing likely non-English customer messages: {len(after_english):,}")

    # Filter 3: drop too-short customer messages (likely low-signal, e.g. just "@Brand")
    after_english = after_english.copy()
    after_english["msg_len"] = after_english["customer_text"].str.len()
    final = after_english[after_english["msg_len"] > 15]
    print(f"After removing very short customer messages (<15 chars): {len(final):,}")

    print(f"\n=== FINAL USABLE PAIRS FOR {brand}: {len(final):,} ===")
    print("\nRule of thumb: you want comfortably more than 1,500-2,000 usable")
    print("pairs to safely build a 6-8 intent taxonomy + 150-250 golden set")
    print("+ a retrieval index without sparse/empty categories.\n")

    print("Sample of 3 usable pairs:")
    for _, row in final.head(3).iterrows():
        print(f"  CUSTOMER: {row['customer_text'][:100]}")
        print(f"  BRAND:    {row['brand_text'][:100]}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="twcs.csv", help="Path to twcs.csv")
    parser.add_argument("--brand", required=True, help="e.g. XboxSupport or SpotifyCares")
    args = parser.parse_args()
    main(args.path, args.brand)