"""
test_classify_intent.py — Sanity-check the intent classifier on 30 random
messages from data/spotify_pairs_sample.csv.

This is NOT the golden-set evaluation (Step 5). It is a quick smoke test to
verify the classifier produces sensible-looking output before further
investment.

Usage:
    export GOOGLE_API_KEY=...
    python test_classify_intent.py [--n 30] [--seed 99]
"""

import argparse
import sys
import time

import pandas as pd

from classify_intent import classify_intent, VALID_INTENTS

SAMPLE_CSV = "data/spotify_pairs_sample.csv"
DEFAULT_N = 30
DEFAULT_SEED = 99   # different from the pipeline seed so we get a fresh draw


def main(n: int, seed: int) -> None:
    # Load sample
    try:
        df = pd.read_csv(SAMPLE_CSV, dtype=str)
    except FileNotFoundError:
        print(f"ERROR: {SAMPLE_CSV} not found. Run build_dataset.py first.")
        sys.exit(1)

    df["customer_text"] = df["customer_text"].fillna("")
    sample = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)

    print(f"\nRunning classify_intent on {len(sample)} random messages "
          f"from {SAMPLE_CSV}  (seed={seed})\n")
    print("=" * 110)
    print(f"{'#':>3}  {'INTENT':<32}  {'CONF':>5}  {'CUSTOMER MESSAGE (first 80 chars)'}")
    print("=" * 110)

    intent_counts: dict[str, int] = {k: 0 for k in VALID_INTENTS}
    fallback_count = 0
    low_conf_count = 0

    for idx, row in sample.iterrows():
        text = row["customer_text"].replace("\n", " ").strip()
        result = classify_intent(text)

        intent = result["intent"]
        conf = result["confidence"]

        intent_counts[intent] = intent_counts.get(intent, 0) + 1
        if conf == 0.0 and intent == "other_uncategorized":
            fallback_count += 1
        if conf < 0.70:
            low_conf_count += 1

        # Truncate message for display
        display_text = text[:78] + ("…" if len(text) > 78 else "")

        print(f"{idx + 1:>3}  {intent:<32}  {conf:>5.2f}  {display_text!r}")

        # Small sleep to avoid rate-limit bursts (gpt-4o-mini tier-1 = 500 RPM)
        time.sleep(0.12)

    # Summary
    print("\n" + "=" * 110)
    print("DISTRIBUTION ACROSS 7 INTENTS")
    print("=" * 110)
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"  {intent:<32}  {count:>3}  {bar}")

    print(f"\n  Total classified : {len(sample)}")
    print(f"  API-error/fallbacks: {fallback_count}")
    print(f"  Low-confidence (<0.70): {low_conf_count}")
    print("\nDone. (Not a golden-set evaluation — see Step 5 for that.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help=f"Number of messages to classify (default: {DEFAULT_N})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed for sample draw (default: {DEFAULT_SEED})")
    args = parser.parse_args()
    main(args.n, args.seed)
