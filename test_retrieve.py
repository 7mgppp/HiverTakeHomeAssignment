"""
test_retrieve.py — Smoke test for retrieve_similar_cases()

Runs 5 hand-picked queries that span different intents, prints the top-3
retrieved pairs for each so we can sanity-check retrieval relevance.

Usage:
    python test_retrieve.py

No API key needed — embeddings are fully local.
First run builds the index (~20 s); subsequent runs load from cache (<2 s).
"""

import logging
import textwrap

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

from retrieve_similar_cases import retrieve_similar_cases

QUERIES = [
    # (label, query_text)
    (
        "playback_technical_issue",
        "@SpotifyCares my app keeps crashing every time I try to shuffle "
        "a playlist on my iPhone after the latest update",
    ),
    (
        "account_login_access",
        "Can't log in to Spotify — I signed up with Facebook and now it says "
        "invalid credentials",
    ),
    (
        "billing_subscription",
        "I was charged twice this month for Premium and I'm still showing "
        "as a free user, wtf",
    ),
    (
        "feature_content_question",
        "When is Taylor Swift's new album going to be on Spotify? It's been "
        "two weeks since release and it's still not there",
    ),
    (
        "cancellation_refund",
        "I want to cancel my subscription and get a refund for this month, "
        "how do I do that",
    ),
]

SEP = "=" * 90

def main():
    print(f"\n{'RETRIEVE_SIMILAR_CASES — SMOKE TEST':^90}")
    print(f"5 queries × k=3 retrieved pairs each\n")

    for intent_label, query in QUERIES:
        print(SEP)
        print(f"[{intent_label}]")
        print(f"QUERY: {textwrap.shorten(query, width=85)}")
        print()

        results = retrieve_similar_cases(query, k=3)

        for rank, r in enumerate(results, 1):
            cust = textwrap.shorten(r["customer_text"], width=80)
            brand = textwrap.shorten(r["brand_text"], width=80)
            print(f"  #{rank}  score={r['score']:.4f}")
            print(f"       CUST : {cust!r}")
            print(f"       BRAND: {brand!r}")

        print()

    print(SEP)
    print("Done.")


if __name__ == "__main__":
    main()
