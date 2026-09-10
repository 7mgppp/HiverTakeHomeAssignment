"""
test_draft_reply.py — Smoke test for draft_reply()

Runs the same 5 queries used in test_retrieve.py, printing the draft,
grounding_quality, and the top retrieved example for each so we can
sanity-check drafting quality before Step 5.

Usage:
    python test_draft_reply.py
"""

import logging
import textwrap
import warnings

logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

from draft_reply import draft_reply

QUERIES = [
    ("playback_technical_issue",
     "@SpotifyCares my app keeps crashing every time I try to shuffle "
     "a playlist on my iPhone after the latest update"),
    ("account_login_access",
     "Can't log in to Spotify — I signed up with Facebook and now it "
     "says invalid credentials"),
    ("billing_subscription",
     "I was charged twice this month for Premium and I'm still showing "
     "as a free user, wtf"),
    ("feature_content_question",
     "When is Taylor Swift's new album going to be on Spotify? It's been "
     "two weeks since release and it's still not there"),
    ("cancellation_refund",
     "I want to cancel my subscription and get a refund for this month, "
     "how do I do that"),
]

SEP  = "=" * 80
SEP2 = "-" * 80

def wrap(text: str, indent: int = 4) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=78, initial_indent=prefix,
                         subsequent_indent=prefix)

def main():
    print(f"\n{'DRAFT_REPLY SMOKE TEST':^80}")
    print(f"{'5 queries — draft + grounding_quality + top retrieved example':^80}\n")

    for intent, query in QUERIES:
        print(SEP)
        print(f"INTENT  : {intent}")
        print(f"QUERY   : {query}")
        print()

        result = draft_reply(customer_text=query, intent=intent)

        quality = result["grounding_quality"]
        quality_badge = {"strong": "✅ strong", "moderate": "🟡 moderate",
                         "weak": "⚠️  weak"}[quality]

        print(f"GROUNDING: {quality_badge}")
        print()
        print("DRAFT:")
        print(wrap(result["draft"]))
        print()

        examples = result["retrieved_examples_used"]
        if examples:
            top = examples[0]
            defl = "  [DEFLECTION]" if top["is_deflection"] else ""
            print(f"TOP RETRIEVED (score={top['score']:.4f}){defl}:")
            print(wrap(f"CUST : {top['customer_text']}"))
            print(wrap(f"BRAND: {top['brand_text']}"))
        else:
            print("TOP RETRIEVED: (none)")
        print()

    print(SEP)
    print("Done.")

if __name__ == "__main__":
    main()
