"""
test_escalate_or_handle.py — Smoke test for decide_escalation()

Tests escalation decisions on the standard 5 sample queries, plus edge cases
covering all escalation rules (financial, weak grounding, anger/urgency, uncategorized).

Usage:
    python test_escalate_or_handle.py
"""

from escalate_or_handle import decide_escalation

# Standard 5 test queries from Steps 2-4 with their computed grounding_quality
SAMPLE_QUERIES = [
    {
        "intent": "playback_technical_issue",
        "customer_text": "@SpotifyCares my app keeps crashing every time I try to shuffle a playlist on my iPhone after the latest update",
        "grounding_quality": "strong",
    },
    {
        "intent": "account_login_access",
        "customer_text": "Can't log in to Spotify — I signed up with Facebook and now it says invalid credentials",
        "grounding_quality": "strong",
    },
    {
        "intent": "billing_subscription",
        "customer_text": "I was charged twice this month for Premium and I'm still showing as a free user, wtf",
        "grounding_quality": "moderate",
    },
    {
        "intent": "feature_content_question",
        "customer_text": "When is Taylor Swift's new album going to be on Spotify? It's been two weeks since release and it's still not there",
        "grounding_quality": "strong",
    },
    {
        "intent": "cancellation_refund",
        "customer_text": "I want to cancel my subscription and get a refund for this month, how do I do that",
        "grounding_quality": "moderate",
    },
]

# Additional edge cases to verify specific escalation rules
EDGE_CASES = [
    {
        "name": "Weak Grounding Edge Case",
        "intent": "playback_technical_issue",
        "customer_text": "My obscure custom Linux car audio setup won't stream properly",
        "grounding_quality": "weak",
    },
    {
        "name": "Anger / Profanity Signal",
        "intent": "general_complaint_feedback",
        "customer_text": "THIS APP IS COMPLETE BULLSHIT AND YOUR SERVICE IS A SCAM FIX THIS ASAP",
        "grounding_quality": "moderate",
    },
    {
        "name": "Legal Action Threat",
        "intent": "general_complaint_feedback",
        "customer_text": "I will be contacting my lawyer and reporting you to the BBB for unauthorized charges",
        "grounding_quality": "moderate",
    },
    {
        "name": "Uncategorized Intent",
        "intent": "other_uncategorized",
        "customer_text": "Can I hire Spotify DJ for my wedding next month???",
        "grounding_quality": "moderate",
    },
]

SEP = "=" * 80
SEP2 = "-" * 80


def main():
    print(f"\n{'ESCALATION DECISION SMOKE TEST':^80}")
    print(f"{'Evaluating 5 Sample Queries':^80}\n")

    for i, item in enumerate(SAMPLE_QUERIES, 1):
        print(SEP)
        print(f"[{i}] INTENT    : {item['intent']}")
        print(f"    GROUNDING : {item['grounding_quality']}")
        print(f"    QUERY     : {item['customer_text']}")
        print()

        res = decide_escalation(
            customer_text=item["customer_text"],
            intent=item["intent"],
            grounding_quality=item["grounding_quality"],
        )

        badge = "🟢 AUTO-HANDLE" if res["decision"] == "auto_handle" else "🔴 ESCALATE"
        print(f"    DECISION  : {badge}")
        print(f"    REASON    : {res['reason']}")
        print()

    print(f"\n{SEP}")
    print(f"{'Evaluating Escalation Rule Edge Cases':^80}")
    print(f"{SEP}\n")

    for i, item in enumerate(EDGE_CASES, 1):
        print(SEP2)
        print(f"[{i}] SCENARIO  : {item['name']}")
        print(f"    INTENT    : {item['intent']} | GROUNDING: {item['grounding_quality']}")
        print(f"    QUERY     : {item['customer_text']}")
        print()

        res = decide_escalation(
            customer_text=item["customer_text"],
            intent=item["intent"],
            grounding_quality=item["grounding_quality"],
        )

        badge = "🟢 AUTO-HANDLE" if res["decision"] == "auto_handle" else "🔴 ESCALATE"
        print(f"    DECISION  : {badge}")
        print(f"    REASON    : {res['reason']}")
        print()

    print(SEP)
    print("Done.")


if __name__ == "__main__":
    main()
