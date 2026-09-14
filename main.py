"""
main.py — Core End-to-End AI Support Agent Pipeline

Processes a single customer message through the full triage pipeline:
1. Intent Classification (Claude Haiku few-shot)
2. Case Retrieval (FAISS + all-MiniLM-L6-v2) & Grounding Quality Assessment
3. Escalation Decision (Safety Guardrails)
4. Reply Drafting (Grounded generation with defensive fallback)

Usage:
    # CLI argument
    python main.py "My Spotify keeps crashing whenever I try to shuffle a playlist"

    # Interactive prompt
    python main.py
"""

import argparse
import json
import sys
from typing import Any

from classify_intent import classify_intent
from draft_reply import _compute_grounding_quality, draft_reply
from escalate_or_handle import decide_escalation
from retrieve_similar_cases import retrieve_similar_cases


def process_customer_message(customer_text: str, index_mode: str = "heldout") -> dict[str, Any]:
    """
    Run the full triage and reply pipeline on a single customer message.
    """
    if not customer_text or not customer_text.strip():
        return {
            "error": "Empty customer message provided."
        }

    # Step 1: Intent Classification
    intent_res = classify_intent(customer_text)
    intent = intent_res["intent"]
    confidence = intent_res["confidence"]

    # Step 2: Similar Case Retrieval & Grounding Evaluation
    similar_cases = retrieve_similar_cases(customer_text, k=3, index_mode=index_mode)
    grounding = _compute_grounding_quality(similar_cases)

    # Step 3: Escalation Decision
    esc_res = decide_escalation(customer_text, intent, grounding)
    decision = esc_res["decision"]
    escalation_reason = esc_res["reason"]

    # Step 4: Grounded Reply Draft
    draft_res = draft_reply(customer_text, intent, index_mode=index_mode)
    draft = draft_res["draft"]

    return {
        "customer_text": customer_text,
        "intent": intent,
        "intent_confidence": confidence,
        "grounding_quality": grounding,
        "decision": decision,
        "escalation_reason": escalation_reason,
        "draft_reply": draft,
        "top_similar_cases": similar_cases,
    }


def print_formatted_result(result: dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("SPOTIFYCARES AI AGENT — TRIAGE & RESPONSE")
    print("=" * 80)
    print(f"Customer Tweet     : {result['customer_text']}")
    print(f"Intent Classified  : {result['intent']} (confidence: {result['intent_confidence']:.2f})")
    print(f"Grounding Quality  : {result['grounding_quality'].upper()}")
    
    decision_tag = "🔴 ESCALATE TO HUMAN" if result["decision"] == "escalate" else "🟢 AUTO-HANDLE"
    print(f"Routing Decision   : {decision_tag}")
    print(f"Decision Reason    : {result['escalation_reason']}")
    print("-" * 80)
    print(f"Draft Reply        :\n\"{result['draft_reply']}\"")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Process a single customer message with SpotifyCares AI Agent.")
    parser.add_argument("message", nargs="?", type=str, help="Customer message / tweet text.")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of human-readable text.")
    args = parser.parse_args()

    if args.message:
        text = args.message
    else:
        print("\nEnter a customer support message (or press Ctrl+C to exit):")
        try:
            text = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)

    if not text:
        print("No text provided.")
        sys.exit(1)

    result = process_customer_message(text)
    if args.json:
        # Filter non-serializable elements if any
        print(json.dumps(result, indent=2))
    else:
        print_formatted_result(result)


if __name__ == "__main__":
    main()
