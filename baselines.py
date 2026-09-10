"""
baselines.py — Baseline Pipelines for Comparison & Benchmarking

Defines two reference baselines to benchmark the main AI agent against:
1. Trivial Baseline:
   - Always predicts majority class: 'playback_technical_issue'
   - Always decides 'auto_handle'
   - Drafts a generic static placeholder reply

2. Simple Baseline:
   - Rule/keyword-based intent classifier (deterministic regex matching)
   - Closest-match 1-NN retrieval: returns the single closest past brand reply verbatim
   - Basic keyword escalation: escalates on financial keywords, severe anger, or uncategorized
"""

import re
from typing import TypedDict
from retrieve_similar_cases import retrieve_similar_cases


class PipelineResult(TypedDict):
    intent: str
    confidence: float
    decision: str
    reason: str
    draft: str


# ---------------------------------------------------------------------------
# 1. Trivial Baseline
# ---------------------------------------------------------------------------

def trivial_pipeline(customer_text: str) -> PipelineResult:
    """
    Trivial Majority-Class Baseline:
    - Always predicts 'playback_technical_issue' (majority intent)
    - Always auto-handles
    - Uses static canned reply
    """
    return {
        "intent": "playback_technical_issue",
        "confidence": 1.0,
        "decision": "auto_handle",
        "reason": "Trivial baseline default: always auto-handle majority class.",
        "draft": "Hey there! Thanks for reaching out. Please restart your device and let us know if that helps!",
    }


# ---------------------------------------------------------------------------
# 2. Simple Rule/Keyword Baseline
# ---------------------------------------------------------------------------

KEYWORD_INTENT_RULES = [
    ("cancellation_refund", re.compile(r"\b(cancel|refund|downgrade|unsubscribe|money back|stop bill)\b", re.I)),
    ("billing_subscription", re.compile(r"\b(charg|bill|payment|receipt|card|student|family|premium price|cost|invoice)\b", re.I)),
    ("account_login_access", re.compile(r"\b(password|login|log in|username|facebook|email|locked|hack|credentials|access)\b", re.I)),
    ("feature_content_question", re.compile(r"\b(how to|how do|where is|when will|album|lyrics|equalizer|playlist|song)\b", re.I)),
    ("general_complaint_feedback", re.compile(r"\b(hate|sucks|worst|terrible|awful|trash|annoying|useless|rubbish)\b", re.I)),
    ("playback_technical_issue", re.compile(r"\b(crash|shuffle|lag|sound|pause|stops|skip|offline|volume|bluetooth|connect)\b", re.I)),
]

ANGER_KEYWORDS = re.compile(r"\b(wtf|fuck|bullshit|scam|fraud|lawyer|sue|asap|urgent|unacceptable)\b", re.I)


def simple_keyword_classify(text: str) -> tuple[str, float]:
    """Simple regex/keyword based intent classifier."""
    t = text.strip()
    for intent, pattern in KEYWORD_INTENT_RULES:
        if pattern.search(t):
            return intent, 0.70
    return "other_uncategorized", 0.30


def simple_pipeline(customer_text: str) -> PipelineResult:
    """
    Simple Heuristic Baseline:
    - Regex keyword classifier
    - 1-NN exact closest brand reply verbatim (no generative drafting)
    - Simple heuristic escalation on money/anger keywords
    """
    intent, conf = simple_keyword_classify(customer_text)

    # Escalation rule
    if intent in ("billing_subscription", "cancellation_refund"):
        decision = "escalate"
        reason = "Keyword rule: financial intent detected."
    elif ANGER_KEYWORDS.search(customer_text):
        decision = "escalate"
        reason = "Keyword rule: anger/urgency keyword detected."
    elif intent == "other_uncategorized":
        decision = "escalate"
        reason = "Keyword rule: no known intent pattern matched."
    else:
        decision = "auto_handle"
        reason = f"Keyword rule: standard {intent} matched."

    # Retrieval: take closest brand reply verbatim
    cases = retrieve_similar_cases(customer_text, k=1)
    draft = cases[0]["brand_text"] if cases else "Hey there! Thanks for reaching out. Let us know how we can help!"

    return {
        "intent": intent,
        "confidence": conf,
        "decision": decision,
        "reason": reason,
        "draft": draft,
    }
