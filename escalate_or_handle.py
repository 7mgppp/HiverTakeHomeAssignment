"""
escalate_or_handle.py — Step 5: Escalation Decision

Public API
----------
    from escalate_or_handle import decide_escalation

    result = decide_escalation(
        customer_text="I was charged twice this month wtf",
        intent="billing_subscription",
        grounding_quality="moderate",
    )
    # {
    #   "decision": "escalate", # "auto_handle" | "escalate"
    #   "reason": "Billing and subscription changes involve financial transactions and require human verification.",
    # }

Design
------
- Purely deterministic and rule-based (no LLM calls, zero latency, explainable).
- Decision rules in priority order:
  1. Financial / Account Security:
     - 'billing_subscription' & 'cancellation_refund' require human/backend verification.
  2. Grounding Quality:
     - 'weak' grounding indicates thin/unreliable retrieval context.
  3. Sentiment / Urgency Signals:
     - Keywords indicating anger, frustration, profanity, legal threats, or high urgency.
     - Heuristics: excessive shouting (ALL CAPS) or punctuation distress (e.g. '???', '!!!').
  4. Uncategorized Intent:
     - 'other_uncategorized' cannot be safely routed to automated templates.
  5. Default:
     - 'auto_handle' when intent is safe, grounding is strong/moderate, and customer tone is manageable.
"""

import re
from typing import Literal

EscalationDecision = Literal["auto_handle", "escalate"]

# ---------------------------------------------------------------------------
# Sensitive Intents & Keywords
# ---------------------------------------------------------------------------

SENSITIVE_INTENTS = {
    "billing_subscription": (
        "Billing and subscription changes involve financial transactions "
        "and require account-level human verification."
    ),
    "cancellation_refund": (
        "Cancellations and refund requests involve monetary actions "
        "and require human agent processing."
    ),
}

# Regex patterns for anger, frustration, urgency, profanity, and legal threats
ANGER_URGENCY_PATTERNS = [
    # Profanity / extreme frustration
    (r"\b(wtf|f+u+c+k|b+u+l+l+s+h+i+t|damn|pissed|crap|hell)\b", "Detected strong frustration or profanity in message."),
    # Explicit distress / complaints
    (r"\b(scam|ripoff|thieves|stole|fraud|robbed)\b", "Customer expressed allegations of fraud or unfair charges."),
    (r"\b(unacceptable|horrible|terrible|worst service|ridiculous|disaster)\b", "Customer expressed high dissatisfaction."),
    # Legal / external escalation threats
    (r"\b(lawyer|attorney|lawsuit|sue|legal action|better business bureau|bbb|ftc)\b", "Customer mentioned potential legal action or regulatory escalation."),
    # High urgency keywords
    (r"\b(urgent|immediately|asap|right now|emergency)\b", "Customer indicated high urgency."),
]

_COMPILED_ANGER_PATTERNS = [
    (re.compile(pat, re.IGNORECASE), reason) for pat, reason in ANGER_URGENCY_PATTERNS
]


def _detect_anger_urgency(text: str) -> str | None:
    """
    Check if the customer text contains signals of anger, urgency, or extreme distress.
    Returns a descriptive reason string if triggered, else None.
    """
    if not text:
        return None

    # 1. Regex keyword matches
    for pattern, reason in _COMPILED_ANGER_PATTERNS:
        if pattern.search(text):
            return reason

    # 2. Heuristic: Excessive shouting (all caps words >= 3 letters, at least 45% of words)
    words = [w for w in re.findall(r"\b[A-Za-z]+\b", text) if len(w) >= 2]
    if len(words) >= 4:
        caps_words = [w for w in words if w.isupper() and len(w) >= 3]
        if len(caps_words) / len(words) >= 0.45:
            return "Customer message contains excessive capitalized text indicating elevated distress."

    # 3. Heuristic: Punctuation distress (multiple '?!' or '???' or '!!!!')
    if re.search(r"(\?{3,}|\!{3,}|\?\!|\!\?)", text):
        return "Message contains multiple exclamation or question marks indicating urgency/distress."

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decide_escalation(
    customer_text: str,
    intent: str,
    grounding_quality: str,
) -> dict:
    """
    Determine whether a customer inquiry should be auto-handled or escalated to a human agent.

    Parameters
    ----------
    customer_text : str
        Incoming customer tweet text.
    intent : str
        Taxonomy intent classification label.
    grounding_quality : str
        "strong" | "moderate" | "weak" from retrieval/drafting stage.

    Returns
    -------
    dict with keys:
        decision : "auto_handle" | "escalate"
        reason   : str (human-readable explanation)
    """
    text = (customer_text or "").strip()
    norm_intent = (intent or "").strip()
    norm_quality = (grounding_quality or "").strip().lower()

    # Rule (a): Sensitive financial / account-modifying intents
    if norm_intent in SENSITIVE_INTENTS:
        return {
            "decision": "escalate",
            "reason": SENSITIVE_INTENTS[norm_intent],
        }

    # Rule (b): Weak retrieval grounding
    if norm_quality == "weak":
        return {
            "decision": "escalate",
            "reason": (
                "Retrieval grounding is weak or lacks relevant historical precedents; "
                "requires human assistance to avoid inaccurate resolution."
            ),
        }

    # Rule (c): Anger / urgency / sentiment signals
    anger_reason = _detect_anger_urgency(text)
    if anger_reason:
        return {
            "decision": "escalate",
            "reason": anger_reason,
        }

    # Rule (d): Uncategorized intent
    if norm_intent == "other_uncategorized":
        return {
            "decision": "escalate",
            "reason": "Intent could not be categorized into known support workflows; routing to human agent.",
        }

    # Rule (e): Safe for automated response
    return {
        "decision": "auto_handle",
        "reason": f"Standard {norm_intent.replace('_', ' ')} inquiry with adequate grounding ({norm_quality}).",
    }
