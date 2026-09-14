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

# Churn & cancellation threat patterns (escalate regardless of intent)
CHURN_THREAT_PATTERNS = [
    (r"\b(going|about|have|might|may|ready|decided)\s+(to\s+|just\s+|have\s+to\s+)*(cancel|unsubscribe|leave|quit)\b",
     "Customer expressed intent or threat to cancel subscription/account."),
    (r"\bcancel\s+(my\s+|the\s+|this\s+|your\s+)?(subscription|account|premium|service|membership|app|everything)\b",
     "Customer mentioned canceling account or subscription."),
    (r"\bunsubscrib(e|ing)\b",
     "Customer mentioned unsubscribing from service."),
    (r"\bswitch(ing)?\s+(to|over\s+to)\s+(apple(\s+music)?|google(\s+music|\s+play)?|amazon(\s+music)?|tidal|deezer|pandora|youtube(\s+music)?|competitor|@\d+)\b",
     "Customer mentioned switching to a competing service."),
    (r"\bwhich\s+(one\s+)?(of\s+y['’]all\s+)?should\s+i\s+switch\s+to\b",
     "Customer solicited competitor recommendations."),
    (r"\bcancel\s+that\b",
     "Customer requested to cancel transaction/request."),
]

# Security breach, account takeover, and data privacy patterns (escalate regardless of intent)
SECURITY_BREACH_PATTERNS = [
    (r"\b(hack(ed|ing|er)?|compromised?|breach(ed)?)\b",
     "Customer reported account security breach or unauthorized access."),
    (r"\b(stolen|theft|fraudulent\s+access|credit\s+theft)\b",
     "Customer reported theft or fraudulent account activity."),
    (r"\bpersonal\s+information\s+(is\s+)?(in\s+danger|compromised|exposed|leaked)\b",
     "Customer expressed concern regarding personal data security."),
    (r"\bsomeone\s+(else\s+)?is\s+accessing\s+(my|our)\b",
     "Customer reported unauthorized third-party accessing their account."),
    (r"\bunauthorized\s+(access|charge|activity|login)\b",
     "Customer reported unauthorized account activity."),
]

_COMPILED_ANGER_PATTERNS = [
    (re.compile(pat, re.IGNORECASE), reason) for pat, reason in ANGER_URGENCY_PATTERNS
]
_COMPILED_CHURN_PATTERNS = [
    (re.compile(pat, re.IGNORECASE), reason) for pat, reason in CHURN_THREAT_PATTERNS
]
_COMPILED_SECURITY_PATTERNS = [
    (re.compile(pat, re.IGNORECASE), reason) for pat, reason in SECURITY_BREACH_PATTERNS
]

# Negation patterns preceding churn-threat keywords (e.g. "don't want to unsubscribe", "no plans to cancel")
_NEGATION_PREFIX_RE = re.compile(
    r"\b(don['’]?t|do\s+not|does\s+not|doesn['’]?t|did\s+not|didn['’]?t|not|never|no\s+plans\s+to|no\s+intention\s+to|no\s+need\s+to|wouldn['’]?t|would\s+not|won['’]?t|will\s+not|hate\s+to|refuse\s+to|avoid|without)\s+(\w+\s+){0,4}$",
    re.IGNORECASE,
)


def _detect_churn_threat(text: str) -> str | None:
    """True if message contains explicit churn, cancellation, or competitor switching threats (unless negated)."""
    if not text:
        return None
    for pattern, reason in _COMPILED_CHURN_PATTERNS:
        for match in pattern.finditer(text):
            prefix = text[:match.start()]
            if _NEGATION_PREFIX_RE.search(prefix.strip() + " "):
                continue
            return reason
    return None


def _detect_security_breach(text: str) -> str | None:
    """True if message reports hacking, security breaches, stolen credentials, or exposed private data."""
    if not text:
        return None
    for pattern, reason in _COMPILED_SECURITY_PATTERNS:
        if pattern.search(text):
            return reason
    return None


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

    # Rule (b): Security breach / account takeover / compromised data (regardless of intent)
    security_reason = _detect_security_breach(text)
    if security_reason:
        return {
            "decision": "escalate",
            "reason": security_reason,
        }

    # Rule (c): Churn threat / competitor switching / cancellation intent (regardless of intent)
    churn_reason = _detect_churn_threat(text)
    if churn_reason:
        return {
            "decision": "escalate",
            "reason": churn_reason,
        }

    # Rule (d): Weak retrieval grounding
    if norm_quality == "weak":
        return {
            "decision": "escalate",
            "reason": (
                "Retrieval grounding is weak or lacks relevant historical precedents; "
                "requires human assistance to avoid inaccurate resolution."
            ),
        }

    # Rule (e): Anger / urgency / sentiment signals
    anger_reason = _detect_anger_urgency(text)
    if anger_reason:
        return {
            "decision": "escalate",
            "reason": anger_reason,
        }

    # Rule (f): Uncategorized intent
    if norm_intent == "other_uncategorized":
        return {
            "decision": "escalate",
            "reason": "Intent could not be categorized into known support workflows; routing to human agent.",
        }

    # Rule (g): Safe for automated response
    return {
        "decision": "auto_handle",
        "reason": f"Standard {norm_intent.replace('_', ' ')} inquiry with adequate grounding ({norm_quality}).",
    }
