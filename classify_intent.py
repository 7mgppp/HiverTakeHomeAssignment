"""
classify_intent.py — Step 2: Intent Classifier for SpotifyCares

Public API
----------
    from classify_intent import classify_intent

    result = classify_intent("My app keeps crashing whenever I hit shuffle")
    # {
    #   "intent": "playback_technical_issue",
    #   "confidence": 0.95,
    #   "raw_model_output": "<the model's full response>"
    # }

Backend: Anthropic Claude claude-haiku-4-5-20251001 (cheapest/fastest tier).
Confirmed model string from https://docs.anthropic.com/en/docs/about-claude/models/all-models
on 2026-09-11.

Pricing (claude-haiku-4-5-20251001):
    Input : $0.80 / 1M tokens
    Output: $4.00 / 1M tokens
    30-msg test   ≈ ~45k input + ~2k output tokens ≈ $0.04
    33k-pair batch ≈ ~50M input + ~2M output tokens ≈ $48 — use sample.csv for bulk work.

Requires:
    ANTHROPIC_API_KEY in environment (or in .env file at project root)
    pip install anthropic python-dotenv
"""

import json
import logging
import os
import re

import anthropic
from dotenv import load_dotenv

# Load .env at import time so the key is available without exporting manually
load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed intent taxonomy (do NOT change labels — used in Step 5 golden set)
# ---------------------------------------------------------------------------

VALID_INTENTS = {
    "playback_technical_issue",
    "account_login_access",
    "billing_subscription",
    "cancellation_refund",
    "feature_content_question",
    "general_complaint_feedback",
    "other_uncategorized",
}

FALLBACK = {"intent": "other_uncategorized", "confidence": 0.0, "raw_model_output": ""}

# Confirmed API model string — claude-haiku-4-5-20251001 is the latest Haiku
# as of 2026-09-11 per https://docs.anthropic.com/en/docs/about-claude/models/all-models
MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Prompt — top-level constant so it's easy to read and iterate without
# touching any logic
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an intent classifier for a Spotify customer-support Twitter dataset.

## Task
Classify the customer message into EXACTLY ONE of the 7 intents listed below
and return a JSON object — nothing else, no markdown fences.

## Intent definitions

| intent | description |
|--------|-------------|
| playback_technical_issue | Shuffle/repeat broken, app crashing, audio not playing, songs skipping, sync issues, device-specific playback bugs, media controls not responding |
| account_login_access | Cannot log in, password reset, linked account issues (Facebook/Apple/Google), account recovery, email changed without permission, security concerns |
| billing_subscription | Premium vs Free confusion, charged incorrectly, subscription not activating, family plan / student plan eligibility or setup, payment date queries |
| cancellation_refund | Explicit cancel request, refund request, downgrade request, threatening to leave over billing |
| feature_content_question | How-to questions, content availability / licensing (missing albums/artists), playlist or library limits, feature requests, availability by country |
| general_complaint_feedback | Pure venting or frustration with no specific fixable ask, general praise or thanks, feedback with no actionable request |
| other_uncategorized | Spam, unrelated content, context-free one-word replies, ambiguous fragments, non-support content |

## Overlap rule
If a message combines a fixable technical ask with anger/frustration, classify
by the fixable ask (e.g. playback_technical_issue), NOT the tone.
Only use general_complaint_feedback when there is NO specific fixable ask.

## Few-shot examples (all real customer messages from the dataset)

### playback_technical_issue
- "I've removed the old version and reinstalled the Spotify Windows app but it still won't actually play any songs :-/"
- "my media controls no longer work in macOS High Sierra. Are you able to fix this? Very frustrating. Spotify v1.0.66"
- "The song progress bar is stuck at 0:02, on the lock screen, while playing. iOS 11.1 bug?"

### account_login_access
- "someone changed my account email without having access to my email. Seems like a security flaw..."
- "trying to join wife's premium family plan but can't seem to join. heeelp!!"
- "how do I find and copy the link to my Spotify profile if I signed up with Facebook?"

### billing_subscription
- "I am paying 10 euros/month and I suddenly can not listen to all the music anymore!? :("
- "Why is the website not working? And just had my money taken yet my premium isn't working. Not on..."
- "thinking of switching over because of issues with the family premium plan. Does that work in conjunction with the promo of 50% off?"

### cancellation_refund
- "Cancelled due to your policy with the family plan. My adult children can't use my plan because they don't live in my house."
- "please help me cancel my payments for an account I do not have access to"
- "I am actually going to cancel my subscription based on this as I have no way to redownload them all. Shame"

### feature_content_question
- "Do you know when lovelyz and infinite company is gonna put their music back on spotify?"
- "will I ever be able to download more than 3333 songs? Or save more albums?"
- "When will you add an explicit filter? My kids don't want to hear those songs."

### general_complaint_feedback
- "STOP PLAYING SUGGESTED TRACKS WHEN IM TRYING TO STREAM SOMETHING IM TOO BROKE FOR PREMIUM"
- "you have such high quality branding but why are your ads so annoying? You have so much potential"
- "What the actual fuck why isn't this album available!!??"

### other_uncategorized
- "Still does not help."
- "All sorted now, thanks though"
- "please reply to my direct message. Thank you!"

## Output format — raw JSON only, no markdown, no preamble
{
  "intent": "<one of the 7 intent keys>",
  "confidence": <float 0.0-1.0, calibrated — do not default to 0.9>,
  "reasoning": "<one short sentence>"
}

Calibration guidance:
- 0.95+ : single clear intent, unambiguous
- 0.80-0.94 : probable intent, minor ambiguity
- 0.60-0.79 : best guess, meaningful ambiguity
- below 0.60 : very uncertain — consider other_uncategorized
"""

USER_PROMPT_TEMPLATE = 'Classify this customer message:\n\n"""{customer_text}"""'

# ---------------------------------------------------------------------------
# Client (lazy-initialised so import doesn't fail if key missing at import time)
# ---------------------------------------------------------------------------

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to .env at the project root or export it: "
                "export ANTHROPIC_API_KEY=sk-ant-..."
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Core classifier — same signature as before, nothing downstream changes
# ---------------------------------------------------------------------------

def classify_intent(customer_text: str) -> dict:
    """
    Classify a customer message into one of the 7 SpotifyCares intents.

    Parameters
    ----------
    customer_text : str
        Raw customer tweet text (may contain @mentions, URLs, emoji).

    Returns
    -------
    dict with keys:
        intent           : str   — one of VALID_INTENTS
        confidence       : float — 0.0-1.0
        raw_model_output : str   — full model response string (for debugging)
    """
    if not customer_text or not customer_text.strip():
        logger.warning("classify_intent called with empty text; returning fallback")
        return {**FALLBACK, "raw_model_output": ""}

    try:
        client = _get_client()
        message = client.messages.create(
            model=MODEL,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": USER_PROMPT_TEMPLATE.format(
                        customer_text=customer_text.strip()
                    ),
                }
            ],
        )
        raw = message.content[0].text if message.content else ""
    except Exception as exc:
        logger.warning("Anthropic API call failed: %s", exc)
        return {**FALLBACK, "raw_model_output": str(exc)}

    # Strip any accidental markdown fences the model adds despite instructions
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Model returned non-JSON output: %r", raw[:200])
        return {**FALLBACK, "raw_model_output": raw}

    intent = parsed.get("intent", "")
    if intent not in VALID_INTENTS:
        logger.warning("Model returned unknown intent %r; falling back", intent)
        return {**FALLBACK, "raw_model_output": raw}

    try:
        confidence = float(parsed.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "intent": intent,
        "confidence": confidence,
        "raw_model_output": raw,
    }
