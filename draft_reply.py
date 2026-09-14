"""
draft_reply.py — Step 4: Reply Drafting

Public API
----------
    from draft_reply import draft_reply

    result = draft_reply(
        customer_text="My Spotify keeps crashing when I try to shuffle",
        intent="playback_technical_issue",
    )
    # {
    #   "draft":                   "Hey there! Sorry to hear about the...",
    #   "grounding_quality":       "strong",      # "strong" | "moderate" | "weak"
    #   "retrieved_examples_used": [              # top-3 from retrieve_similar_cases
    #       {"customer_text": ..., "brand_text": ..., "score": ..., "is_deflection": ...},
    #       ...
    #   ],
    # }

Grounding quality tiers
-----------------------
  strong   top adjusted score ≥ 0.80, top result is non-deflection, non-fragment.
           Prompt instructs Claude to closely follow retrieved example style/structure.
  moderate top adjusted score ≥ 0.70, at least one usable (non-deflection, non-fragment)
           result exists. Prompt uses examples as loose guidance only.
  weak     top adjusted score < 0.70, OR all results are deflections/fragments, OR
           no cases returned. Prompt instructs Claude to give a cautious, safe
           "let us gather more details" style response rather than inventing specifics.

Step 5 (escalation) can use grounding_quality directly as a risk signal:
  strong   → safe to surface draft with light review
  moderate → flag for human review before sending
  weak     → escalate; draft is conservative placeholder only

Draft style
-----------
- Matches SpotifyCares Twitter voice: friendly, concise, first-person plural ("We", "we'll").
- Does NOT include @userID prefixes or agent initials (/NG, /CB …) from training data.
- Does NOT fabricate t.co shortlinks.
- Length target: 1-3 sentences (~100-200 chars), Twitter-appropriate.
- Asks for clarifying info (device, OS, account email) only when examples show that's
  the right next step — doesn't invent diagnostic steps without grounding.
"""

import json
import logging
import os
import re
from typing import Literal

import anthropic
from dotenv import load_dotenv

from retrieve_similar_cases import retrieve_similar_cases

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "claude-haiku-4-5-20251001"

GroundingQuality = Literal["strong", "moderate", "weak"]

# Fragment heuristic: brand replies that start with "2:", "3:", etc. are
# mid-thread continuation tweets that make no sense without prior context.
_FRAGMENT_RE = re.compile(r"^\s*\d+\s*:", re.ASCII)


# ---------------------------------------------------------------------------
# Grounding quality
# ---------------------------------------------------------------------------

def _is_fragment(brand_text: str) -> bool:
    """True when brand_text is a mid-thread continuation (starts with 'N:')."""
    return bool(_FRAGMENT_RE.match(brand_text.strip()))


def _compute_grounding_quality(cases: list[dict]) -> GroundingQuality:
    """
    Classify how well the retrieved cases can ground the reply draft.

    Parameters
    ----------
    cases : list of dicts returned by retrieve_similar_cases()

    Returns
    -------
    "strong" | "moderate" | "weak"
    """
    if not cases:
        return "weak"

    top_score = cases[0]["score"]  # adjusted score (post deflection penalty)

    # Usable = non-deflection AND non-fragment
    usable = [
        c for c in cases
        if not c["is_deflection"] and not _is_fragment(c["brand_text"])
    ]
    top_usable = usable[0] if usable else None

    if (
        top_score >= 0.80
        and top_usable is not None
        and top_usable["score"] >= 0.80
    ):
        return "strong"

    if top_score >= 0.70 and top_usable is not None:
        return "moderate"

    return "weak"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_prompt(
    customer_text: str,
    intent: str,
    cases: list[dict],
    quality: GroundingQuality,
) -> str:
    """
    Build the user-turn message that goes to Claude.
    The system prompt holds persona + global rules; this turn holds
    the specific query context.
    """
    # Format retrieved examples block
    example_lines = []
    for i, c in enumerate(cases, 1):
        tag = ""
        if c["is_deflection"]:
            tag = " [deflection — do not copy this style]"
        elif _is_fragment(c["brand_text"]):
            tag = " [fragment — context missing, use with caution]"
        example_lines.append(
            f"Example {i} (similarity={c['score']:.3f}){tag}:\n"
            f"  Customer: {c['customer_text']}\n"
            f"  SpotifyCares: {c['brand_text']}"
        )
    examples_block = "\n\n".join(example_lines) if example_lines else "(none retrieved)"

    # Grounding instruction varies by quality
    if quality == "strong":
        grounding_instruction = (
            "The retrieved examples are closely matched and highly relevant. "
            "USE THE ANSWER THE EXAMPLES GIVE — do not ask a clarifying question "
            "if the examples provide a direct answer. "
            "Mirror the tone and phrasing closely. "
            "If the examples give a specific factual answer (e.g. content not available, "
            "a troubleshooting step, a subscription detail), repeat that answer — "
            "it is grounded in real resolved cases and you are not fabricating it."
        )
    elif quality == "moderate":
        grounding_instruction = (
            "The retrieved examples are moderately relevant. "
            "Use them as loose style and structure guidance. "
            "If the examples suggest a next step (e.g. asking for account details "
            "or device info), follow that pattern. "
            "Do not assert specific facts from the examples if they may not apply "
            "to this exact situation."
        )
    else:  # weak
        grounding_instruction = (
            "The retrieved examples are not a close match or are mostly deflections. "
            "Do NOT invent specific troubleshooting steps, billing details, or "
            "content availability info you cannot ground. "
            "Instead, write a warm, cautious response that acknowledges the issue "
            "and asks for more details (e.g. device type, account email, or a "
            "brief description of the problem) before promising a resolution."
        )

    return (
        f"Intent classification: {intent}\n"
        f"Grounding quality: {quality}\n\n"
        f"Customer message:\n\"{customer_text}\"\n\n"
        f"Retrieved similar resolved cases:\n{examples_block}\n\n"
        f"Drafting instruction: {grounding_instruction}\n\n"
        f"Write the SpotifyCares reply draft now."
    )


_SYSTEM_PROMPT = """\
You are a SpotifyCares Twitter support agent drafting a reply to a customer tweet.

Rules — follow all of them exactly:
1. Voice: warm, helpful, concise. Use "we"/"our"/"we'll". Match the casual-but-professional
   tone of real SpotifyCares tweets.
2. Length: 1-3 sentences. Twitter-appropriate (~100-200 characters ideally; never over 400).
3. Do NOT include @userID prefixes (e.g. "@12345 Hey…") — start directly with a greeting
   or the content.
4. Do NOT include agent initials suffixes (e.g. "/NG", "/CB") — these are internal codes.
5. Do NOT fabricate links, URLs, or t.co shortlinks.
6. Do NOT invent specific facts (dates, prices, account details, content availability)
   that are NOT present in the retrieved examples. However, if the retrieved examples
   unanimously give a specific factual answer, you MUST use that answer — hedging or
   asking a clarifying question when a clear grounded answer exists is wrong.
7. If grounding quality is "weak" or examples are flagged [deflection]/[fragment]:
   acknowledge the issue warmly + ask one focused clarifying question.
   Do NOT invent specific diagnostic steps or resolution details.
8. Return ONLY the reply text — no preamble, no explanation, no quotes around it.
"""


# ---------------------------------------------------------------------------
# Client (lazy-initialised)
# ---------------------------------------------------------------------------

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set. Add it to .env or export it."
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def draft_reply(customer_text: str, intent: str, index_mode: str | None = None) -> dict:
    """
    Draft a SpotifyCares reply to a customer tweet.

    Parameters
    ----------
    customer_text : str
        Raw customer tweet text.
    intent : str
        Intent label from classify_intent() (e.g. "playback_technical_issue").
        Used to inform the drafting prompt.
    index_mode : str, optional
        Index mode to use for retrieval ('heldout' or 'full'). Defaults to 'heldout'.

    Returns
    -------
    dict with keys:
        draft                   : str  — the reply draft
        grounding_quality       : str  — "strong" | "moderate" | "weak"
        retrieved_examples_used : list — top-3 cases from retrieve_similar_cases()
    """
    if not customer_text or not customer_text.strip():
        return {
            "draft": "Hey there! Could you share more details about what you're experiencing? We're here to help.",
            "grounding_quality": "weak",
            "retrieved_examples_used": [],
        }

    # Step 1 — retrieve
    cases = retrieve_similar_cases(customer_text.strip(), k=3, index_mode=index_mode)

    # Step 2 — grounding quality
    quality = _compute_grounding_quality(cases)

    # Step 3 — draft via Claude
    user_prompt = _build_prompt(customer_text, intent, cases, quality)
    draft = ""
    try:
        client = _get_client()
        message = client.messages.create(
            model=MODEL,
            max_tokens=256,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        draft = message.content[0].text.strip() if message.content else ""
    except Exception as exc:
        logger.warning("Claude API call failed: %s", exc)
        draft = (
            "Hey there! Sorry to hear you're having trouble. "
            "Could you share a few more details so we can look into this for you?"
        )

    return {
        "draft": draft,
        "grounding_quality": quality,
        "retrieved_examples_used": cases,
    }
