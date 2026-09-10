"""
judge_reply.py — Step 7: LLM-as-a-Judge for Reply Quality

Public API
----------
    from judge_reply import judge_reply

    result = judge_reply(
        customer_text="My app keeps crashing on shuffle",
        draft_reply="Hey there! Sorry about that. Which iOS and Spotify version are you on?",
        grounding_quality="strong"
    )
    # {
    #   "score": 5,
    #   "rationale": "Directly asks for standard diagnostic details without fabricating ungrounded claims."
    # }

Rubric Dimensions (Overall 1-5 Score)
--------------------------------------
1. Groundedness: Does the reply align with real support resolution patterns without fabricating claims/links?
2. Correctness: Is the reply factually reasonable, safe, and logically sound given the customer query?
3. Tone: Matches SpotifyCares voice (warm, empathetic, concise, professional, Twitter-appropriate)?
4. Actionability: Provides the customer with a clear next step or focused diagnostic question?

Score Scale:
  5 - Excellent: Perfectly grounded, on-tone, accurate, clear actionable next step.
  4 - Good: Relevant, polite, and helpful; minor style or phrasing imperfection.
  3 - Acceptable: Safe and generic; asks a reasonable clarifying question, but could be more targeted.
  2 - Poor: Somewhat unhelpful, awkward tone, ungrounded assumption, or missing clear next step.
  1 - Unacceptable: Hallucinates URLs/facts, aggressive tone, harmful advice, or completely irrelevant.
"""

import json
import logging
import os
import re
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv

from draft_reply import draft_reply

load_dotenv()
logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"

DATA_DIR = Path(__file__).parent / "data"
GOLDEN_SET_CSV = DATA_DIR / "golden_set_labeled.csv"
JUDGE_SCORES_CSV = DATA_DIR / "judge_scores.csv"

_SYSTEM_PROMPT = """\
You are an expert Quality Assurance evaluator for SpotifyCares Twitter support responses.
Evaluate the provided draft reply to a customer tweet based on four core criteria:

1. Groundedness: Does the reply adhere to realistic support capabilities without fabricating URLs, fake dates, false promises, or unsupported claims?
2. Correctness: Is the response factually reasonable and appropriate for the customer's specific problem?
3. Tone: Is it friendly, empathetic, concise, and professional, matching the casual-yet-helpful Spotify brand voice (no robotic phrasing, no internal agent codes)?
4. Actionability: Does it give the user a clear next step (e.g. focused diagnostic question, setting instruction, or DM request)?

Rating Scale (1-5 integer):
5 - Excellent: Flawless tone, accurate, realistic, clear next step.
4 - Good: Safe, helpful, polite; minor phrasing imperfection.
3 - Acceptable: Safe but generic placeholder/clarification; lacks depth but causes no harm.
2 - Poor: Irrelevant elements, awkward tone, or misses key context.
1 - Unacceptable: Hallucinates facts/links, misleading advice, rude, or dangerous.

Output Format:
You must respond with ONLY a valid JSON object with exactly two keys:
{
  "score": <int 1-5>,
  "rationale": "<1-2 concise sentences explaining the score>"
}
"""

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set. Add it to .env.")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def judge_reply(customer_text: str, draft_reply_text: str, grounding_quality: str = "moderate") -> dict:
    """
    Evaluate the quality of a support draft reply using Claude Haiku as an impartial judge.

    Returns
    -------
    dict:
        score: int (1-5)
        rationale: str (1-2 sentences)
    """
    user_prompt = (
        f"Customer Tweet:\n\"{customer_text}\"\n\n"
        f"Draft Reply to Evaluate:\n\"{draft_reply_text}\"\n\n"
        f"Retrieval Grounding Quality: {grounding_quality}\n\n"
        f"Evaluate the reply according to the rubric and return the JSON score."
    )

    try:
        client = _get_client()
        message = client.messages.create(
            model=MODEL,
            max_tokens=256,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw_text = message.content[0].text.strip() if message.content else ""
        clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.MULTILINE).strip()
        data = json.loads(clean_json)
        score = int(data.get("score", 3))
        score = max(1, min(5, score))
        rationale = str(data.get("rationale", "")).strip()
        return {"score": score, "rationale": rationale}
    except Exception as exc:
        logger.warning("Judge call failed: %s", exc)
        return {
            "score": 3,
            "rationale": f"Fallback evaluation due to API error: {exc}",
        }


def run_golden_set_judging(input_csv: Path = GOLDEN_SET_CSV, output_csv: Path = JUDGE_SCORES_CSV):
    """Run draft generation and LLM judging across all 150 golden-set examples."""
    if not input_csv.exists():
        raise FileNotFoundError(f"{input_csv} not found.")

    logger.info("Loading golden set from %s...", input_csv)
    df = pd.read_csv(input_csv, dtype=str).fillna("")

    results = []
    print(f"\nEvaluating reply quality with LLM Judge across {len(df)} golden-set messages...\n")

    for i, row in df.iterrows():
        c_text = row["customer_text"].strip()
        intent = row.get("human_intent", row.get("auto_intent", "playback_technical_issue")).strip()

        # Generate draft reply
        draft_res = draft_reply(customer_text=c_text, intent=intent)
        draft = draft_res["draft"]
        quality = draft_res["grounding_quality"]

        # Judge draft reply
        judge_res = judge_reply(customer_text=c_text, draft_reply_text=draft, grounding_quality=quality)

        results.append({
            "id": row["id"],
            "customer_text": c_text,
            "draft": draft,
            "judge_score": judge_res["score"],
            "judge_rationale": judge_res["rationale"],
        })

        if (i + 1) % 25 == 0 or (i + 1) == len(df):
            logger.info("Judged %d/%d messages", i + 1, len(df))

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    logger.info("Saved judge scores to %s", output_csv)

    # Print summary statistics
    scores = out_df["judge_score"].astype(int)
    print("\n" + "=" * 80)
    print("LLM-AS-A-JUDGE EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Total Evaluated : {len(scores)}")
    print(f"Average Score   : {scores.mean():.2f} / 5.00")
    print(f"Median Score    : {scores.median():.1f} / 5.00")
    print("\nScore Distribution:")
    for s in range(5, 0, -1):
        count = (scores == s).sum()
        pct = (count / len(scores)) * 100
        bar = "█" * int(pct // 3)
        print(f"  {s} Stars: {count:>3} ({pct:>5.1f}%) | {bar}")

    print("\n" + "=" * 80)
    print("PREVIEW: First 5 Judged Drafts")
    print("=" * 80)
    for _, r in out_df.head(5).iterrows():
        print(f"\n[ID {r['id']:>2}] Score: {r['judge_score']}/5 | Rationale: {r['judge_rationale']}")
        print(f"      Customer: {r['customer_text'][:90]}...")
        print(f"      Draft   : {r['draft']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_golden_set_judging()
