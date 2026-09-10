"""
sample_golden_set.py — Step 6: Stratified Golden-Set Sample Generator

Generates data/golden_set_unlabeled.csv containing 150 examples from
data/spotify_pairs_clean.csv stratified across the 7 intents.

Stratification Targets (Sum = 150):
  - playback_technical_issue    : 35
  - account_login_access         : 25
  - billing_subscription         : 25
  - feature_content_question     : 25
  - general_complaint_feedback   : 18
  - cancellation_refund          : 12 (min >= 10)
  - other_uncategorized          : 10 (min >= 10)

Output columns:
  id, customer_text, brand_text, auto_intent, auto_confidence,
  auto_decision, auto_reason, human_intent, human_decision,
  human_reason, notes

Usage:
    python sample_golden_set.py
"""

import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from classify_intent import classify_intent
from draft_reply import _compute_grounding_quality
from escalate_or_handle import decide_escalation
from retrieve_similar_cases import retrieve_similar_cases

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
INPUT_CSV = DATA_DIR / "spotify_pairs_clean.csv"
OUTPUT_CSV = DATA_DIR / "golden_set_unlabeled.csv"

# Target quotas per intent (sum = 150)
TARGET_QUOTAS = {
    "playback_technical_issue": 35,
    "account_login_access": 25,
    "billing_subscription": 25,
    "feature_content_question": 25,
    "general_complaint_feedback": 18,
    "cancellation_refund": 12,
    "other_uncategorized": 10,
}
TOTAL_TARGET = sum(TARGET_QUOTAS.values())

# Keyword seed filters to oversample candidate pools for rare/specific intents
KEYWORD_HINTS = {
    "cancellation_refund": [
        r"\bcancel\b", r"\brefund\b", r"\bdowngrade\b", r"\bunsubscribe\b",
        r"\bstop bill\b", r"\bmoney back\b"
    ],
    "other_uncategorized": [
        r"\bhire\b", r"\bjob\b", r"\bcareer\b", r"\bintern\b", r"\bartist payout\b",
        r"\broyalt\b", r"\bapi\b", r"\bdeveloper\b", r"\bwedding\b", r"\bmerch\b",
        r"\bpress\b", r"\binvestor\b", r"\bpartnership\b"
    ],
    "billing_subscription": [
        r"\bcharg\b", r"\bbill\b", r"\bpayment\b", r"\breceipt\b", r"\bcard\b",
        r"\bstudent\b", r"\bfamily\b", r"\bprice\b", r"\bcost\b"
    ],
    "general_complaint_feedback": [
        r"\bhate\b", r"\bsucks\b", r"\bworst\b", r"\bterrible\b", r"\bawful\b",
        r"\btrash\b", r"\bannoy\b", r"\brubbish\b", r"\buseless\b"
    ],
    "account_login_access": [
        r"\bpassword\b", r"\blogin\b", r"\blog in\b", r"\busername\b", r"\bfacebook\b",
        r"\bemail\b", r"\block\b", r"\bhack\b", r"\baccess\b"
    ],
    "feature_content_question": [
        r"\bhow to\b", r"\bhow do\b", r"\bwhere is\b", r"\bwhen will\b", r"\balbum\b",
        r"\blyric\b", r"\bequaliz\b", r"\bplaylist\b"
    ],
}


def build_candidate_queue(df: pd.DataFrame) -> list[dict]:
    """
    Construct a prioritized candidate queue to find examples for all intents
    with minimal API calls.
    """
    records = df.to_dict(orient="records")
    seen_indices = set()
    queue = []

    # 1. First add targeted candidates for rare intents (cancellation, other, complaints)
    for intent_name, patterns in KEYWORD_HINTS.items():
        combined_pat = re.compile("|".join(patterns), re.IGNORECASE)
        for i, r in enumerate(records):
            if i not in seen_indices and combined_pat.search(r["customer_text"]):
                queue.append(r)
                seen_indices.add(i)

    # 2. Add remaining records shuffled for general distribution
    remaining = [r for i, r in enumerate(records) if i not in seen_indices]
    queue.extend(remaining)
    return queue


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"{INPUT_CSV} not found. Run build_dataset.py first.")

    logger.info("Loading cleaned pairs from %s...", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")
    # Shuffle with fixed seed for reproducibility
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    candidate_queue = build_candidate_queue(df)
    logger.info("Candidate queue built with %d items.", len(candidate_queue))

    buckets = defaultdict(list)
    classified_count = 0

    logger.info("Sampling & auto-labeling up to %d stratified items...", TOTAL_TARGET)

    for cand in candidate_queue:
        # Check if all quotas are fulfilled
        if all(len(buckets[k]) >= TARGET_QUOTAS[k] for k in TARGET_QUOTAS):
            break

        c_text = cand["customer_text"].strip()
        b_text = cand["brand_text"].strip()

        # Classify intent using Claude Haiku
        res = classify_intent(c_text)
        classified_count += 1
        intent = res["intent"]
        conf = res["confidence"]

        # If bucket still has room, accept candidate
        if len(buckets[intent]) < TARGET_QUOTAS[intent]:
            # Step 4: retrieve and compute grounding quality
            cases = retrieve_similar_cases(c_text, k=3)
            grounding_quality = _compute_grounding_quality(cases)

            # Step 5: decide escalation
            esc_res = decide_escalation(
                customer_text=c_text,
                intent=intent,
                grounding_quality=grounding_quality,
            )

            item = {
                "customer_text": c_text,
                "brand_text": b_text,
                "auto_intent": intent,
                "auto_confidence": conf,
                "auto_decision": esc_res["decision"],
                "auto_reason": esc_res["reason"],
                "human_intent": "",
                "human_decision": "",
                "human_reason": "",
                "notes": "",
            }
            buckets[intent].append(item)

            if classified_count % 10 == 0 or len(buckets[intent]) == TARGET_QUOTAS[intent]:
                logger.info(
                    "Classified %d | Intent buckets: %s",
                    classified_count,
                    {k: f"{len(buckets[k])}/{TARGET_QUOTAS[k]}" for k in TARGET_QUOTAS}
                )

    # Flatten and order final dataset
    final_rows = []
    current_id = 1
    for intent, rows in buckets.items():
        for r in rows:
            r_ordered = {"id": current_id}
            r_ordered.update(r)
            final_rows.append(r_ordered)
            current_id += 1

    out_df = pd.DataFrame(final_rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    logger.info("Successfully generated %s with %d rows.", OUTPUT_CSV, len(out_df))

    # Print summary & preview
    print("\n" + "=" * 80)
    print("GOLDEN SET SAMPLING COMPLETE")
    print("=" * 80)
    print(f"Total classified API calls: {classified_count}")
    print("\nIntent Distribution:")
    dist = out_df["auto_intent"].value_counts()
    for intent, count in dist.items():
        print(f"  - {intent:<30}: {count:>3} (target: {TARGET_QUOTAS.get(intent, 0)})")

    print("\nEscalation Decision Breakdown:")
    dec_dist = out_df["auto_decision"].value_counts()
    for dec, count in dec_dist.items():
        print(f"  - {dec:<30}: {count:>3} ({count/len(out_df)*100:.1f}%)")

    print("\n" + "=" * 80)
    print("PREVIEW: First 10 Rows")
    print("=" * 80)
    preview_cols = ["id", "auto_intent", "auto_decision", "customer_text", "auto_reason"]
    for _, row in out_df.head(10)[preview_cols].iterrows():
        print(f"\n[ID {row['id']:>2}] INTENT: {row['auto_intent']} | DECISION: {row['auto_decision']}")
        print(f"      CUST  : {row['customer_text'][:100]}...")
        print(f"      REASON: {row['auto_reason']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
