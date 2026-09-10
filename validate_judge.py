"""
validate_judge.py — Step 7: Judge Validation & Human Agreement Harness

1. Samples 30 rows randomly (seed=42) from data/judge_scores.csv to create
   a blind human annotation template: data/human_judge_sample.csv
2. When data/human_judge_sample.csv is labeled with human_score (1-5),
   computes agreement metrics between LLM Judge and Human Evaluator:
   - Exact Match Rate (%)
   - 1-Point Tolerance Agreement (% within ±1 point)
   - Mean Absolute Error (MAE)
   - Cohen's Weighted / Unweighted Kappa
   - Score Distribution Comparison

Usage:
    # 1. Generate the 30-sample blind template
    python validate_judge.py --sample

    # 2. Compute agreement metrics after filling in human_score column
    python validate_judge.py --evaluate
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
JUDGE_SCORES_CSV = DATA_DIR / "judge_scores.csv"
HUMAN_SAMPLE_CSV = DATA_DIR / "human_judge_sample.csv"


def generate_sample(n: int = 30, seed: int = 42):
    """Sample n rows from judge_scores.csv for blind human scoring."""
    if not JUDGE_SCORES_CSV.exists():
        raise FileNotFoundError(
            f"{JUDGE_SCORES_CSV} not found. Run judge_reply.py first to generate judge scores."
        )

    df = pd.read_csv(JUDGE_SCORES_CSV)
    logger.info("Loaded %d judge records. Sampling %d rows (seed=%d)...", len(df), n, seed)

    sample_df = df.sample(n=n, random_state=seed).reset_index(drop=True)

    # Save blind template without revealing judge_score or judge_rationale
    template_df = sample_df[["id", "customer_text", "draft"]].copy()
    template_df["human_score"] = ""

    template_df.to_csv(HUMAN_SAMPLE_CSV, index=False)
    logger.info("Successfully generated blind human evaluation template at %s", HUMAN_SAMPLE_CSV)

    print("\n" + "=" * 80)
    print(f"BLIND HUMAN EVALUATION TEMPLATE GENERATED ({n} ROWS)")
    print("=" * 80)
    print(f"File: {HUMAN_SAMPLE_CSV}")
    print("Columns: id, customer_text, draft, human_score")
    print("\nPreview of first 5 sample items for human rating (1-5):")
    for _, r in template_df.head(5).iterrows():
        print(f"\n[ID {r['id']:>3}]")
        print(f"  Customer: {r['customer_text'][:90]}...")
        print(f"  Draft   : {r['draft']}")
        print("  Human Score (1-5): [   ]")
    print("=" * 80 + "\n")


def calculate_cohen_kappa(y1: np.ndarray, y2: np.ndarray, categories: list[int] = [1, 2, 3, 4, 5]) -> float:
    """Compute unweighted Cohen's Kappa coefficient."""
    n = len(y1)
    if n == 0:
        return 0.0

    k = len(categories)
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    cm = np.zeros((k, k), dtype=float)

    for a, b in zip(y1, y2):
        if a in cat_to_idx and b in cat_to_idx:
            cm[cat_to_idx[a], cat_to_idx[b]] += 1

    p_observed = np.trace(cm) / n
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=2 if cm.ndim > 2 else 0)
    p_expected = np.sum(row_sums * col_sums) / (n * n)

    if p_expected == 1.0:
        return 1.0
    return (p_observed - p_expected) / (1.0 - p_expected)


def evaluate_agreement():
    """Evaluate agreement between judge_scores and human_scores."""
    if not HUMAN_SAMPLE_CSV.exists():
        raise FileNotFoundError(f"{HUMAN_SAMPLE_CSV} not found. Run with --sample first.")
    if not JUDGE_SCORES_CSV.exists():
        raise FileNotFoundError(f"{JUDGE_SCORES_CSV} not found.")

    human_df = pd.read_csv(HUMAN_SAMPLE_CSV)
    if "human_score" not in human_df.columns or human_df["human_score"].isna().all() or (human_df["human_score"] == "").all():
        print(f"\n⚠️  {HUMAN_SAMPLE_CSV} contains empty human_score column. Please score the rows (1-5) before evaluating.")
        return

    # Filter non-empty human scores
    valid_human = human_df[human_df["human_score"].astype(str).str.strip() != ""].copy()
    valid_human["human_score"] = valid_human["human_score"].astype(int)

    judge_df = pd.read_csv(JUDGE_SCORES_CSV)
    merged = pd.merge(valid_human, judge_df[["id", "judge_score", "judge_rationale"]], on="id", suffixes=("_human", "_judge"))

    n = len(merged)
    if n == 0:
        print("No matching scored rows found.")
        return

    h_scores = merged["human_score"].to_numpy()
    j_scores = merged["judge_score"].to_numpy()

    exact_matches = np.sum(h_scores == j_scores)
    exact_match_rate = exact_matches / n

    diffs = np.abs(h_scores - j_scores)
    within_one = np.sum(diffs <= 1)
    within_one_rate = within_one / n

    mae = np.mean(diffs)
    kappa = calculate_cohen_kappa(h_scores, j_scores)

    print("\n" + "=" * 80)
    print("LLM JUDGE VALIDATION & HUMAN AGREEMENT REPORT")
    print("=" * 80)
    print(f"Total Sampled & Scored : {n}")
    print(f"Human Average Score    : {h_scores.mean():.2f} / 5.00")
    print(f"Judge Average Score    : {j_scores.mean():.2f} / 5.00")
    print(f"Mean Absolute Error    : {mae:.2f} points\n")

    print(f"Exact Match Agreement  : {exact_matches:>2}/{n} ({exact_match_rate * 100:.1f}%)")
    print(f"Within ±1 Tolerance    : {within_one:>2}/{n} ({within_one_rate * 100:.1f}%)")
    print(f"Cohen's Kappa (κ)      : {kappa:.4f}")

    print("\n" + "-" * 80)
    print(f"{'ID':<6} {'Human':<8} {'Judge':<8} {'Diff':<8} {'Customer Snippet'}")
    print("-" * 80)
    for _, r in merged.iterrows():
        diff = abs(int(r["human_score"]) - int(r["judge_score"]))
        mark = "✓" if diff == 0 else ("~" if diff == 1 else "✗")
        print(f"{r['id']:<6} {r['human_score']:<8} {r['judge_score']:<8} {diff} {mark:<4} {r['customer_text'][:55]}...")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Judge validation and human agreement harness.")
    parser.add_argument("--sample", action="store_true", help="Generate the 30-item blind template.")
    parser.add_argument("--evaluate", action="store_true", help="Calculate agreement metrics after filling human scores.")
    args = parser.parse_args()

    if args.evaluate:
        evaluate_agreement()
    else:
        generate_sample()


if __name__ == "__main__":
    main()
