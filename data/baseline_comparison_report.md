# Benchmark & Baseline Comparison Report

Evaluated on 150 stratified golden-set customer messages from `data/golden_set_labeled.csv`.

## Comparison Table

| System / Baseline | Intent Accuracy | Escalation Precision | Escalation Recall | False Auto-Handles (Safety Risk) | LLM Judge Score (1-5) |
|---|:---:|:---:|:---:|:---:|:---:|
| **AI Agent Pipeline (Claude Haiku)** | **96.7%** | **72.7%** | **100.0%** | **0 (0.0%)** | **3.61 / 5.00** |
| **Simple Baseline (Keyword + 1-NN)** | 21.3% | 53.8% | 98.6% | 1 (1.4%) | 2.29 / 5.00 |
| **Trivial Baseline (Majority/Canned)** | 23.3% | 0.0% | 0.0% | 72 (100.0%) | 1.49 / 5.00 |

## Key Takeaways

1. **Intent Understanding**: The AI Agent achieves **96.7% intent accuracy** and 0.961 Macro-F1 across 7 classes, compared to only 21.3% for simple keywords and 23.3% for majority guessing. Keyword classification fails on conversational nuances, sarcasm, and multi-sentence complaints.
2. **Safety & Risk Mitigation**: The AI Agent achieves **100% Escalation Recall with 0 False Auto-Handles**, ensuring zero dangerous account-breach or churn-threat misses. The Trivial Baseline misses 100% of escalations (72/72 misses).
3. **Response Quality**: Grounded generative drafting achieves an average score of **3.61 / 5.00** on realistic support capability, significantly outperforming ungrounded static templates (1.49) and verbatim 1-NN retrieval (2.29, which frequently copies stale user tags, mid-thread fragments, or mismatched customer context).