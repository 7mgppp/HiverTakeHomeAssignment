# SpotifyCares AI Support Agent — Golden Set Evaluation Report

## 1. Executive Summary & Baseline Comparison

| System / Baseline | Intent Accuracy | Intent Macro F1 | Escalation Accuracy | Escalation F1 | False Auto-Handles (Risk) |
|---|:---:|:---:|:---:|:---:|:---:|
| **AI Agent Pipeline (Claude Haiku)** | 96.0% | 0.958 | 80.0% | 0.828 | **0** (0.0%) |
| **Trivial Baseline (Majority Class)** | 23.3% | 0.054 | 52.0% | 0.000 | **72** (100.0%) |
| **Simple Baseline (Keyword Heuristics)** | 21.3% | 0.194 | 58.7% | 0.696 | **1** (1.4%) |

## 2. Intent Classification Breakdown (AI Agent)

| Intent Class | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| `general_complaint_feedback` | 1.000 | 1.000 | 1.000 | 18 |
| `feature_content_question` | 1.000 | 0.920 | 0.958 | 25 |
| `account_login_access` | 0.923 | 0.960 | 0.941 | 25 |
| `other_uncategorized` | 1.000 | 1.000 | 1.000 | 10 |
| `playback_technical_issue` | 0.972 | 1.000 | 0.986 | 35 |
| `cancellation_refund` | 0.846 | 0.917 | 0.880 | 12 |
| `billing_subscription` | 0.958 | 0.920 | 0.939 | 25 |

### Intent Confusion Matrix

```
Actual                     | general_ feature_ account_ other_un playback cancella billing_
-------------------------------------------------------------------------------------------
general_complaint_feedback |       18        0        0        0        0        0        0 
feature_content_question   |        0       23        0        0        0        1        1 
account_login_access       |        0        0       24        0        1        0        0 
other_uncategorized        |        0        0        0       10        0        0        0 
playback_technical_issue   |        0        0        0        0       35        0        0 
cancellation_refund        |        0        0        1        0        0       11        0 
billing_subscription       |        0        0        1        0        0        1       23 
```

## 3. Escalation Decision & Safety Analysis

- **Escalation Accuracy**: 80.0%
- **Precision**: 0.706 | **Recall**: 1.000 | **F1**: 0.828
- **False Auto-Handle Count (Dangerous Misses)**: 0 (0.0%)
- **False Escalation Count (Unnecessary Agent Workload)**: 30 (38.5%)

## 4. Data Leakage Audit: Leaked Index vs. Held-Out Index Comparison

To guarantee evaluation integrity and prevent circular grounding, all 158 matching and near-duplicate queries from the 150 golden-set examples were removed from the corpus (`data/spotify_pairs_clean.csv` $\to$ `data/spotify_pairs_heldout.csv`, retaining 32,996 independent historical pairs).

| Metric | Original (Leaked Index) | Clean Held-Out Index | Change / Impact |
|---|:---:|:---:|:---|
| **Mean Top-1 Retrieval Cosine Sim** | 0.9474 | 0.7766 | -0.1708 (Realistic similarity on unseen customer queries) |
| **Grounding Tier: Strong** | 133 (88.7%) | 69 (46.0%) | -42.7% (Fewer verbatim duplicate resolutions) |
| **Grounding Tier: Moderate** | 13 (8.7%) | 57 (38.0%) | +29.3% (Shifted to generalized pattern guidance) |
| **Grounding Tier: Weak** | 4 (2.7%) | 24 (16.0%) | +13.3% (Appropriately flagged novel/uncovered queries) |
| **Intent Classification Accuracy** | 96.0% (144/150) | 96.0% (144/150) | 0.0% (Zero-shot intent classification is ungrounded & invariant) |
| **Intent Macro-F1** | 0.955 | 0.958 | +0.003 (Minor LLM variance) |
| **Escalation Recall (Safety Metric)** | **100.0% (72/72)** | **100.0% (72/72)** | **0.0% (Zero false auto-handles maintained)** |
| **False Auto-Handles** | **0 (0.0%)** | **0 (0.0%)** | **0 (Perfect safety preserved)** |
| **Escalation Precision** | 80.9% | 70.6% | -10.3% (Safely routed weak-grounding cases to humans) |
| **False Escalation Count** | 17 (21.8%) | 30 (38.5%) | +13 (Safe fallback behavior on weak grounding) |
| **LLM-as-a-Judge Average Score** | **3.65 / 5.00** | **3.61 / 5.00** | **-0.04 points (Robust quality across unseen queries)** |
| **Judge Score: 4-5 Stars Rate** | 65.3% (98/150) | 67.3% (101/150) | +2.0% (Empathetic diagnostic replies scored favorably) |

