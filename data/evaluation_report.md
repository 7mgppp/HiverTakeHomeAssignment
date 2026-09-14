# SpotifyCares AI Support Agent — Golden Set Evaluation Report

## 1. Executive Summary & Baseline Comparison

| System / Baseline | Intent Accuracy | Intent Macro F1 | Escalation Accuracy | Escalation F1 | False Auto-Handles (Risk) |
|---|:---:|:---:|:---:|:---:|:---:|
| **AI Agent Pipeline (Claude Haiku)** | 96.7% | 0.961 | 82.0% | 0.842 | **0** (0.0%) |
| **Trivial Baseline (Majority Class)** | 23.3% | 0.054 | 52.0% | 0.000 | **72** (100.0%) |
| **Simple Baseline (Keyword Heuristics)** | 21.3% | 0.194 | 58.7% | 0.696 | **1** (1.4%) |

## 2. Intent Classification Breakdown (AI Agent)

| Intent Class | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| `general_complaint_feedback` | 1.000 | 0.944 | 0.971 | 18 |
| `cancellation_refund` | 0.750 | 1.000 | 0.857 | 12 |
| `playback_technical_issue` | 1.000 | 1.000 | 1.000 | 35 |
| `account_login_access` | 1.000 | 0.960 | 0.980 | 25 |
| `feature_content_question` | 1.000 | 0.920 | 0.958 | 25 |
| `other_uncategorized` | 1.000 | 1.000 | 1.000 | 10 |
| `billing_subscription` | 0.960 | 0.960 | 0.960 | 25 |

### Intent Confusion Matrix

```
Actual                     | general_ cancella playback account_ feature_ other_un billing_
-------------------------------------------------------------------------------------------
general_complaint_feedback |       17        1        0        0        0        0        0 
cancellation_refund        |        0       12        0        0        0        0        0 
playback_technical_issue   |        0        0       35        0        0        0        0 
account_login_access       |        0        1        0       24        0        0        0 
feature_content_question   |        0        1        0        0       23        0        1 
other_uncategorized        |        0        0        0        0        0       10        0 
billing_subscription       |        0        1        0        0        0        0       24 
```

## 3. Escalation Decision & Safety Analysis

- **Escalation Accuracy**: 82.0%
- **Precision**: 0.727 | **Recall**: 1.000 | **F1**: 0.842
- **False Auto-Handle Count (Dangerous Misses)**: 0 (0.0%)
- **False Escalation Count (Unnecessary Agent Workload)**: 27 (34.6%)
