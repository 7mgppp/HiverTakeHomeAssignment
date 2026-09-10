# SpotifyCares AI Support Agent — Golden Set Evaluation Report

## 1. Executive Summary & Baseline Comparison

| System / Baseline | Intent Accuracy | Intent Macro F1 | Escalation Accuracy | Escalation F1 | False Auto-Handles (Risk) |
|---|:---:|:---:|:---:|:---:|:---:|
| **AI Agent Pipeline (Claude Haiku)** | 100.0% | 1.000 | 90.7% | 0.894 | **13** (18.1%) |
| **Trivial Baseline (Majority Class)** | 23.3% | 0.054 | 52.0% | 0.000 | **72** (100.0%) |
| **Simple Baseline (Keyword Heuristics)** | 21.3% | 0.194 | 58.7% | 0.696 | **1** (1.4%) |

## 2. Intent Classification Breakdown (AI Agent)

| Intent Class | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| `playback_technical_issue` | 1.000 | 1.000 | 1.000 | 35 |
| `billing_subscription` | 1.000 | 1.000 | 1.000 | 25 |
| `feature_content_question` | 1.000 | 1.000 | 1.000 | 25 |
| `other_uncategorized` | 1.000 | 1.000 | 1.000 | 10 |
| `cancellation_refund` | 1.000 | 1.000 | 1.000 | 12 |
| `account_login_access` | 1.000 | 1.000 | 1.000 | 25 |
| `general_complaint_feedback` | 1.000 | 1.000 | 1.000 | 18 |

### Intent Confusion Matrix

```
Actual                     | playback billing_ feature_ other_un cancella account_ general_
-------------------------------------------------------------------------------------------
playback_technical_issue   |       35        0        0        0        0        0        0 
billing_subscription       |        0       25        0        0        0        0        0 
feature_content_question   |        0        0       25        0        0        0        0 
other_uncategorized        |        0        0        0       10        0        0        0 
cancellation_refund        |        0        0        0        0       12        0        0 
account_login_access       |        0        0        0        0        0       25        0 
general_complaint_feedback |        0        0        0        0        0        0       18 
```

## 3. Escalation Decision & Safety Analysis

- **Escalation Accuracy**: 90.7%
- **Precision**: 0.983 | **Recall**: 0.819 | **F1**: 0.894
- **False Auto-Handle Count (Dangerous Misses)**: 13 (18.1%)
- **False Escalation Count (Unnecessary Agent Workload)**: 1 (1.3%)
