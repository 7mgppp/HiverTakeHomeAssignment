# SpotifyCares AI Support Agent — Golden Set Evaluation Report

## 1. Executive Summary & Baseline Comparison

| System / Baseline | Intent Accuracy | Intent Macro F1 | Escalation Accuracy | Escalation F1 | False Auto-Handles (Risk) |
|---|:---:|:---:|:---:|:---:|:---:|
| **AI Agent Pipeline (Claude Haiku)** | 97.3% | 0.972 | 90.0% | 0.887 | **13** (18.1%) |
| **Trivial Baseline (Majority Class)** | 23.3% | 0.054 | 52.0% | 0.000 | **72** (100.0%) |
| **Simple Baseline (Keyword Heuristics)** | 21.3% | 0.194 | 58.7% | 0.696 | **1** (1.4%) |

## 2. Intent Classification Breakdown (AI Agent)

| Intent Class | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| `billing_subscription` | 0.958 | 0.920 | 0.939 | 25 |
| `account_login_access` | 1.000 | 0.960 | 0.980 | 25 |
| `playback_technical_issue` | 0.972 | 1.000 | 0.986 | 35 |
| `feature_content_question` | 1.000 | 0.960 | 0.980 | 25 |
| `general_complaint_feedback` | 1.000 | 1.000 | 1.000 | 18 |
| `other_uncategorized` | 1.000 | 1.000 | 1.000 | 10 |
| `cancellation_refund` | 0.857 | 1.000 | 0.923 | 12 |

### Intent Confusion Matrix

```
Actual                     | billing_ account_ playback feature_ general_ other_un cancella
-------------------------------------------------------------------------------------------
billing_subscription       |       23        0        0        0        0        0        2 
account_login_access       |        0       24        1        0        0        0        0 
playback_technical_issue   |        0        0       35        0        0        0        0 
feature_content_question   |        1        0        0       24        0        0        0 
general_complaint_feedback |        0        0        0        0       18        0        0 
other_uncategorized        |        0        0        0        0        0       10        0 
cancellation_refund        |        0        0        0        0        0        0       12 
```

## 3. Escalation Decision & Safety Analysis

- **Escalation Accuracy**: 90.0%
- **Precision**: 0.967 | **Recall**: 0.819 | **F1**: 0.887
- **False Auto-Handle Count (Dangerous Misses)**: 13 (18.1%)
- **False Escalation Count (Unnecessary Agent Workload)**: 2 (2.6%)
