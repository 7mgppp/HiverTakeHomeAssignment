# SpotifyCares AI Support Agent — Golden Set Evaluation Report

## 1. Executive Summary & Baseline Comparison

| System / Baseline | Intent Accuracy | Intent Macro F1 | Escalation Accuracy | Escalation F1 | False Auto-Handles (Risk) |
|---|:---:|:---:|:---:|:---:|:---:|
| **AI Agent Pipeline (Claude Haiku)** | 96.0% | 0.955 | 88.7% | 0.894 | **0** (0.0%) |
| **Trivial Baseline (Majority Class)** | 23.3% | 0.054 | 52.0% | 0.000 | **72** (100.0%) |
| **Simple Baseline (Keyword Heuristics)** | 21.3% | 0.194 | 58.7% | 0.696 | **1** (1.4%) |

## 2. Intent Classification Breakdown (AI Agent)

| Intent Class | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| `feature_content_question` | 1.000 | 1.000 | 1.000 | 25 |
| `general_complaint_feedback` | 1.000 | 1.000 | 1.000 | 18 |
| `billing_subscription` | 0.958 | 0.920 | 0.939 | 25 |
| `other_uncategorized` | 1.000 | 1.000 | 1.000 | 10 |
| `account_login_access` | 0.957 | 0.880 | 0.917 | 25 |
| `playback_technical_issue` | 0.972 | 1.000 | 0.986 | 35 |
| `cancellation_refund` | 0.786 | 0.917 | 0.846 | 12 |

### Intent Confusion Matrix

```
Actual                     | feature_ general_ billing_ other_un account_ playback cancella
-------------------------------------------------------------------------------------------
feature_content_question   |       25        0        0        0        0        0        0 
general_complaint_feedback |        0       18        0        0        0        0        0 
billing_subscription       |        0        0       23        0        1        0        1 
other_uncategorized        |        0        0        0       10        0        0        0 
account_login_access       |        0        0        0        0       22        1        2 
playback_technical_issue   |        0        0        0        0        0       35        0 
cancellation_refund        |        0        0        1        0        0        0       11 
```

## 3. Escalation Decision & Safety Analysis

- **Escalation Accuracy**: 88.7%
- **Precision**: 0.809 | **Recall**: 1.000 | **F1**: 0.894
- **False Auto-Handle Count (Dangerous Misses)**: 0 (0.0%)
- **False Escalation Count (Unnecessary Agent Workload)**: 17 (21.8%)
