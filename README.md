# SpotifyCares AI Support Agent

An end-to-end AI-powered customer support triage and response system for Spotify customer inquiries, built with retrieval-augmented generation (RAG), intent classification, safety-biased escalation guardrails, and LLM-as-a-judge quality assurance.

---

## System Architecture & Pipeline

```
                       Customer Tweet
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   [Intent Classifier]              [FAISS RAG Retrieval]
  (Claude Haiku Few-Shot)        (all-MiniLM-L6-v2 Embeddings)
            │                                 │
            │                         [Grounding Signal]
            │                     (Strong / Moderate / Weak)
            │                                 │
            ├─────────────────────────────────┘
            ▼
   [Escalation Decision]
 (Safety Rules + Urgency + Grounding)
            │
     ┌──────┴─────────────────────────┐
     ▼                                ▼
[Auto-Handle]                    [Escalate]
     │                       (Route to Human Agent)
     ▼
[Grounded Reply Drafter]
 (Defensive Persona Prompt)
     │
     ▼
[LLM-as-a-Judge QA]
 (Rubric: Groundedness, Correctness, Tone, Actionability)
```

---

## Pipeline Components

1. **Data Pipeline (`build_dataset.py`)**: Filters 43,092 raw Spotify tweets from the Customer Support on Twitter dataset down to 33,147 clean customer-brand interaction pairs.
2. **Intent Classification (`classify_intent.py`)**: Few-shot classification into 7 distinct intents (`playback_technical_issue`, `account_login_access`, `billing_subscription`, `cancellation_refund`, `feature_content_question`, `general_complaint_feedback`, `other_uncategorized`).
3. **Retrieval Engine (`retrieve_similar_cases.py`)**: High-speed cosine similarity search over historical resolved cases using `all-MiniLM-L6-v2` and FAISS `IndexFlatIP` with DM-deflection down-weighting and deduplication. Includes a dedicated held-out index to prevent evaluation leakage.
4. **Grounded Reply Drafter (`draft_reply.py`)**: Generates 1-3 sentence SpotifyCares brand replies conditioned on retrieved case grounding (`strong`, `moderate`, `weak`). Employs defensive drafting when grounding is uncertain.
5. **Escalation Engine (`escalate_or_handle.py`)**: Safety-biased routing using intent sensitivity, churn threats, account breach keywords, and grounding quality.
6. **LLM-as-a-Judge & Validation (`judge_reply.py`, `validate_judge.py`)**: Automated 1-5 quality assessment validated against blind human scoring (90.0% within $\pm 1$ point agreement).

---

## Evaluation & Benchmark Results

Evaluated on the 150-example hand-labeled golden dataset (`data/golden_set_labeled.csv`) using the independent held-out index:

| System / Model | Intent Accuracy | Escalation Precision | Escalation Recall | False Auto-Handles (Safety Risk) | LLM Judge Score (1-5) |
|---|:---:|:---:|:---:|:---:|:---:|
| **AI Agent Pipeline (Claude Haiku)** | **96.0%** | **70.6%** | **100.0%** | **0 (0.0%)** | **3.61 / 5.00** |
| **Simple Baseline (Keyword + 1-NN)** | 21.3% | 53.8% | 98.6% | 1 (1.4%) | 2.29 / 5.00 |
| **Trivial Baseline (Majority/Canned)** | 23.3% | 0.0% | 0.0% | 72 (100.0%) | 1.49 / 5.00 |

---

## Getting Started

### Prerequisites
- Python 3.10+
- Anthropic API Key (in `.env`)

### Installation
```bash
git clone <repo-url>
cd Hiver
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # or install anthropic, faiss-cpu, sentence-transformers, pandas, numpy, python-dotenv
```

### Running Evaluation
```bash
# Run full three-system benchmark
python eval.py

# Run live golden set evaluation harness
python eval_harness.py --live

# Run LLM-as-a-judge scoring
python judge_reply.py

# Validate judge scores against human annotations
python validate_judge.py --evaluate
```
