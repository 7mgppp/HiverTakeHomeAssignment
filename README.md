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
| **AI Agent Pipeline (Claude Haiku)** | **96.7%** | **72.7%** | **100.0%** | **0 (0.0%)** | **3.61 / 5.00** |
| **Simple Baseline (Keyword + 1-NN)** | 21.3% | 53.8% | 98.6% | 1 (1.4%) | 2.29 / 5.00 |
| **Trivial Baseline (Majority/Canned)** | 23.3% | 0.0% | 0.0% | 72 (100.0%) | 1.49 / 5.00 |

---

## Getting Started

### Prerequisites
- Python 3.10+
- Anthropic API Key (in `.env`: `ANTHROPIC_API_KEY=sk-ant-...`)

### Installation
```bash
git clone https://github.com/7mgppp/HiverTakeHomeAssignment.git
cd HiverTakeHomeAssignment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Data Setup

1. **Download Kaggle Dataset**:
   Download `twcs.csv` from the Kaggle [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset and place `twcs.csv` in the project root.

2. **Clean & Build Dataset**:
   Filter the raw tweets down to clean, non-deflection Spotify customer-brand pairs:
   ```bash
   python build_dataset.py
   ```
   *(Produces `data/spotify_pairs_clean.csv` and sample `data/spotify_pairs_sample.csv`)*

3. **Build Held-Out Retrieval Index (Leakage-Free)**:
   Generate the clean, held-out FAISS index excluding golden-set queries:
   ```bash
   python build_heldout_index.py
   ```

---

## Single-Message Pipeline Execution

Run the complete triage and response pipeline on any individual customer message:

```bash
# Process a single customer tweet
python main.py "My Spotify keeps crashing whenever I try to shuffle a playlist"

# Or run interactively
python main.py
```

### Python SDK Usage
```python
from main import process_customer_message

result = process_customer_message("I got charged twice for family plan this month!")
print(result["intent"])        # "billing_subscription"
print(result["decision"])      # "escalate"
print(result["draft_reply"])   # "Hey! We'd be glad to look into this..."
```

---

## Running Evaluation & Benchmarks

> [!NOTE]
> **Precomputed Evaluation Artifacts**:
> All evaluation reports and scored datasets are already committed to the repository. Reviewers can inspect the complete results directly without needing an API key or running anything live:
> - **[`data/evaluation_report.md`](data/evaluation_report.md)**: Detailed breakdown of intent accuracy (96.7%), confusion matrix, escalation safety metrics (100% recall, 0% false auto-handles), and the held-out index leakage audit.
> - **[`data/golden_set_labeled.csv`](data/golden_set_labeled.csv)**: All 150 hand-annotated golden-set messages with human intent, escalation ground truth, and reasoning.
> - **[`data/judge_scores.csv`](data/judge_scores.csv)**: All 150 generated draft replies with 1–5 LLM judge quality scores and 1-2 sentence rationales.
> - **[`data/human_judge_sample.csv`](data/human_judge_sample.csv)**: 30-message blind human validation sample with agreement statistics ($\kappa = 0.3390$, 90.0% within $\pm 1$ pt).
> - **[`data/baseline_comparison_report.md`](data/baseline_comparison_report.md)**: Side-by-side benchmark comparison of AI Agent vs. Simple Baseline vs. Trivial Baseline.

If you wish to re-execute or verify the evaluation live:

```bash
# 1. Run full three-system benchmark (AI Agent vs Simple vs Trivial)
python eval.py

# 2. Run live golden set evaluation harness (Intent & Safety Escalation)
python eval_harness.py --live

# 3. Run LLM-as-a-judge scoring across golden-set drafts
python judge_reply.py

# 4. Validate judge scores against 30 human ground-truth ratings
python validate_judge.py --evaluate
```
