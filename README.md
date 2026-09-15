# SpotifyCares AI Support Agent

An end-to-end AI-powered customer support triage and response system for Spotify customer inquiries, built with retrieval-augmented generation (RAG), intent classification, safety-biased escalation guardrails, and LLM-as-a-judge quality assurance.

---

## 1. Problem Framing: What "Good" Means

For SpotifyCares Twitter support, the system optimizes for three operational goals in strict priority order:

1. **Safety First (Zero Missed Escalations)**: Never automatically handle a message that should reach a human agent (e.g. security compromises, financial billing disputes, churn threats, high customer distress, or ungrounded technical queries).
2. **Correct Triage**: Accurately classify the customer's core underlying support need into an actionable 7-intent operational taxonomy.
3. **Grounded & Useful Responses**: When auto-handling, draft concise, on-brand 1–3 sentence responses strictly anchored in real historical resolutions without hallucinating URLs, policies, or numeric limits.

### Scope Boundaries & What We Chose NOT to Build
- **No Autonomous Account Mutations**: The system intentionally does **not** execute direct account changes, password resets, or refund transactions. When a customer requires account-level modifications, the system safely gathers necessary details and routes to a human agent.
- **No Unconstrained Generation**: The agent is explicitly forbidden from inventing troubleshooting steps or policies when retrieval context is weak. Under weak grounding, it falls back to cautious diagnostic questions or escalation.
- **No Multi-Turn Stateful Agent Loop**: For Twitter/X first-response triage, complex multi-turn dialog state tracking adds latency and non-deterministic failure points. We opted for a deterministic, explainable single-turn pipeline with strict guardrails.

---

## 2. System Architecture & Pipeline

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

### Core Pipeline Components
1. **Data Pipeline (`build_dataset.py`)**: Filters 43,092 raw Spotify tweets down to 33,154 high-quality customer-brand interaction pairs, stripping DM deflections and non-English text.
2. **Intent Classification (`classify_intent.py`)**: Few-shot classification using Claude Haiku across 7 operational intents (`playback_technical_issue`, `account_login_access`, `billing_subscription`, `cancellation_refund`, `feature_content_question`, `general_complaint_feedback`, `other_uncategorized`).
3. **Retrieval Engine (`retrieve_similar_cases.py`)**: Semantic search using `all-MiniLM-L6-v2` + FAISS `IndexFlatIP` with customer-text deduplication and DM-deflection down-weighting. Operates on a dedicated held-out index to eliminate evaluation leakage.
4. **Escalation Engine (`escalate_or_handle.py`)**: Deterministic, safety-biased guardrails evaluating sensitive intents, account breach language, churn keywords (with negation-awareness), and weak retrieval signals.
5. **Grounded Reply Drafter (`draft_reply.py`)**: Claude Haiku drafter generating 1-3 sentence replies conditioned on retrieval quality tiers (`strong`, `moderate`, `weak`).
6. **LLM-as-a-Judge & Validation (`judge_reply.py`, `validate_judge.py`)**: Automated 1–5 QA evaluator validated against blind human scoring.

---

## 3. Benchmark Results & Baseline Comparison

Evaluated on the 150-example hand-labeled golden dataset (`data/golden_set_labeled.csv`) against the independent held-out index:

| System                                       | Intent Accuracy | Escalation Precision | Escalation Recall | False Auto-Handles | LLM Judge Score |
| -------------------------------------------- | --------------: | -------------------: | ----------------: | -----------------: | --------------: |
| AI Agent Pipeline (Claude Haiku + FAISS RAG) |           96.7% |                72.7% |            100.0% |           0 (0.0%) |     3.61 / 5.00 |
| Simple Baseline (Keyword + 1-NN Verbatim)    |           21.3% |                53.8% |             98.6% |           1 (1.4%) |     2.29 / 5.00 |
| Trivial Baseline (Majority Class + Canned)   |           23.3% |                 0.0% |              0.0% |        72 (100.0%) |     1.49 / 5.00 |

---

## 4. "What is Misleading About My Headline Number?"

The headline **96.7% intent accuracy** and **100% escalation recall** are strong results, but interpreting them as evidence that the system can autonomously resolve 96.7% of customer conversations is misleading:

1. **Classification vs. Full Resolution**: Correctly identifying that a tweet is a `playback_technical_issue` is only the first step. An agent can classify the intent perfectly while still drafting an incomplete or insufficiently specific diagnostic reply.
2. **Quality Remains Substantially Less Solved**: The LLM judge assigns the generated replies an average quality score of **3.61 / 5.00** (with 12% scoring 2/5). While safe and friendly, many responses are generic holding replies rather than one-shot resolutions.
3. **Safety Bias Limits Automation Coverage**: Achieving **100% escalation recall and zero false auto-handles** comes at the expense of a **34.6% false-escalation rate** (precision = 72.7%). The system intentionally routes borderline or weakly grounded cases to humans, trading agent workload for absolute safety.
4. **Curated Golden Set vs. Live Production Traffic**: The 150-example evaluation set was stratified across the 7 intents and enriched with subtle edge cases. It is not an unconstrained, non-stationary live stream of real-world Twitter noise.

> **Operational Takeaway**: The true engineering value is not raw "96.7% accuracy", but rather **high-precision intent triage operating under a safety-biased routing policy with zero observed dangerous auto-handles on held-out evaluation**.

---

## 5. Golden Set Construction & Methodology

The evaluation set consists of **150 hand-labeled examples** drawn from Spotify customer support interactions:

- **Stratified & Edge-Case Sampling**: Rather than sampling purely proportional to raw data frequency (which is dominated by generic playback bugs), the set was stratified across all 7 taxonomy classes (support: 10 to 35 per class) and enriched with difficult edge cases:
  - Sarcastic refund hashtags (`#refund` without billing intent).
  - Subtle churn threats (*"might have to cancel if this persists"*).
  - Negated churn phrases (*"I don't want to unsubscribe but..."*).
  - Security compromises (*"friend's account hacked and email changed"*).
  - Obscure OS/DLL crashes and multi-turn fragments.
- **Manual Labeling Protocol**: Every row was hand-annotated with three fields:
  1. `human_intent`: The true underlying support need.
  2. `human_decision`: Binary routing target (`auto_handle` vs. `escalate`).
  3. `human_reason`: Explicit rationale explaining the decision.
- **Held-Out Index Separation**: All 158 matching and near-duplicate source queries were stripped from `data/spotify_pairs_clean.csv` to build a clean held-out FAISS index (`data/spotify_pairs_heldout.csv`, $N = 32,996$), preventing circular evaluation.

---

## 6. Failure Analysis: Top 5 Real Failures

Detailed trace of representative failure modes identified during evaluation:

### 1. Semantic Overlap: Billing vs. Cancellation vs. Account Access
- **Query (ID 70)**: *"@SpotifyCares I need to cancel my account but my email was hacked and i don't have access to it and I don't want to keep being charged."*
- **Observed Intent**: `billing_subscription` | **Expected Intent**: `cancellation_refund` / `account_login_access`
- **Hypothesis**: Multi-faceted support requests span overlapping domains (cancellation + hacking + billing charges). The model predicts a single top label based on financial keywords rather than detecting multi-intent structure.
- **Proposed Fix**: Implement hierarchical intent classification separating the *account state* from the *requested action*.

### 2. Precise Numeric Hallucination Risk
- **Query (ID 131)**: *"if I cancel my premium account will I still have all of my playlists? Will renew in a month"*
- **Observed Output**: Draft asserted exact numeric limits (*"up to 10,000 songs on 5 devices"*) with high confidence without grounding support.
- **Hypothesis**: For common brand parameters, LLMs fall back to parametric pre-training weights when retrieval context lacks explicit numbers.
- **Proposed Fix**: Add an automated claim/numeric entity-verification post-filter against retrieved evidence.

### 3. Missing Legacy Platform Deprecation Context
- **Query (ID 14)**: *"Cannot find 'API-MS-WIN-CORE-PROCESSTHREADS-L1-1-2.DLL'. I'm using Windows 7 64bit."*
- **Observed Output**: Draft recommended generic app reinstallation without recognizing Windows 7 deprecation.
- **Hypothesis**: Cosine similarity in FAISS favors generic crash-resolution text over specific OS-version keywords.
- **Proposed Fix**: Add metadata-conditioned retrieval filtering on device/OS tags.

### 4. LLM Judge Scope Blindness on Out-of-Scope Queries
- **Query (ID 147)**: *"@SpotifyCares please help me get a job at your NY office. Thanks!"*
- **Observed**: Judge scored **4/5** (rewarding the careers link), while Human scored **2/5** (penalizing handling off-scope inquiries).
- **Hypothesis**: LLM judges reward conversational fluency and link validity rather than operational support scope boundaries.
- **Proposed Fix**: Add explicit out-of-scope penalty rules to the judge prompt rubric.

### 5. False Escalations on Non-Urgent Weak Grounding
- **Query (ID 77)**: *"iP7, 11.0.3... Search UX is atrocious. Sub. $, so why fix it unless I cancel my acct.?..."*
- **Observed**: System escalated due to weak grounding (`0.601`) and rhetorical "cancel" mention.
- **Hypothesis**: Rule-based safety guardrails cannot always distinguish sarcastic feedback from real cancellation threats when retrieval similarity is low.
- **Proposed Fix**: Use sentiment-calibrated intent parsing to differentiate cynical feedback from actionable churn risks.

---

## 7. Decision Log

1. **Spotify as Target Brand**: Selected for high volume (43k+ raw interactions) and rich diversity of technical, billing, and account issues.
2. **7-Intent Taxonomy**: Derived directly from real Spotify support conversation clusters rather than importing generic benchmark taxonomies like Banking77.
3. **Few-Shot Claude Haiku**: Chosen for speed (<1s latency), cost-efficiency, and strong adherence to structured JSON schemas.
4. **Local FAISS + all-MiniLM-L6-v2 Embeddings**: Replaced API-based embeddings with local indexing, cutting retrieval latency to <5ms at $0 cost.
5. **Held-Out Retrieval Corpus**: Explicitly purged golden-set queries from the FAISS database to ensure honest, leakage-free RAG evaluation.
6. **Down-Weighting DM Deflections**: Penalized uninformative "DM us" brand tweets by 0.5× so substantive troubleshooting precedents rank first.
7. **Three-Tier Grounding Signal (`strong`/`moderate`/`weak`)**: Dynamically adjusts drafting prompt temperature and instructions based on retrieved similarity scores.
8. **Defensive Drafting Strategy**: Under weak or moderate grounding, prompts Claude to ask focused diagnostic questions rather than inventing facts.
9. **Safety-Biased Escalation Policy**: Prioritized zero false auto-handles over raw automation rate, accepting a 34.6% false-escalation rate to protect customer relationships.
10. **Negation-Aware Churn Matching**: Added prefix negation checks (`don't want to cancel`, `no plans to unsubscribe`) to prevent false alarms on feedback queries.
11. **LLM-as-a-Judge with Blind Human Validation**: Scored drafts on a 1–5 scale and validated against 30 human ratings (80.0% within $\pm 1$ pt, 50.0% exact match, $\kappa = 0.2537$, MAE = 0.77).
12. **Three-System Benchmark**: Benchmarked against both a Trivial majority baseline and a Simple keyword/1-NN baseline to prove RAG pipeline lift.

---

## 8. What I'd Do With One More Week

1. **Hierarchical Intent & Entity Routing**: Replace flat 7-class classification with a two-level classifier (Domain $\to$ Specific Action) coupled with named-entity extraction (OS, App Version, Device).
2. **Numeric Claim Verification Guardrail**: Build an automated fact-checking layer that regex-scans drafts for numbers, prices, or limits and validates them against verified Spotify knowledge base tables.
3. **Metadata-Filtered Hybrid Retrieval**: Combine FAISS dense embeddings with BM25 sparse keyword search filtered on client platform tags.
4. **Multi-Turn Context Tracking**: Expand the pipeline to process Twitter conversation threads rather than isolated single-turn tweets.
5. **Human-in-the-Loop CRM Integration**: Implement webhook endpoints for Zendesk / Hiver to surface drafted replies as agent suggestions for escalated tickets.

---

## 9. Reproducibility & Execution Time

The complete benchmark and evaluation can be fully reproduced in **under 15 minutes**:

- **Precomputed Artifacts (Instant Inspection, 0 API Calls)**: All metrics, confusion matrices, judge scores, and labeled datasets are committed and viewable directly in `data/`.
- **Live Execution (~8–10 minutes)**: Running `eval.py` or `eval_harness.py --live` across all 150 rows executes live Claude Haiku API calls in ~8 minutes.

---

## 10. Getting Started & Setup

### Prerequisites
- Python 3.10+
- Anthropic API Key (`ANTHROPIC_API_KEY=sk-ant-...` in `.env`)

### Installation
```bash
git clone https://github.com/7mgppp/HiverTakeHomeAssignment.git
cd HiverTakeHomeAssignment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data Setup
1. **Download Kaggle Dataset**: Place `twcs.csv` from [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) in the project root.
2. **Build Clean Dataset & Held-Out Index**:
   ```bash
   python build_dataset.py
   python build_heldout_index.py
   ```

### Single-Message Pipeline Execution
```bash
# Process an individual customer tweet
python main.py "My Spotify keeps crashing whenever I try to shuffle a playlist"

# Interactive CLI mode
python main.py
```

### Running Evaluation & Benchmarks
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
