# SpotifyCares AI Agent — Decision Log

## Step 1: Data Pipeline

### Filter funnel (build_dataset.py)
| Stage | Rows | Removed | Reason |
|-------|------|---------|--------|
| Raw SpotifyCares pairs | 43,092 | — | Initial filter from twcs.csv |
| After DM deflection removal | 33,303 | 9,789 | Regex list: "DM us", "send us a direct message", etc. |
| After non-English removal | 33,264 | 39 | >85% ASCII-letter heuristic |
| After short message removal | 33,148 | 116 | Customer message < 15 chars |
| After bare-link-only removal | 33,147 | 1 | Reply is only a URL |
| **Final dataset** | **33,147** | | Written to data/spotify_pairs_clean.csv |

Sampled 2,000 rows → `data/spotify_pairs_sample.csv` (committed to git for reproducibility).

---

## Step 2: Intent Classifier

### Intent taxonomy (fixed — do not rename)
7 labels: `playback_technical_issue`, `account_login_access`, `billing_subscription`,
`cancellation_refund`, `feature_content_question`, `general_complaint_feedback`,
`other_uncategorized`.

### Backend API decisions

| Date | Decision | Reason |
|------|----------|--------|
| 2026-09-09 | Started with OpenAI GPT-4o-mini | Assignment allows any LLM API; GPT-4o-mini is cost-effective |
| 2026-09-09 | Switched to Google Gemini (gemini-3.6-flash) | OpenAI account had zero credits (credit_balance_exhausted) |
| 2026-09-10 | Dropped Gemini | Free tier = 20 req/day; exhausted during debugging; quota reset at midnight Pacific which blocked the 30-msg smoke test |
| 2026-09-11 | **Switched to Anthropic Claude claude-haiku-4-5-20251001** | Paid account ($5 credits); no rate-limit blocking; model string confirmed from official docs; Haiku is cheapest/fastest Claude tier |

### Decision: chose Anthropic Claude Haiku over cloud free-tier models

**Chosen model:** `claude-haiku-4-5-20251001`
**Reason:** Paid Anthropic account eliminates the free-tier rate-limit problem entirely.
The 20 req/day cap on Gemini's free tier was blocking the 30-message smoke test and would
have made the 150-250 example golden-set evaluation (Step 5) completely infeasible in a
single session. Claude Haiku is optimised for classification latency and cost.

**Tradeoff acknowledged:** chose local Ollama model over cloud API to avoid free-tier rate
limits blocking the 150-250 example golden-set evaluation; tradeoff is somewhat lower
classification quality vs GPT-4/Gemini-class models, judged acceptable for a 7-intent
classification task with few-shot prompting.

**Estimated cost:**
- 30-msg smoke test ≈ $0.04
- 150-250 example golden-set evaluation ≈ $0.20-0.30
- Full 33,147-pair batch (if needed) ≈ $48 — use data/spotify_pairs_sample.csv for bulk work

**Model string confirmation source:**
https://docs.anthropic.com/en/docs/about-claude/models/all-models — checked 2026-09-11.

---

## Steps 3 & 4: Retrieval & Grounded Drafting

- **Local Embeddings**: Chose `sentence-transformers/all-MiniLM-L6-v2` + FAISS `IndexFlatIP` (L2 normalized) over calling an embedding API. Caches embeddings locally (`data/spotify_embeddings.npy` & `data/spotify_faiss.index`) to reduce query latency to <5ms and cost to $0.00.
- **Grounding Quality Signal**: Formulated three tiers (`strong`, `moderate`, `weak`) based on cosine similarity and deflection/fragment filtering. Directs Claude Haiku to strictly commit to grounded facts when `strong`, and fallback to safe clarifying questions when `weak`.

---

## Steps 5 & 6: Escalation Decision & Golden-Set Evaluation

### Decision: Safety-Biased Escalation Rules

Chose to bias escalation rules toward safety after golden-set eval showed 18.1% false-auto-handle rate (customers with real churn/security risk being auto-handled). Added churn-threat and security-breach keyword rules — this eliminated all false auto-handles (0%) but increased false-escalation rate from ~1% to 21.8% (precision dropped 0.983→0.809). Accepted this tradeoff: in a support context, an unnecessary escalation costs a few minutes of human agent time, while a missed escalation on a real churn/security risk could cost a customer relationship or enable real harm. Also note: intent accuracy showed minor variance between live eval runs (97.3%→96.0%) despite no changes to the classification logic — likely LLM non-determinism rather than a real regression, worth a second run to confirm before final report.

---

## Step 7: LLM-as-a-Judge Validation & Failure Analysis

### Limitation: Lack of Fact-Verification on Numeric Claims
Row 131 revealed the reply-drafting pipeline has no fact-verification step for specific numeric claims (e.g. download limits). Draft stated precise numbers with high confidence with no grounding source for them — a real risk if a customer acts on an incorrect figure. Worth flagging as a limitation in the failure analysis section.

### Systematic Divergence: LLM-as-a-Judge vs. Human Evaluator
Analysis of the 30-row validation sample revealed two systematic patterns where the LLM judge and human ratings diverge:
1. **Scope Blindness (Over-rewarding off-scope fluency)**: The judge has no domain boundary for "what is out-of-scope for a consumer support agent" (e.g., ID 147 job inquiry scored 4 by Judge vs. 2 by Human). The judge rewards a plausible, friendly deflection even when the bot should not have attempted to handle the query at all.
2. **Actionability Asymmetry (Over-penalizing safe generality)**: The judge penalizes safe-but-generic responses much more harshly than human raters when the underlying issue is urgent or emotionally charged (e.g., ID 70 hacked email, ID 125 refund complaint). The judge demands concrete diagnostic next steps or direct resolution rather than accepting a polite holding reply.
3. **Net Evaluation Bias**: Consequently, LLM judge scores tend to understate quality on cautious, generic-but-safe replies and overstate quality on out-of-scope but fluent replies.

---

## Step 8: Data Leakage Audit & Held-out Index Benchmark

### Decision: Rebuilding Retrieval Index on Held-out Corpus
Identified potential data leakage — golden-set queries were retrievable from the same corpus used for grounding. Rebuilt retrieval index excluding golden-set source tweets and re-ran full evaluation to get honest, leakage-free numbers. See before/after comparison in evaluation_report.md.

Our initial evaluation showed 88.7% strong grounding and a corresponding high judge score — but this was partly inflated by data leakage (golden-set queries were retrievable from the same corpus used for retrieval). After rebuilding a held-out index, strong grounding dropped to 46.0%, a realistic reflection of grounding quality on genuinely unseen queries. Critically, reply quality only dropped 0.04 points and safety metrics (100% escalation recall, 0% false auto-handles) were fully preserved — demonstrating that the system's defensive-drafting and escalation-on-weak-grounding design choices generalize correctly, not just the surface-level retrieval scores.

---

## Step 9: Negation-Aware Churn Guardrails

### Decision: Negation Prefix Filtering for Churn-Threat Detection
Spot-checking revealed a negation-blindness bug: queries like "I don't want to unsubscribe but..." (ID 130) triggered false escalations solely because the keyword "unsubscribe" appeared. Added regex prefix negation checks (e.g. `don't`, `do not`, `not`, `never`, `no plans to`, `wouldn't`, `hate to`) preceding churn keywords within 1-4 words. This reduced the false-escalation rate from 38.5% (30 cases) down to 34.6% (27 cases), successfully resolving false escalations on feedback inquiries like ID 130 while maintaining 100% recall on genuine churn threats and 0% false auto-handles.


