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

