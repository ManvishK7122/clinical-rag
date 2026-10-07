# Clinical RAG — Project Log

## Overview

A RAG (Retrieval-Augmented Generation) tool that lets a nurse upload a clinical note (PDF) and ask questions about it in plain language, getting back an accurate, sourced answer instead of having to read the whole document manually.

**Stack:** Python, LlamaIndex, OpenAI API, pypdf, Streamlit (UI not yet built)

**Repo structure:**
```
clinical-rag/
├── .env              # API keys (never committed)
├── .gitignore         # excludes venv/, .env, data/, __pycache__
├── data/              # clinical note PDFs (excluded from git)
├── ingest.py          # loads PDF, builds vector index
├── query.py           # loads PDF, builds index, answers questions in a loop
└── app.py             # Streamlit UI (not yet built)
```

---

## What we've built so far (Stage 1A + 1B)

### The pipeline
1. **Load** — `pypdf.PdfReader` extracts raw text from the clinical note PDF
2. **Chunk + embed** — LlamaIndex's `VectorStoreIndex.from_documents()` splits the text into chunks and converts each into a vector (a numeric fingerprint of its meaning) using OpenAI's embedding model
3. **Query** — `index.as_query_engine()` takes a question, finds the most semantically relevant chunks, and sends them + the question to OpenAI to generate an answer

### Key fix along the way
Our first attempt used `SimpleDirectoryReader` to load the PDF, which doesn't have a real PDF parser installed by default — it read the raw binary structure of the PDF file (`%PDF-1.7...`) instead of the actual text. Switching to `pypdf.PdfReader` fixed this by properly extracting the readable text.

### Three debug checkpoints (built into `query.py`)
This is the habit that matters more than the code itself — isolating exactly where a RAG pipeline is failing instead of treating it as one black box:

- **Checkpoint 1 — Document text:** print the first ~300 characters after loading, confirm it's real readable text, not binary garbage
- **Checkpoint 2 — Retrieved chunks:** print the chunks the retriever actually pulled for a given question, with their similarity scores, before trusting the final answer
- **Checkpoint 3 — Final answer:** the actual response returned to the user

---

## First real test result

**Question:** "What medications was this patient given?"

**Answer returned:** ceftriaxone, vancomycin, pip-tazo, argatroban, insulin, fentanyl, midazolam, dexmedetomidine, norepinephrine, vasopressin, pantoprazole

**Cross-checked against the source PDF (a synthetic septic shock / ARDS / AKI ICU note):**

| Medication | Verdict | Notes |
|---|---|---|
| Ceftriaxone | ✅ Correct | Day 4, de-escalated from vanc + pip-tazo |
| Vancomycin | ✅ Correct | Prior therapy, now discontinued |
| Pip-tazo | ✅ Correct | Prior therapy, now discontinued |
| Argatroban | ✅ Correct | Replacing heparin, HIT precaution |
| Insulin | ✅ Correct | Infusion, 2.5 units/hr |
| Fentanyl | ✅ Correct | Sedation |
| Midazolam | ✅ Correct | Sedation |
| Norepinephrine | ✅ Correct | Vasopressor |
| Vasopressin | ✅ Correct | Vasopressor |
| Pantoprazole | ✅ Correct | Stress ulcer prophylaxis |
| **Dexmedetomidine** | ⚠️ **Misleading** | Note says "consider dexmedetomidine... **not today**" — this is a *planned/conditional* med, not one actually administered. The model presented it as given. |

**Result: 10/11 fully correct, 1 subtle hallucination** — not a fabrication from nothing, but a real and dangerous type of error in healthcare: blurring "being considered" with "actually administered."

---

## What this tells us

The core retrieval pipeline works. Chunking and embedding are finding genuinely relevant sections of the document. The failure mode we found isn't random noise — it's a specific, repeatable category of error (conflating a conditional/planned action with a completed one) that's worth testing for deliberately.

---

## How to log tests going forward

Every time you test a question against a document, log it. This becomes your evaluation dataset — the evidence that the system works, and the record of exactly where it doesn't yet.

### Suggested test log format

Create a file `test_log.md` (or a spreadsheet if you prefer) with one row per test:

| Date | Question | Answer Summary | Verdict | Error Type (if any) | Notes |
|---|---|---|---|---|---|
| 2026-09-18 | What medications was this patient given? | 11 meds listed | Mostly correct | Conditional-vs-administered confusion | Dexmedetomidine was "considered," not given |

**Error type categories to track** (this becomes your taxonomy of failure modes):
- **Fabrication** — mentions something with zero basis in the document
- **Conditional/planned vs. actual** — states a considered or future action as if it already happened (today's finding)
- **Omission** — misses something that was clearly asked for and present in the document
- **Wrong attribution** — correct fact, wrong patient/date/section
- **Retrieval miss** — Checkpoint 2 shows the chunks pulled weren't actually relevant to the question

### The 10-question test protocol

Before trusting the system enough to show anyone, run the same 10 questions against the same document and grade each one. Suggested starter set:

1. What medications was this patient given?
2. What are the patient's current vital signs?
3. What is the primary diagnosis?
4. What follow-up actions are planned?
5. What lab results are abnormal?
6. What is the code status?
7. Are there any allergies noted? *(good test — if none are in the doc, does it correctly say "not mentioned" rather than guessing?)*
8. What is the ventilator/respiratory status?
9. What did the family meeting cover?
10. What is the plan for weaning vasopressors?

Score each 1 (fully correct), 0.5 (partially correct / minor issue like today's), or 0 (wrong or hallucinated). **Target: 8/10 average before moving to Stage 1C (UI).**

### Why this matters beyond just this project

This log is also your evidence. When you eventually talk to a real nurse, pitch to a health-tech company, or explain this project in an interview, "I tested it against 10 clinical questions and identified a specific hallucination pattern around conditional vs. administered treatments" is a concrete, credible claim — miles ahead of "I built a chatbot that answers questions about medical notes."

---

## Test Run 1 — 2026-09-18 (baseline, similarity_top_k default = 2)

**Setup:** query.py, similarity_top_k=2 (default), Sample ICU note (septic shock / pneumonia / ARDS / AKI)

| # | Question | Score | Verdict |
|---|---|---|---|
| 1 | What medications was this patient given? | 0.5 | Dexmedetomidine listed as given — was only "considered," explicitly "not today" |
| 2 | What are the patient's current vital signs? | 1.0 | Exact match |
| 3 | What is the primary diagnosis? | 0.5 | Said "severe ARDS with pneumonia" — actual diagnosis is "septic shock secondary to CAP with ARDS, AKI, multi-organ dysfunction." Dropped the primary condition entirely, misstated ARDS severity |
| 4 | What follow-up actions are planned? | 1.0 | Comprehensive, matches source exactly |
| 5 | What lab results are abnormal? | 0.5 | Missed WBC 18.4, ALT, bilirubin, PT/INR. Incorrectly listed lactate as abnormal when source explicitly says it "normalized" |
| 6 | What is the code status? | 1.0 | Exact match |
| 7 | Are there any allergies noted? | 1.0 | Correctly said "not noted" — no hallucination |
| 8 | What is the ventilator/respiratory status? | 0.5 | Got physical exam findings but missed actual vent settings (FiO2, PEEP, TV, P/F ratio) |
| 9 | What did the family meeting cover? | 1.0 | Exact match |
| 10 | What is the plan for weaning vasopressors? | 1.0 | Exact match |

**Baseline score: 8.5/10**

### Pattern identified
Every imperfect answer (1, 3, 5, 8) was missing information that existed elsewhere in the document — not fabricating anything. Root cause: retriever was only pulling 2 chunks per query by default, not enough coverage for a document this dense.

---

## Test Run 2 — 2026-09-18 (after fix)

**Change made:** `as_query_engine()` → `as_query_engine(similarity_top_k=4)`
**Reason:** Test more chunks per query to fix the omission pattern from Run 1
**Retested:** the 4 questions that scored 0.5 in Run 1

| # | Question | Run 1 | Run 2 | Change |
|---|---|---|---|---|
| 1 | Medications | 0.5 | 0.75 | Dexmedetomidine hallucination FIXED. New issue: dropped norepinephrine + vasopressin (active vasopressors) from the list |
| 3 | Primary diagnosis | 0.5 | **1.0** | Fixed — now matches source exactly |
| 5 | Abnormal labs | 0.5 | 0.65 | Lactate now correctly framed as "normalized" instead of flatly wrong. Still missing WBC 18.4 and PT/INR |
| 8 | Ventilator/respiratory status | 0.5 | **1.0** | Fixed — full match on all vent settings and ABG values |

**Retested subtotal: 2.0/4 → 3.4/4**
**Updated full-set estimate: 8.5/10 → ~9.1/10**

### Findings
- `similarity_top_k=4` meaningfully improved completeness where the answer was scattered across the document
- Fixing dexmedetomidine surfaced a new, smaller gap (missing active vasopressors) — expected trade-off; always retest the full set after a change, not just the question you targeted
- WBC 18.4 missed in both runs despite the retriever finding the right chunk (same chunk as Hgb/Plt, which it does catch) — this is a reasoning gap, not retrieval. Candidate fix: prompt instruction to extract every abnormal-flagged value from a lab chunk, not just the most clinically prominent ones

### Open issues for next pass
- [ ] Q1: current vasopressors (norepi, vasopressin) dropped from medication list
- [ ] Q5: WBC and PT/INR still missed despite being present in retrieved chunks — needs a prompt-level fix
- [ ] Decide whether the dexmedetomidine-type error needs a prompt-engineering fix (e.g., instructing the model to distinguish administered vs. considered/planned treatments) before Stage 1C
- [ ] Build `app.py` (Streamlit UI) — file upload, question box, answer panel, source citation
- [ ] Deploy publicly (Hugging Face Spaces) by end of November


