## Session Update — 2026-09-22

### Context coming into this session

Picking up from the prior session's handoff notes. State at the time: page-level citations had been added (each PDF page becomes its own `Document` with `metadata={"page": page_num}`), an `eval.py` harness had been built using GPT-4o-mini as an LLM judge (checks both factual accuracy and citation accuracy against the full source text), and adding those citations had caused a regression — eval score dropped from 8/10 (Run 2, no citations) to 7/10 (Run 3, with citations).

**Known failing questions going into this session (7/10 baseline):**
- Q1 (medications) — dexmedetomidine hallucination had resurfaced (same conditional-vs-administered error as the original 2026-09-18 finding)
- Q3 (primary diagnosis) — new regression. The model was pulling the "Primary ICD-10 Code" field (page 6: "A40.3 — Sepsis due to Streptococcus pneumoniae") instead of the actual "Primary ICU Diagnosis" field (page 1: "Septic shock secondary to community-acquired pneumonia (CAP) with ARDS, acute kidney injury (AKI) on CKD, and multi-organ dysfunction"). Root cause: per-page chunking changed how LlamaIndex splits text, and the ICD code chunk shares the literal word "primary" with the question, so it out-competes the real diagnosis chunk.
- Q5 (abnormal labs) — lactate 1.6 mmol/L (explicitly stated in the note as "normalized from 5.2 on admission") still being flagged as abnormal; WBC 18.4 (genuinely abnormal, down from 22.6) still being missed

The planned first step, per the prior session's TODO, was to bump `SIMILARITY_TOP_K` from 4 to 6 and rerun eval to see if wider retrieval alone fixed Q3.

---

### What we tested and found

**Step 1 — top_k alone (4 → 6), no other changes.**
Reran `eval.py`. Result: still 7/10, identical three failures. Confirmed via Checkpoint 2 output that both the ICD-10 chunk (page 6) and the actual diagnosis chunk (page 1) were now both being retrieved in the top-6 — so the retrieval problem was fixed, but the model still chose the wrong one. This told us definitively that all three failures were **generation-side problems, not retrieval-side problems**. Chunk coverage was no longer the bottleneck; which chunk the LLM decided to trust was.

**Step 2 — diagnosed root cause across all three failures.**
Pulled the actual PDF and checked ground truth against each failing answer, line by line, rather than trusting the eval judge's verdict alone. Found one shared root cause underneath three different symptoms: **the model was anchoring on directional/trend language and lexical keyword overlap instead of reasoning about current status.**
- Dexmedetomidine: trend word "consider" → treated as administered
- Lactate: trend phrase "normalized" → still flagged abnormal despite being explicitly stated as back in normal range
- Diagnosis: lexical overlap on "primary" → wrong field selected over the field whose *label* actually matched the question's intent

### What we changed

**New file: `prompts.py`**
Built a custom LlamaIndex `PromptTemplate` (`CLINICAL_QA_TEMPLATE`) that overrides the default generic QA prompt with clinical-specific reading rules:
1. Current status only — don't report considered/planned/discontinued/held items as administered; judge lab values by their current number, not by trend language
2. Prefer the field matching the question's clinical intent over the field with the most keyword overlap (diagnosis field vs. billing/ICD field)
3. Reason against normal reference ranges for labs, not against adjectives like "improving"
4. Cite page + field/section for every fact used
5. Say "not documented" rather than guess

Also added `get_clinical_query_engine()` — a helper that builds the query engine and swaps in this template via `query_engine.update_prompts()`.

**Modified: `query.py`**
One import added, one line changed inside `query_index()`:
- Before: `query_engine = index.as_query_engine(similarity_top_k=SIMILARITY_TOP_K)`
- After: `query_engine = get_clinical_query_engine(index, similarity_top_k=SIMILARITY_TOP_K)`

All three debug checkpoints, error handling, and the interactive loop were left untouched — this was a single, surgical change to only the prompt used during generation.

**Modified: `config.py`**
`SIMILARITY_TOP_K` changed from 4 to 6, kept from Step 1 since wider retrieval — while not sufficient by itself — is still a necessary precondition for the prompt-level fix to have both competing chunks available to disambiguate between.

**`eval.py` required no changes.** It imports `query_index` directly from `query.py` and calls it as-is, so it inherited the new prompt template automatically. This confirmed the value of the original design decision to centralize the pipeline in one function rather than duplicating query logic across files.

**`eval.py` itself was also committed to version control for the first time this session** — it had existed locally but was previously untracked, meaning eval results weren't reproducible by anyone pulling the repo fresh.

---

### Result: Test Run 4 — 2026-09-22 (clinical prompt template + top_k=6)

| # | Question | Run 3 (7/10) | Run 4 | Change |
|---|---|---|---|---|
| 1 | Medications | FAIL | **PASS** | Dexmedetomidine hallucination fixed. New side effect: answer is now narrower and dropped norepinephrine + vasopressin (active vasopressors) from the list — a preexisting omission pattern from Run 2 that's more visible now that dexmedetomidine isn't padding the list |
| 2 | Vital signs | PASS | PASS | No change |
| 3 | Primary diagnosis | FAIL | **PASS** | Now correctly cites "Primary ICU Diagnosis" field (page 1) verbatim instead of the ICD-10 billing code |
| 4 | Follow-up actions | PASS | PASS | No change |
| 5 | Abnormal labs | FAIL | **FAIL** | Lactate 1.6 still incorrectly flagged abnormal despite explicit "normalized" language. New issue surfaced: D-dimer 8,400 (genuinely abnormal) is now omitted entirely |
| 6 | Code status | PASS | PASS | No change |
| 7 | Allergies | PASS | PASS | No change |
| 8 | Ventilator status | PASS | PASS | No change |
| 9 | Family meeting | PASS | PASS | No change |
| 10 | Vasopressor wean plan | PASS | PASS | No change |

**Result: 7/10 → 9/10.** Two of three targeted failures resolved cleanly. Root causes confirmed correct for both (field-label disambiguation, conditional-vs-administered distinction).

### Why Q5 is still failing, specifically

The prompt instructs the model to "reason against normal reference ranges" but doesn't actually supply what those ranges are. The model is still inferring normal/abnormal from adjectives in the source text ("normalized," "down from") rather than comparing the raw number against a real threshold. This works fine when the adjective and the correct classification happen to agree, but fails exactly when they don't — lactate is described with improving language ("normalized") while still needing to be excluded, which requires the model to know 1.6 mmol/L is inside normal range, not infer it from tone.

Next fix, not yet made: supply actual reference ranges in the prompt (e.g., "lactate normal <2.0 mmol/L, WBC normal 4–11 x10⁹/L, D-dimer normal <500 ng/mL") so the model has real numbers to compare against instead of inferring normalcy from surrounding language.

### New gap surfaced (not previously tracked)

Q1's answer, now that dexmedetomidine is correctly excluded, is missing norepinephrine and vasopressin — both active, currently-running vasopressors, unambiguously "given." This wasn't caught by the LLM-judge (it only checks whether *cited* items are accurate, not whether the list is *complete*), which is worth flagging as a known limitation of the current eval design, not a new regression from this session's fix.

---

### Committed this session

Commit `5f55eaf` — "Add clinical-aware QA prompt template; bump retrieval to top_k=6"
Files changed: `config.py` (top_k 4→6), `query.py` (prompt template swap), `prompts.py` (new), `eval.py` (tracked for the first time)

---

### Open issues for next pass

- [ ] Q5: give the model real reference ranges instead of relying on inferred "normal" from trend language — needs prompt update, not retrieval or architecture change
- [ ] Q5: D-dimer 8,400 now omitted — check whether this is a one-off side effect of the current prompt template or a resurfacing of the original "abnormal values get missed even when the right chunk is retrieved" pattern from Run 1/2
- [ ] Q1: vasopressors (norepinephrine, vasopressin) dropped from medication list — decide whether this needs prompt language distinguishing "medications" broadly from a narrower reading, or whether it's a retrieval/chunking issue worth checking against Checkpoint 2 output specifically
- [ ] Test current fixes against a second, different clinical note — right now every fix has only been validated against one synthetic document, so it's unconfirmed whether the field-label-preference rule and conditional-vs-administered rule generalize or are overfit to this note's specific layout
- [ ] `answers.txt` exists locally, untracked — decide if it's meant to be version-controlled eval history or local scratch output; either commit it properly or add to `.gitignore`
- [ ] Build `app.py` (Streamlit UI) — file upload, question box, answer panel, source citation
- [ ] Deploy publicly (Hugging Face Spaces) by end of November
