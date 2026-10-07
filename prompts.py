"""
Clinical-specific prompt template for the RAG query engine.

Why this exists: LlamaIndex's default QA prompt is generic ("answer the
question using the context below"). Testing against Sample_ICU_note.pdf
surfaced three repeatable failure modes that a generic prompt can't catch:

1. Conditional/planned actions reported as if administered
   (dexmedetomidine: note says "consider... not today", model said "given")
2. A lexically-similar but wrong field chosen over the correct one
   (model picked "Primary ICD-10 Code" over "Primary ICU Diagnosis" because
   both contain the word "primary")
3. A value described as normalized/resolved still flagged as abnormal
   (lactate 1.6, explicitly "normalized from 5.2", still listed as abnormal)

All three share one root cause: the model anchors on keyword overlap and
trend language instead of reasoning about current status vs. history/plan.
This prompt explicitly instructs against that.

Design decision (2026-09-22): rule 3 does NOT use an external reference-range
table. An earlier version gave the model a hardcoded list of normal lab
ranges (lactate < 2.0, WBC 4.5-11.0, etc.). That fixed the one failing test
case, but it meant the system's real behavior was "PDF plus facts a
developer typed into a prompt from memory" rather than "reasoning from the
document." That doesn't generalize to a document the system hasn't seen,
and it isn't something a nurse could audit or trust the provenance of.

Instead, rule 3 asks the model to rely on the document's OWN interpretive
language (e.g. "normalized," "elevated," "within normal limits") as the
source of truth. If the document gives a bare number with no interpretation
attached, the system says so honestly instead of guessing against outside
knowledge. This keeps every answer traceable to the document, which is the
actual point of a citation-first clinical tool: a nurse can verify a claim
against the source, but can't verify a claim that came from the model's
internal assumptions about what "normal" means.
"""

from llama_index.core import PromptTemplate

CLINICAL_QA_TEMPLATE = PromptTemplate(
    """You are answering questions about a single patient's clinical note.
Use ONLY the context sections below. Do not use outside medical knowledge to
fill gaps — if the note doesn't say it, say it isn't documented. This
includes lab reference ranges: do not judge whether a value is high, low,
or normal based on your own general medical knowledge. Only use what the
document itself says about that value.

Context sections from the note (each may include a page number and field
label):
---------------------
{context_str}
---------------------

Follow these rules when reading the context:

1. CURRENT STATUS, NOT HISTORY OR PLANS.
   Medications: only list a medication as "given" or "administered" if the
   note states it was actively administered. If the note says a medication
   was "considered," "discontinued," "held," "not today," or is a future/
   conditional option ("if X, then consider Y"), do NOT list it as given —
   name it separately as considered/discontinued/held if the question asks
   about the full medication picture.

2. PREFER THE FIELD THAT MATCHES THE QUESTION'S CLINICAL INTENT, NOT THE
   FIELD THAT SHARES THE MOST WORDS WITH THE QUESTION.
   Clinical notes contain fields with similar-sounding labels that mean
   different things — for example "Primary ICU Diagnosis" (the clinical
   diagnosis) versus "Primary ICD-10 Code" (a billing code). When asked
   about a diagnosis, use the diagnosis/assessment field, not a billing or
   coding field, even if the billing field's text overlaps more with the
   question's wording.

3. FOR LAB VALUES, TRUST ONLY THE DOCUMENT'S OWN INTERPRETIVE LANGUAGE —
   NEVER YOUR OWN KNOWLEDGE OF NORMAL RANGES.
   Clinical notes often say whether a value is normal or abnormal directly,
   using words like "normalized," "elevated," "low," "within normal
   limits," "critical," or by explicitly flagging it. Use that language as
   the answer.

   - If the note says a value has "normalized," "resolved," or returned to
     baseline, treat it as CURRENTLY normal — even if the same sentence
     mentions a worse prior value. Do not call it abnormal just because it
     used to be abnormal.
   - If the note explicitly calls a value elevated, low, critical, or
     abnormal, report it as abnormal, even if the note also says it's
     improving or trending in the right direction (report both: abnormal,
     but improving).
   - If the note gives a bare number with NO interpretive language at all
     (no adjective, no flag, nothing saying whether it's high/low/normal),
     do NOT guess based on outside knowledge of what a normal range should
     be. Instead say something like: "The note lists [value] but does not
     state whether this is normal or abnormal — verify against a reference
     range."

4. CITE WHERE YOU FOUND IT.
   Reference the page number and field/section name for each fact used, so
   the answer can be checked against the source document directly.

5. IF NOT DOCUMENTED, SAY SO.
   Do not guess or infer information the note doesn't contain.

Question: {query_str}
Answer:"""
)


def get_clinical_query_engine(index, similarity_top_k: int = 6):
    """
    Build a query engine using the clinical-aware prompt template.

    Args:
        index: VectorStoreIndex built from the clinical note
        similarity_top_k: number of chunks to retrieve (6 confirmed to
            surface both competing chunks in the diagnosis case, so this
            is a floor, not a fix by itself — the prompt template is the
            actual fix)

    Returns:
        A LlamaIndex query engine with the clinical prompt applied
    """
    query_engine = index.as_query_engine(similarity_top_k=similarity_top_k)

    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": CLINICAL_QA_TEMPLATE}
    )

    return query_engine