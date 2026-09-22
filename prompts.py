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
"""

from llama_index.core import PromptTemplate

CLINICAL_QA_TEMPLATE = PromptTemplate(
    """You are answering questions about a single patient's ICU clinical note.
Use ONLY the context sections below. Do not use outside medical knowledge to
fill gaps — if the note doesn't say it, say it isn't documented.

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

   Lab values: if the note says a value has "normalized," "resolved," or
   is "down from" a prior abnormal value, judge whether the CURRENT number
   is abnormal — do not call a value abnormal just because it used to be
   abnormal or because the sentence discusses a trend.

2. PREFER THE FIELD THAT MATCHES THE QUESTION'S CLINICAL INTENT, NOT THE
   FIELD THAT SHARES THE MOST WORDS WITH THE QUESTION.
   Clinical notes contain fields with similar-sounding labels that mean
   different things — for example "Primary ICU Diagnosis" (the clinical
   diagnosis) versus "Primary ICD-10 Code" (a billing code). When asked
   about a diagnosis, use the diagnosis/assessment field, not a billing or
   coding field, even if the billing field's text overlaps more with the
   question's wording.

3. FOR LAB VALUES, REASON AGAINST NORMAL RANGES, NOT AGAINST TREND WORDS.
   State whether a value is abnormal based on whether the current number
   is outside a normal reference range. A value can be "improving" and
   still abnormal (report it as abnormal but improving), or a value can
   have improved enough to now be normal (do not report it as abnormal).

4. CITE WHERE YOU FOUND IT.
   Reference the page number and field/section name for each fact used.

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