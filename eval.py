from query import build_index, query_index
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

QUESTIONS = [
    "What medications was this patient given?",
    "What are the patient's current vital signs?",
    "What is the primary diagnosis?",
    "What follow-up actions are planned?",
    "What lab results are abnormal?",
    "What is the code status?",
    "Are there any allergies noted?",
    "What is the ventilator/respiratory status?",
    "What did the family meeting cover?",
    "What is the plan for weaning vasopressors?",
]

def judge_answer(source_text, question, answer, cited_pages):
    prompt = f"""You are a strict fact-checker reviewing an AI's answer to a question about a clinical document.

SOURCE DOCUMENT:
{source_text}

QUESTION: {question}
AI'S ANSWER: {answer}
PAGES CITED AS SOURCES: {cited_pages}

Check two things:
1. FACTUAL ACCURACY — is the answer supported by the source document? Flag anything stated as true that isn't clearly supported, and anything important that was available in the source but missing from the answer.
2. CITATION ACCURACY — do the cited pages plausibly contain the information used in the answer, based on the source document?

Respond in exactly this format:
VERDICT: PASS or FAIL
REASON: one sentence explaining why, covering both factual accuracy and citation accuracy
"""
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return result.choices[0].message.content

def run_eval():
    print("Loading and indexing document...")
    index, documents = build_index()
    source_text = "\n".join(doc.text for doc in documents)

    passed = 0
    total = len(QUESTIONS)

    print(f"\nRunning {total} test cases with LLM-as-judge (accuracy + citation)...\n")

    for i, question in enumerate(QUESTIONS, 1):
        response = query_index(index, question)
        if response is None:
            print(f"[{i}] FAIL — no response returned\n  Q: {question}\n")
            continue

        cited_pages = sorted(set(
            node.metadata.get("page", "unknown") for node in response.source_nodes
        ))

        verdict = judge_answer(source_text, question, str(response), cited_pages)
        print(f"[{i}] {question}")
        print(f"    Answer: {response}")
        print(f"    Cited pages: {cited_pages}")
        print(f"    {verdict}\n")

        if "VERDICT: PASS" in verdict:
            passed += 1

    print(f"\n--- Results: {passed}/{total} passed ---")

if __name__ == "__main__":
    run_eval()