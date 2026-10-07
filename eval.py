import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from config import SIMILARITY_TOP_K
from query import build_index, query_index

load_dotenv()
client = OpenAI()

JUDGE_MODEL = "gpt-4o-mini"
QUESTIONS_FILE = Path("evals/questions.json")
RESULTS_DIR = Path("evals/results")


def load_questions(path=QUESTIONS_FILE):
    with open(path) as f:
        return json.load(f)


def judge_answer(source_text, question, answer, cited_pages, expected=""):
    expected_block = ""
    if expected:
        expected_block = f"\nKEY FACTS THE ANSWER MUST GET RIGHT (written by a human reviewer):\n{expected}\n"

    prompt = f"""You are a strict fact-checker reviewing an AI's answer to a question about a clinical document.

SOURCE DOCUMENT:
{source_text}

QUESTION: {question}
AI'S ANSWER: {answer}
PAGES CITED AS SOURCES: {cited_pages}
{expected_block}
Check two things:
1. FACTUAL ACCURACY — is the answer supported by the source document? Flag anything stated as true that isn't clearly supported, and anything important that was available in the source but missing from the answer.
2. CITATION ACCURACY — do the cited pages plausibly contain the information used in the answer, based on the source document?

Respond in exactly this format:
VERDICT: PASS or FAIL
REASON: one sentence explaining why, covering both factual accuracy and citation accuracy
"""
    result = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return result.choices[0].message.content


def save_results(results, passed, total):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_path = RESULTS_DIR / f"run_{timestamp}.json"
    summary = {
        "timestamp": timestamp,
        "judge_model": JUDGE_MODEL,
        "similarity_top_k": SIMILARITY_TOP_K,
        "passed": passed,
        "total": total,
        "score": round(passed / total, 2) if total else 0,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return out_path


def run_eval():
    questions = load_questions()

    print("Loading and indexing document...")
    index, documents = build_index()
    source_text = "\n".join(doc.text for doc in documents)

    passed = 0
    total = len(questions)
    results = []

    print(f"\nRunning {total} test cases with LLM-as-judge (accuracy + citation)...\n")

    for i, item in enumerate(questions, 1):
        question = item["question"]
        expected = item.get("expected", "")

        response = query_index(index, question)
        if response is None:
            print(f"[{i}] FAIL — no response returned\n  Q: {question}\n")
            results.append({"id": item["id"], "question": question, "passed": False,
                            "answer": None, "cited_pages": [], "verdict": "No response returned"})
            continue

        cited_pages = sorted(set(
            node.metadata.get("page", "unknown") for node in response.source_nodes
        ))

        verdict = judge_answer(source_text, question, str(response), cited_pages, expected)
        did_pass = "VERDICT: PASS" in verdict

        print(f"[{i}] {question}")
        print(f"    Answer: {response}")
        print(f"    Cited pages: {cited_pages}")
        print(f"    {verdict}\n")

        if did_pass:
            passed += 1

        results.append({
            "id": item["id"],
            "question": question,
            "expected": expected,
            "answer": str(response),
            "cited_pages": cited_pages,
            "verdict": verdict,
            "passed": did_pass,
        })

    out_path = save_results(results, passed, total)
    print(f"\n--- Results: {passed}/{total} passed ---")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    run_eval()