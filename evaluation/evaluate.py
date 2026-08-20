"""
RAG evaluation framework.

Two-stage pipeline, run standalone (needs the active LLM provider's API
key configured, e.g. GROQ_API_KEY):

  1. generate_qa_pairs() — samples chunks from each department and asks
     the LLM to write a question a real employee in that department might
     ask, plus the reference answer grounded in that chunk. Saved to
     qa_pairs.csv.

  2. evaluate() — for each QA pair, runs it through the real /chat-style
     pipeline (retriever + generate_answer) as if asked by a user with
     access to that department, then asks the LLM to score the actual
     answer against the reference on three axes:
        - faithfulness: is the answer grounded in the retrieved context
          (not hallucinated)?
        - relevance: does it actually address the question asked?
        - conciseness: is it direct, without unnecessary padding?
     Each scored 1-5. Saved to evaluation_results.csv.

Run with:
    python -m evaluation.evaluate
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import ROLE_PERMISSIONS
from backend.ingest import load_all_chunks
from backend.retriever import get_retriever
from backend.llm import generate_answer
from backend import llm_client

OUT_DIR = Path(__file__).resolve().parent
QA_PAIRS_PATH = OUT_DIR / "qa_pairs.csv"
RESULTS_PATH = OUT_DIR / "evaluation_results.csv"

SAMPLES_PER_DEPARTMENT = 3
# A role that has access to each department, used to run the RAG pipeline
# the same way a real user of that department would experience it.
DEPARTMENT_TO_SAMPLE_ROLE = {
    "engineering": "engineering",
    "finance": "finance",
    "marketing": "marketing",
    "hr": "hr",
    "general": "employee",
}


def _check_llm_available() -> None:
    if not llm_client.api_key_configured():
        raise RuntimeError(
            f"{llm_client.PROVIDER.upper()}_API_KEY must be set to run the evaluation framework."
        )


def generate_qa_pairs() -> list[dict]:
    _check_llm_available()

    all_chunks = load_all_chunks()
    by_department: dict[str, list] = {}
    for c in all_chunks:
        by_department.setdefault(c.department, []).append(c)

    qa_pairs = []
    for department, chunks in by_department.items():
        sample = random.sample(chunks, min(SAMPLES_PER_DEPARTMENT, len(chunks)))
        for chunk in sample:
            prompt = (
                "Given this internal company document excerpt, write ONE realistic "
                "question a company employee might ask an internal assistant, "
                "answerable from this excerpt, plus a concise reference answer.\n\n"
                f"EXCERPT:\n{chunk.text}\n\n"
                'Respond as JSON only: {"question": "...", "reference_answer": "..."}'
            )
            text = llm_client.chat([{"role": "user", "content": prompt}], max_tokens=300).strip()
            text = text.strip("`").removeprefix("json").strip()
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            qa_pairs.append({
                "department": department,
                "source": chunk.source,
                "question": parsed.get("question", ""),
                "reference_answer": parsed.get("reference_answer", ""),
            })

    with open(QA_PAIRS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["department", "source", "question", "reference_answer"])
        writer.writeheader()
        writer.writerows(qa_pairs)

    print(f"Wrote {len(qa_pairs)} QA pairs to {QA_PAIRS_PATH}")
    return qa_pairs


def _score_answer(question: str, reference: str, actual: str, context: str) -> dict:
    prompt = (
        "Score the ACTUAL_ANSWER against the QUESTION, REFERENCE_ANSWER, and "
        "RETRIEVED_CONTEXT on three axes, each 1-5 (5 = best):\n"
        "- faithfulness: is it grounded in RETRIEVED_CONTEXT, with no invented facts?\n"
        "- relevance: does it actually address QUESTION?\n"
        "- conciseness: is it direct, without unnecessary padding?\n\n"
        f"QUESTION: {question}\n\n"
        f"REFERENCE_ANSWER: {reference}\n\n"
        f"RETRIEVED_CONTEXT: {context}\n\n"
        'Respond as JSON only: {"faithfulness": N, "relevance": N, "conciseness": N}'
    )
    text = llm_client.chat([{"role": "user", "content": prompt}], max_tokens=100).strip()
    text = text.strip("`").removeprefix("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"faithfulness": None, "relevance": None, "conciseness": None}


def evaluate(qa_pairs: list[dict] | None = None) -> list[dict]:
    _check_llm_available()
    retriever = get_retriever()

    if qa_pairs is None:
        if not QA_PAIRS_PATH.exists():
            qa_pairs = generate_qa_pairs()
        else:
            with open(QA_PAIRS_PATH, encoding="utf-8") as f:
                qa_pairs = list(csv.DictReader(f))

    results = []
    for pair in qa_pairs:
        department = pair["department"]
        role = DEPARTMENT_TO_SAMPLE_ROLE.get(department, "c-level")
        allowed = ROLE_PERMISSIONS[role]

        chunks = retriever.query(pair["question"], allowed, role, top_k=5)
        actual_answer = generate_answer(
            pair["question"], chunks, {"role": role}, history=[], best_possible_score=1.0
        )
        context_text = "\n".join(c["text"] for c in chunks)

        scores = _score_answer(pair["question"], pair["reference_answer"], actual_answer, context_text)
        results.append({
            "department": department,
            "role_used": role,
            "question": pair["question"],
            "reference_answer": pair["reference_answer"],
            "actual_answer": actual_answer,
            **scores,
        })

    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["department", "role_used", "question", "reference_answer", "actual_answer",
                      "faithfulness", "relevance", "conciseness"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote {len(results)} scored results to {RESULTS_PATH}")
    _print_summary(results)
    return results


def _print_summary(results: list[dict]) -> None:
    metrics = ["faithfulness", "relevance", "conciseness"]
    print("\n--- Average scores by department ---")
    by_dept: dict[str, list[dict]] = {}
    for r in results:
        by_dept.setdefault(r["department"], []).append(r)
    for dept, rows in by_dept.items():
        avgs = []
        for m in metrics:
            vals = [row[m] for row in rows if row[m] is not None]
            avgs.append(f"{m}={sum(vals)/len(vals):.2f}" if vals else f"{m}=N/A")
        print(f"  {dept}: " + ", ".join(avgs))


if __name__ == "__main__":
    evaluate()
