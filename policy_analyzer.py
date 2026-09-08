"""
policy_analyzer.py
-------------------
The orchestrator: ties together PDF processing, retrieval, and the LLM into
one pipeline, and adds a Python-based calculator for anything involving
arithmetic (attendance percentages, CGPA thresholds).

Why calculate in Python instead of asking the LLM to do the math?
LLMs predict text one token at a time -- they don't actually "compute". They
are often right on simple arithmetic but get shakier as numbers or steps get
more complex, and a wrong number in a policy answer ("you're allowed to miss
5 more classes") could genuinely mislead a student. So the rule in this
project is: the LLM's job is reading comprehension (find and explain the
rule), Python's job is arithmetic (apply the rule to the user's numbers).
"""

import re
import math

from src.pdf_processor import process_pdf
from src.retriever import VectorStore
from src.llm import answer_question, extract_numeric_rules


def build_policy_index(pdf_path: str) -> VectorStore:
    """Full ingestion pipeline: PDF -> chunks -> embeddings -> searchable index."""
    chunks = process_pdf(pdf_path)
    store = VectorStore()
    store.build(chunks)
    return store


# ---------------------------------------------------------------------------
# Python-side calculator
# ---------------------------------------------------------------------------
# These regexes pull numbers out of the *user's question*. We deliberately
# keep this narrow (attendance + CGPA are the two example scenarios in the
# spec) rather than trying to build a general-purpose math engine -- see the
# README's "Limitations" section for why this is a reasonable MVP boundary.

ATTENDANCE_PATTERN = re.compile(
    r"attend(?:ed|ing)?\s*(\d+)\s*(?:out of|/)\s*(\d+)", re.IGNORECASE
)
MISS_MORE_PATTERN = re.compile(r"miss(?:ed)?\s*(\d+)\s*more", re.IGNORECASE)
# Matches both orders: "CGPA is 8.2" / "CGPA of 8.2" and "8.2 CGPA"
CGPA_PATTERN = re.compile(
    r"cgpa\D{0,10}?(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*cgpa", re.IGNORECASE
)


def calculate_attendance(question: str, required_percent: float) -> dict | None:
    """
    Handles questions like:
    "I have attended 32 out of 45 classes. What happens if I miss 3 more?"

    Returns a dict of computed numbers, or None if the question doesn't
    contain an attendance figure to work with.
    """
    match = ATTENDANCE_PATTERN.search(question)
    if not match:
        return None

    attended, total_so_far = int(match.group(1)), int(match.group(2))
    current_percent = round((attended / total_so_far) * 100, 1)

    result = {
        "attended": attended,
        "total_so_far": total_so_far,
        "current_percent": current_percent,
        "required_percent": required_percent,
        "currently_meets_requirement": current_percent >= required_percent,
    }

    miss_match = MISS_MORE_PATTERN.search(question)
    if miss_match:
        miss_more = int(miss_match.group(1))
        new_total = total_so_far + miss_more
        new_percent = round((attended / new_total) * 100, 1)
        result["classes_missed_more"] = miss_more
        result["projected_total"] = new_total
        result["projected_percent"] = new_percent
        result["would_meet_requirement"] = new_percent >= required_percent

        # Bonus: how many more classes (from here) can they afford to miss
        # and still hit the requirement, assuming no more classes are added
        # beyond what's already scheduled? Solve for max misses `m` such
        # that attended / (total_so_far + m) >= required_percent / 100.
        if required_percent > 0:
            max_total_allowed = math.floor(attended / (required_percent / 100))
            max_more_misses = max(max_total_allowed - total_so_far, 0)
            result["max_additional_classes_can_miss"] = max_more_misses

    return result


def calculate_cgpa_eligibility(question: str, required_cgpa: float) -> dict | None:
    """
    Handles questions like:
    "Am I eligible for this scholarship if my CGPA is 8.2?"
    """
    match = CGPA_PATTERN.search(question)
    if not match:
        return None

    # Exactly one of the two capture groups will have matched, depending on
    # which order the number/word "CGPA" appeared in.
    user_cgpa = float(match.group(1) or match.group(2))
    return {
        "user_cgpa": user_cgpa,
        "required_cgpa": required_cgpa,
        "is_eligible": user_cgpa >= required_cgpa,
    }


# ---------------------------------------------------------------------------
# Main entry point used by the Streamlit app
# ---------------------------------------------------------------------------

def answer_policy_question(store: VectorStore, question: str, k: int = 4) -> dict:
    """
    Full pipeline for one question:
      1. Retrieve the most relevant chunks (retriever.py)
      2. Ask the LLM to answer, grounded in those chunks (llm.py)
      3. Ask the LLM to extract any numeric rules mentioned (llm.py)
      4. Run those numbers through the Python calculator, if applicable
      5. Package everything together, keeping "what the policy says" (LLM)
         clearly separate from "what we computed" (Python)

    Returns a dict with:
      answer          -- the LLM's grounded explanation (str)
      evidence        -- the retrieved chunks, for citation display (list[dict])
      calculation     -- Python-computed numbers, or None if not applicable (dict)
    """
    # Step 1: retrieval
    retrieved_chunks = store.search(question, k=k)

    if not retrieved_chunks:
        return {
            "answer": "No policy document has been indexed yet, or no relevant text was found.",
            "evidence": [],
            "calculation": None,
        }

    # Step 2: grounded generation
    answer_text = answer_question(question, retrieved_chunks)

    # Step 3: extract any explicit numeric thresholds from the same chunks
    rules = extract_numeric_rules(retrieved_chunks)

    # Step 4: run the appropriate calculator, if the question + policy give
    # us enough to work with
    calculation = None
    if rules.get("attendance_required_percent") is not None:
        calculation = calculate_attendance(question, rules["attendance_required_percent"])
        if calculation:
            calculation["type"] = "attendance"
    if calculation is None and rules.get("cgpa_required") is not None:
        calculation = calculate_cgpa_eligibility(question, rules["cgpa_required"])
        if calculation:
            calculation["type"] = "cgpa"

    return {
        "answer": answer_text,
        "evidence": retrieved_chunks,
        "calculation": calculation,
    }
