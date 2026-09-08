"""
llm.py
------
All calls to the LLM live here, in one place, so it's easy to see exactly
what prompts we send and why.

This project uses a LOCAL model through Ollama instead of a paid API.
Ollama runs a model (e.g. llama3.2:3b) directly on your machine and exposes
it on http://localhost:11434. The `ollama` Python package just talks to that
local server -- no API key, no internet required after the model is
downloaded, no billing.

Two jobs for the LLM in this project:
1. answer_question()   -> answer the user's question using ONLY the
                           retrieved policy chunks (this is the "G" -
                           Generation - step of RAG).
2. classify_change()    -> used by the Policy Comparison feature, to
                           describe how one version of a clause differs
                           from another.

Grounding / anti-hallucination strategy:
We do not ask the model "what do you know about this policy". We ask it to
answer strictly from the text we paste into the prompt, and we explicitly
tell it to say "the policy does not specify this" when the answer isn't in
that text. This doesn't make hallucination impossible, but it removes the
main cause of it (the model falling back on general knowledge or guessing).
"""

import os
import json
import ollama

# Which local model to use. llama3.2:3b is a good balance of quality and
# speed on a normal laptop with no GPU. Override with the OLLAMA_MODEL
# environment variable if you've pulled a different model (e.g. "phi3",
# "mistral", "llama3.2:1b" for an even lighter/faster option).
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")


ANSWER_SYSTEM_PROMPT = """You are PolicyLens, an assistant that explains policy documents.

Rules you MUST follow:
1. Answer ONLY using the "POLICY EXCERPTS" provided below. Do not use outside knowledge.
2. If the excerpts do not contain enough information to answer, say so explicitly
   (e.g. "The provided policy text does not specify this."). Do not guess or make up rules.
3. When you reference a rule or number from the policy, mention which excerpt it came from
   (e.g. "Excerpt 2").
4. If the question requires arithmetic (percentages, counts, deadlines), do NOT do the
   math yourself. Instead, clearly state the raw numbers/thresholds found in the policy
   text (e.g. "minimum attendance required: 75%") in a line starting with "RULE:" so a
   separate calculator can use them. Keep your explanation in plain, simple language.
5. Be concise and clear. This is being read by a student, not a lawyer.
"""


def _format_context(chunks: list[dict]) -> str:
    """Turn retrieved chunks into a numbered block of text for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Excerpt {i} | Page {chunk.get('page', '?')} | "
            f"Section: {chunk.get('section', 'General')}]\n{chunk['text']}"
        )
    return "\n\n".join(parts)


def answer_question(question: str, chunks: list[dict]) -> str:
    """
    Core RAG "generation" step: ask the local LLM to answer `question`,
    grounded strictly in the given `chunks` (already retrieved by
    retriever.py). Returns the raw answer text from the model.
    """
    context = _format_context(chunks)

    user_prompt = f"""POLICY EXCERPTS:
{context}

QUESTION:
{question}

Answer the question using only the excerpts above."""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.1},  # low temperature: faithful, consistent answers
    )
    return response["message"]["content"]


EXTRACTION_SYSTEM_PROMPT = """You extract numeric rules from a short piece of policy text.
Return STRICT JSON only, no other text, matching this shape:
{
  "attendance_required_percent": <number or null>,
  "cgpa_required": <number or null>,
  "deadline_mentioned": <string or null>,
  "notes": <short string or null>
}
If a field isn't mentioned in the text, use null. Do not guess numbers that aren't present."""


def extract_numeric_rules(chunks: list[dict]) -> dict:
    """
    Ask the LLM to pull out any explicit numeric thresholds (attendance %,
    CGPA, deadlines) from the retrieved chunks, as structured JSON.

    Why: we don't trust the LLM to do the arithmetic on top of these numbers
    (see policy_analyzer.py), but it's good at reading messy natural-language
    text and pulling out "the required percentage is 75%". Extraction is a
    much easier, more reliable task for an LLM than multi-step arithmetic.

    We use Ollama's `format="json"` option, which constrains the model's
    output to valid JSON -- this is what makes it safe to json.loads() below.
    """
    context = _format_context(chunks)
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
        format="json",
        options={"temperature": 0},
    )
    default = {
        "attendance_required_percent": None,
        "cgpa_required": None,
        "deadline_mentioned": None,
        "notes": None,
    }
    try:
        parsed = json.loads(response["message"]["content"])
        # Merge onto the default so a partially-formed response (missing a
        # key) doesn't break the rest of the pipeline.
        default.update(parsed)
        return default
    except (json.JSONDecodeError, TypeError, KeyError):
        return default


CHANGE_SYSTEM_PROMPT = """You compare an OLD and NEW version of a policy clause.
Respond with STRICT JSON only, matching this shape:
{
  "change_type": "Added" | "Removed" | "Modified",
  "summary": "<one sentence describing what changed>",
  "who_is_affected": "<one sentence on who this impacts and how>"
}
Be factual and concise. Base your answer only on the two text versions given."""


def classify_change(old_text: str, new_text: str) -> dict:
    """
    Given an old and new version of roughly the same clause, ask the LLM to
    (a) label the change type, (b) summarize it, (c) say who's affected.
    Used by the Policy Comparison feature.
    """
    user_prompt = f"""OLD VERSION:
{old_text or "(this clause did not exist in the old version)"}

NEW VERSION:
{new_text or "(this clause was removed in the new version)"}"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": CHANGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        format="json",
        options={"temperature": 0},
    )
    try:
        return json.loads(response["message"]["content"])
    except (json.JSONDecodeError, TypeError, KeyError):
        return {
            "change_type": "Modified",
            "summary": "Could not automatically summarize this change.",
            "who_is_affected": "Unknown",
        }


def check_ollama_ready() -> tuple[bool, str]:
    """
    Health check used by app.py at startup: is the Ollama server running,
    and is the model we need already pulled? Returns (is_ready, message) so
    the UI can show a clear, actionable error instead of a raw traceback.
    """
    try:
        models_response = ollama.list()
    except Exception:
        return False, (
            "Could not connect to Ollama. Make sure the Ollama app/service is "
            "installed and running, then reload this page."
        )

    # Different versions of the ollama package structure this response
    # slightly differently, so we handle it defensively rather than assuming
    # one exact shape.
    raw_models = models_response.get("models", []) if isinstance(models_response, dict) \
        else getattr(models_response, "models", [])

    available_names = []
    for m in raw_models:
        if isinstance(m, dict):
            name = m.get("model") or m.get("name")
        else:
            name = getattr(m, "model", None) or getattr(m, "name", None)
        if name:
            available_names.append(name)

    # Match loosely on the base model name (before the ":tag") since Ollama
    # may list "llama3.2:3b" while we only care that some llama3.2 is present.
    base_name = OLLAMA_MODEL.split(":")[0]
    if not any(base_name in name for name in available_names):
        return False, (
            f"Ollama is running, but the model '{OLLAMA_MODEL}' isn't downloaded yet. "
            f"Run this in a terminal: ollama pull {OLLAMA_MODEL}"
        )

    return True, "Ollama is ready."
