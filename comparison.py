"""
comparison.py
--------------
The second feature: compare an OLD and NEW version of the same policy and
report what changed.

Approach:
1. Extract plain text from both PDFs and split each into paragraphs (a
   coarser split than the RAG chunking in pdf_processor.py -- for comparison
   we want whole clauses, not overlapping fragments).
2. Use Python's built-in `difflib` to align the two paragraph lists. This is
   the same general technique behind tools like `git diff`: it finds which
   paragraphs are unchanged, which are new, which were deleted, and which
   were replaced -- using pure text similarity, no AI involved yet.
3. For every paragraph that actually changed (skip exact matches), ask the
   LLM to (a) label it Added/Removed/Modified, (b) summarize the change in
   plain English, and (c) note who's likely affected.

Splitting the work this way keeps the LLM calls small and focused (one
paragraph pair at a time) instead of asking it to "find all the differences"
in two huge documents, which tends to produce vague or incomplete answers.
"""

import difflib

from src.pdf_processor import extract_pages
from src.llm import classify_change


def _extract_paragraphs(pdf_path: str) -> list[str]:
    """PDF -> list of non-empty paragraphs (split on blank lines)."""
    pages = extract_pages(pdf_path)
    full_text = "\n".join(pages)
    raw_paragraphs = full_text.split("\n\n")
    return [p.strip() for p in raw_paragraphs if len(p.strip()) > 20]


def compare_policies(old_pdf_path: str, new_pdf_path: str) -> list[dict]:
    """
    Compare two policy PDFs and return a list of meaningful changes.

    Each item in the returned list looks like:
      {
        "change_type": "Added" | "Removed" | "Modified",
        "old_text": str,
        "new_text": str,
        "summary": str,
        "who_is_affected": str,
      }
    """
    old_paragraphs = _extract_paragraphs(old_pdf_path)
    new_paragraphs = _extract_paragraphs(new_pdf_path)

    # SequenceMatcher finds the best alignment between the two paragraph
    # lists and gives us "opcodes" describing each aligned block:
    #   'equal'   -> unchanged, we skip these
    #   'replace' -> paragraph(s) modified
    #   'delete'  -> paragraph(s) only in the old version (removed)
    #   'insert'  -> paragraph(s) only in the new version (added)
    matcher = difflib.SequenceMatcher(None, old_paragraphs, new_paragraphs)

    changes = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        old_block = "\n".join(old_paragraphs[i1:i2])
        new_block = "\n".join(new_paragraphs[j1:j2])

        # Skip trivial/whitespace-only differences (e.g. page-break artifacts).
        if not old_block and not new_block:
            continue

        classification = classify_change(old_block, new_block)
        changes.append(
            {
                "change_type": classification.get("change_type", tag.capitalize()),
                "old_text": old_block,
                "new_text": new_block,
                "summary": classification.get("summary", ""),
                "who_is_affected": classification.get("who_is_affected", ""),
            }
        )

    return changes
