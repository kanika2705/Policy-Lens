"""
pdf_processor.py
-----------------
Turns a PDF policy document into a list of small, searchable "chunks" of text.

Why do we chunk at all?
LLMs (and vector search) work best on small, focused pieces of text rather than
one giant document. A 20-page policy PDF is too big to hand to an LLM at once
(expensive, slow, and the model tends to "lose" details in the middle of long
inputs). So we break the PDF into ~800-character chunks, each tagged with the
page number it came from, so we can later show the user exactly where an
answer's evidence lives.
"""

import re
import pymupdf as fitz  # PyMuPDF (imported under its old alias `fitz` by convention)


# A chunk is just a plain dict. Using a dict (instead of a custom class) keeps
# this beginner-friendly -- no need to explain classes/inheritance to explain
# the pipeline.
#
# Chunk fields:
#   chunk_id : unique integer id
#   text     : the actual chunk text
#   page     : the PDF page number this chunk starts on (1-indexed)
#   section  : best-guess section/clause heading the chunk falls under


def extract_pages(pdf_path: str) -> list[str]:
    """
    Open a PDF and return a list of strings, one per page.
    Using PyMuPDF (imported as `fitz`) because it's fast and keeps page
    boundaries clean, which we need for page-number citations later.
    """
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return pages


# Very light heuristic for spotting section/clause headings, e.g.
# "3.2 Attendance Requirements" or "Section 4: Eligibility".
# This is intentionally simple (a regex, not an ML model) -- good enough to
# give the user a helpful label, not meant to be perfect.
SECTION_HEADING_PATTERN = re.compile(
    r"^\s*(?:(\d+(?:\.\d+)*\.?)\s+)?([A-Z][A-Za-z0-9 ,\-/]{3,60})\s*$"
)


def _looks_like_heading(line: str) -> bool:
    """Heuristic: short line, no ending punctuation, mostly capitalized words."""
    line = line.strip()
    if not (3 < len(line) < 70):
        return False
    if line.endswith((".", ",", ";")):
        return False
    return bool(SECTION_HEADING_PATTERN.match(line))


def chunk_text(
    pages: list[str],
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[dict]:
    """
    Walk through the extracted pages and build overlapping text chunks.

    - chunk_size: target number of characters per chunk
    - overlap: how many characters of the previous chunk to repeat at the
      start of the next one, so we don't accidentally cut a sentence (or a
      key number/clause) exactly at a chunk boundary.

    Returns a list of chunk dicts (see module docstring for fields).
    """
    chunks = []
    chunk_id = 0
    current_section = "General"
    buffer = ""
    buffer_start_page = 1

    for page_num, page_text in enumerate(pages, start=1):
        for raw_line in page_text.split("\n"):
            line = raw_line.strip()

            # Update our "current section" guess whenever we spot a heading.
            # This gets attached to every chunk that follows, so evidence
            # shown to the user can say "Section: 3.2 Attendance Requirements".
            if _looks_like_heading(line):
                current_section = line

            if not buffer:
                buffer_start_page = page_num

            buffer += raw_line + "\n"

            # Once the buffer is big enough, cut a chunk.
            if len(buffer) >= chunk_size:
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": buffer.strip(),
                        "page": buffer_start_page,
                        "section": current_section,
                    }
                )
                chunk_id += 1
                # Keep the last `overlap` characters so context carries over.
                buffer = buffer[-overlap:]
                buffer_start_page = page_num

    # Don't lose the final partial chunk.
    if buffer.strip():
        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": buffer.strip(),
                "page": buffer_start_page,
                "section": current_section,
            }
        )

    return chunks


def process_pdf(pdf_path: str) -> list[dict]:
    """Convenience wrapper: PDF path -> list of chunk dicts, ready to embed."""
    pages = extract_pages(pdf_path)
    return chunk_text(pages)
