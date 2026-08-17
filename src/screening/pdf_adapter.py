"""The PDF adapter: a PDF goes in, Resume text comes out, in a form the core
consumes unchanged (ticket 10).

Lives outside the core and outside every evaluated path, per the spec's
Ingestion decision - the evaluated path (screening.core, the evaluation
harness, every sweep script) consumes plain Resume text directly and never
imports this module. Only the demo UI (screening.web, ticket 11) calls it,
so a parsing bug here can never be mistaken for a ranking bug in the metrics.

Tested entirely on its own, against real PDF bytes, independently of the
core (tests/test_pdf_adapter.py) - no fake or double stands in for pypdf.
"""

from __future__ import annotations

import io

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from screening.domain import Resume


class PdfExtractionError(Exception):
    """Raised when a PDF cannot be converted into Resume text - an
    unreadable file, an encrypted one, or one with no extractable text
    layer (e.g. a scanned image). Raised rather than returning an empty or
    partial Resume, so an extraction failure is surfaced clearly instead of
    silently becoming a Candidate with nothing for Screening to read.
    """


def extract_resume(pdf_bytes: bytes) -> Resume:
    """Converts the bytes of a single PDF Resume into a Resume the core
    consumes exactly as it consumes corpus text.
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except (PdfReadError, ValueError) as exc:
        raise PdfExtractionError(f"could not read PDF: {exc}") from exc

    if reader.is_encrypted:
        raise PdfExtractionError("PDF is encrypted and cannot be read without a password")

    try:
        pages_text = [page.extract_text() or "" for page in reader.pages]
    except (PdfReadError, ValueError) as exc:
        raise PdfExtractionError(f"could not extract text from PDF: {exc}") from exc

    text = "\n".join(pages_text).strip()
    if not text:
        raise PdfExtractionError(
            "PDF produced no extractable text - it may be a scanned image with no text layer"
        )
    return Resume(text=text)
