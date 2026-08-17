"""The PDF adapter is tested entirely on its own, independently of the core
(ticket 10) - these tests only import screening.pdf_adapter and
screening.domain, run against real PDF bytes built by hand below, and never
touch screening.core or the model client seam.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from screening.pdf_adapter import PdfExtractionError, extract_resume
from tests.pdf_fixtures import encrypted_pdf as _encrypted_pdf
from tests.pdf_fixtures import minimal_pdf as _minimal_pdf

SRC_ROOT = Path(__file__).parent.parent / "src" / "screening"
SCRIPTS_ROOT = Path(__file__).parent.parent / "scripts"


def test_a_pdf_is_converted_to_resume_text_the_core_can_consume_unchanged():
    pdf_bytes = _minimal_pdf("Jordan Alvarez, Backend Engineer")

    resume = extract_resume(pdf_bytes)

    assert resume.text.strip() == "Jordan Alvarez, Backend Engineer"


def test_a_multi_page_pdf_concatenates_its_pages_in_order():
    pdf_bytes = _minimal_pdf("Experience at Acme Corp", "Education at State University")

    resume = extract_resume(pdf_bytes)

    assert "Experience at Acme Corp" in resume.text
    assert "Education at State University" in resume.text
    assert resume.text.index("Experience at Acme Corp") < resume.text.index(
        "Education at State University"
    )


def test_unreadable_bytes_raise_clearly_instead_of_yielding_empty_text():
    with pytest.raises(PdfExtractionError):
        extract_resume(b"this is not a PDF at all")


def test_empty_bytes_raise_clearly_instead_of_yielding_empty_text():
    with pytest.raises(PdfExtractionError):
        extract_resume(b"")


def test_an_encrypted_pdf_raises_clearly_instead_of_yielding_empty_text():
    pdf_bytes = _encrypted_pdf("secret resume text")

    with pytest.raises(PdfExtractionError, match="encrypted"):
        extract_resume(pdf_bytes)


def test_a_pdf_with_no_text_layer_raises_clearly_instead_of_yielding_empty_text():
    pdf_bytes = _minimal_pdf("")

    with pytest.raises(PdfExtractionError, match="no extractable text"):
        extract_resume(pdf_bytes)


def test_a_pdf_of_only_whitespace_raises_clearly_instead_of_yielding_empty_text():
    pdf_bytes = _minimal_pdf("   ")

    with pytest.raises(PdfExtractionError, match="no extractable text"):
        extract_resume(pdf_bytes)


def test_no_evaluated_module_imports_the_pdf_adapter():
    """No run that produces a reported metric passes through the adapter
    (ticket 10's third acceptance criterion): the core, the evaluation
    harness, and every sweep/eval script consume plain Resume text
    directly, so none of them should ever reference this module. The one
    exception is screening.web (ticket 11): the demo UI is the adapter's
    intended, sole caller.
    """
    evaluated_sources = [
        path
        for path in list(SRC_ROOT.rglob("*.py")) + list(SCRIPTS_ROOT.rglob("*.py"))
        if path.name not in ("pdf_adapter.py", "web.py")
    ]

    offenders = [
        path for path in evaluated_sources if "pdf_adapter" in path.read_text()
    ]

    assert offenders == [], f"evaluated-path modules must not import pdf_adapter: {offenders}"
