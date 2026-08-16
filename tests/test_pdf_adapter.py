"""The PDF adapter is tested entirely on its own, independently of the core
(ticket 10) - these tests only import screening.pdf_adapter and
screening.domain, run against real PDF bytes built by hand below, and never
touch screening.core or the model client seam.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter

from screening.pdf_adapter import PdfExtractionError, extract_resume, extract_resume_from_path

SRC_ROOT = Path(__file__).parent.parent / "src" / "screening"
SCRIPTS_ROOT = Path(__file__).parent.parent / "scripts"


def _minimal_pdf(*page_texts: str) -> bytes:
    """Builds a minimal, valid, hand-assembled PDF with one page per text
    given (or a single blank-content page if none is given), by writing raw
    PDF objects directly - no PDF-writing library is used here, so this
    fixture stays independent of the pypdf version the adapter itself uses.

    Object numbering: 1 = Catalog, 2 = Pages, then per page i (0-indexed)
    starting at object 3: the Page dict, its Contents stream, and its Font
    dict, three objects apart.
    """
    page_texts = page_texts or ("",)
    num_pages = len(page_texts)

    def page_obj(i: int) -> int:
        return 3 + 3 * i

    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: (
            f"<< /Type /Pages /Kids [{' '.join(f'{page_obj(i)} 0 R' for i in range(num_pages))}] "
            f"/Count {num_pages} >>"
        ).encode("latin-1"),
    }
    for i, text in enumerate(page_texts):
        p, content_obj, font_obj = page_obj(i), page_obj(i) + 1, page_obj(i) + 2
        content = f"BT /F1 18 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
        objects[p] = (
            f"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 {font_obj} 0 R >> >> "
            f"/MediaBox [0 0 612 792] /Contents {content_obj} 0 R >>"
        ).encode("latin-1")
        objects[content_obj] = (
            f"<< /Length {len(content)} >>\nstream\n".encode("latin-1") + content + b"\nendstream"
        )
        objects[font_obj] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    ordered_objects = [objects[n] for n in sorted(objects)]

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(ordered_objects, start=1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode("latin-1"))
        out.write(obj)
        out.write(b"\nendobj\n")
    xref_offset = out.tell()
    n = len(ordered_objects) + 1
    out.write(f"xref\n0 {n}\n".encode("latin-1"))
    out.write(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.write(f"{off:010d} 00000 n \n".encode("latin-1"))
    out.write(b"trailer\n")
    out.write(f"<< /Size {n} /Root 1 0 R >>\n".encode("latin-1"))
    out.write(b"startxref\n")
    out.write(f"{xref_offset}\n".encode("latin-1"))
    out.write(b"%%EOF")
    return out.getvalue()


def _encrypted_pdf(text: str) -> bytes:
    reader = PdfReader(io.BytesIO(_minimal_pdf(text)))
    writer = PdfWriter()
    writer.append(reader)
    writer.encrypt(user_password="hunter2")
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


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


def test_extract_resume_from_path_reads_the_file_and_extracts_the_same_text(tmp_path: Path):
    pdf_path = tmp_path / "resume.pdf"
    pdf_path.write_bytes(_minimal_pdf("Priya Natarajan, Data Engineer"))

    resume = extract_resume_from_path(pdf_path)

    assert resume.text.strip() == "Priya Natarajan, Data Engineer"


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
    directly, so none of them should ever reference this module.
    """
    evaluated_sources = [
        path
        for path in list(SRC_ROOT.rglob("*.py")) + list(SCRIPTS_ROOT.rglob("*.py"))
        if path.name != "pdf_adapter.py"
    ]

    offenders = [
        path for path in evaluated_sources if "pdf_adapter" in path.read_text()
    ]

    assert offenders == [], f"evaluated-path modules must not import pdf_adapter: {offenders}"
