"""Hand-assembled minimal PDF bytes, shared by tests that need real PDF
bytes without depending on a PDF-writing library for the fixture itself -
kept independent of the pypdf version the adapter (and, through it, the
web UI) uses.
"""

from __future__ import annotations

import io

from pypdf import PdfReader, PdfWriter


def minimal_pdf(*page_texts: str) -> bytes:
    """Builds a minimal, valid PDF with one page per text given (or a
    single blank-content page if none is given), by writing raw PDF
    objects directly.

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


def encrypted_pdf(text: str) -> bytes:
    reader = PdfReader(io.BytesIO(minimal_pdf(text)))
    writer = PdfWriter()
    writer.append(reader)
    writer.encrypt(user_password="hunter2")
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()
