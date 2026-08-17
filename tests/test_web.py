"""The web UI is the third consumer of the core's public entry points,
alongside the CLI and the evaluation harness (ticket 11). These tests drive
it end to end through Flask's test client - no running server, no network -
against a scripted model client, exactly mirroring how tests/test_cli.py
drives the CLI.

The wizard carries state between its four pages in hidden form fields
rather than server-side sessions, so a test walks that same path a
Recruiter's browser would: extract, then edit/approve, then upload, then
read the Shortlist.
"""

from __future__ import annotations

import html
import io
import json
import re

import pytest

from screening.scripted_client import ScriptedModelClient
from screening.web import create_app
from tests.pdf_fixtures import minimal_pdf

_JOB_DESCRIPTION = (
    "Backend Engineer\n"
    "- Python programming experience\n"
    "- Bachelor's degree in Computer Science\n"
)


@pytest.fixture
def client(tmp_path):
    app = create_app(ScriptedModelClient(), runs_dir=tmp_path / "runs")
    app.testing = True
    return app.test_client()


def _propose(client, *, title: str = "Backend Engineer", job_description: str = _JOB_DESCRIPTION):
    return client.post(
        "/requirements", data={"title": title, "job_description": job_description}
    )


def _proposed_ids(html_text: str) -> list[str]:
    return re.findall(r'name="proposed_ids" value="([^"]+)"', html_text)


def _proposed_texts(html_text: str) -> dict[str, str]:
    """The pre-filled value of each proposed Requirement's editable text
    field, as the review page renders it - the same text a Recruiter would
    see before choosing to edit, keep, or drop it.
    """
    return {
        req_id: html.unescape(value)
        for req_id, value in re.findall(r'name="text-([^"]+)" value="([^"]*)"', html_text)
    }


def _approve(
    client,
    *,
    title: str = "Backend Engineer",
    keep_ids: list[str],
    proposed_ids: list[str],
    proposed_texts: dict[str, str],
    texts: dict[str, str] | None = None,
    additional_requirements: str = "",
):
    texts = texts or {}
    form = {"title": title, "proposed_ids": proposed_ids, "additional_requirements": additional_requirements}
    for req_id in proposed_ids:
        form[f"text-{req_id}"] = texts.get(req_id, proposed_texts[req_id])
        if req_id in keep_ids:
            form[f"keep-{req_id}"] = "on"
    return client.post("/requirements/approve", data=form)


def _requirements_json_from(html_text: str) -> list[dict]:
    match = re.search(r'name="requirements_json" value="([^"]*)"', html_text)
    assert match, "requirements_json hidden field not found in upload page"
    return json.loads(html.unescape(match.group(1)))


def _upload_pdfs(client, *, title: str, requirements_json: list[dict], files: list[tuple[bytes, str]]):
    return client.post(
        "/screening",
        data={
            "title": title,
            "requirements_json": json.dumps(requirements_json),
            "resumes": [(io.BytesIO(content), name) for content, name in files],
        },
        content_type="multipart/form-data",
    )


def test_role_form_renders():
    app = create_app(ScriptedModelClient())
    response = app.test_client().get("/")
    assert response.status_code == 200
    assert b"Job Description" in response.data


def test_extraction_proposes_discrete_requirements_for_review(client):
    response = _propose(client)

    assert response.status_code == 200
    text = response.get_data(as_text=True)
    assert "Python programming experience" in text
    assert _proposed_ids(text) == ["req-1", "req-2"]
    assert _proposed_texts(text) == {
        "req-1": "Python programming experience",
        "req-2": "Bachelor's degree in Computer Science",
    }


def test_role_form_requires_a_title_and_a_job_description(client):
    response = client.post("/requirements", data={"title": "", "job_description": ""})

    assert response.status_code == 400
    assert "required" in response.get_data(as_text=True).lower()


def test_recruiter_can_edit_delete_and_add_requirements_before_approval(client):
    """Covers ticket 11's first acceptance criterion end to end: a proposed
    Requirement can be edited, one can be dropped, and one the extraction
    missed can be added - none of that is reflected until the approve step,
    and what comes out the other side is what actually gates Screening.
    """
    proposed_html = _propose(client).get_data(as_text=True)
    proposed_ids = _proposed_ids(proposed_html)
    assert len(proposed_ids) == 2
    degree_id = proposed_ids[1]

    approved = _approve(
        client,
        proposed_ids=proposed_ids,
        proposed_texts=_proposed_texts(proposed_html),
        keep_ids=[proposed_ids[0]],  # drop the degree requirement
        texts={proposed_ids[0]: "Strong Python programming experience"},
        additional_requirements="Willing to work US hours\n\n",
    )

    assert approved.status_code == 200
    requirements = _requirements_json_from(approved.get_data(as_text=True))
    texts = [r["text"] for r in requirements]
    assert texts == ["Strong Python programming experience", "Willing to work US hours"]
    assert not any(r["id"] == degree_id for r in requirements)


def test_approval_is_rejected_when_every_requirement_is_dropped(client):
    proposed_html = _propose(client).get_data(as_text=True)
    proposed_ids = _proposed_ids(proposed_html)

    response = _approve(
        client, proposed_ids=proposed_ids, proposed_texts=_proposed_texts(proposed_html), keep_ids=[]
    )

    assert response.status_code == 400
    assert "at least one" in response.get_data(as_text=True).lower()


def test_shortlist_includes_every_submitted_candidate_qualified_and_disqualified(client):
    proposed_html = _propose(client).get_data(as_text=True)
    proposed_ids = _proposed_ids(proposed_html)
    approved = _approve(
        client,
        proposed_ids=proposed_ids,
        proposed_texts=_proposed_texts(proposed_html),
        keep_ids=[proposed_ids[0]],
    )  # only "Python programming experience" survives
    requirements_json = _requirements_json_from(approved.get_data(as_text=True))

    alice_pdf = minimal_pdf("Built backend services in Python for six years.")
    bob_pdf = minimal_pdf("Led a design team with a marketing background.")
    response = _upload_pdfs(
        client,
        title="Backend Engineer",
        requirements_json=requirements_json,
        files=[(alice_pdf, "alice.pdf"), (bob_pdf, "bob.pdf")],
    )

    assert response.status_code == 200
    out = response.get_data(as_text=True)
    assert "alice" in out and "Qualified" in out
    assert "bob" in out and "Disqualified" in out
    # Every outcome carries a readable Justification, not just a verdict.
    assert "Resume cites" in out or "does not evidence" in out


def test_shortlist_page_explains_the_comparative_passs_order(client):
    """When two or more Candidates qualify, the comparative pass reorders
    the top band, so the Shortlist page must show why - a citable
    justification for each comparison, not just the rubric Fit dimensions
    underneath it (ADR-0001).
    """
    proposed_html = _propose(client).get_data(as_text=True)
    proposed_ids = _proposed_ids(proposed_html)
    approved = _approve(
        client,
        proposed_ids=proposed_ids,
        proposed_texts=_proposed_texts(proposed_html),
        keep_ids=[proposed_ids[0]],
    )
    requirements_json = _requirements_json_from(approved.get_data(as_text=True))

    alice_pdf = minimal_pdf("Built backend services in Python for ten years.")
    bob_pdf = minimal_pdf("Built backend services in Python for eight years.")
    response = _upload_pdfs(
        client,
        title="Backend Engineer",
        requirements_json=requirements_json,
        files=[(alice_pdf, "alice.pdf"), (bob_pdf, "bob.pdf")],
    )

    assert response.status_code == 200
    out = response.get_data(as_text=True)
    assert "How the top of the Shortlist was ordered" in out
    assert " over " in out


def test_a_completed_screening_run_is_recorded_as_a_durable_run_record(client, tmp_path):
    """spec:56/150: each Screening Run is kept as a durable, append-only
    record so a Recruiter can revisit how a decision was reached weeks
    later - the same record the CLI writes via FileRunStore, not something
    only the CLI path produces.
    """
    proposed_html = _propose(client).get_data(as_text=True)
    proposed_ids = _proposed_ids(proposed_html)
    approved = _approve(
        client,
        proposed_ids=proposed_ids,
        proposed_texts=_proposed_texts(proposed_html),
        keep_ids=[proposed_ids[0]],
    )
    requirements_json = _requirements_json_from(approved.get_data(as_text=True))

    alice_pdf = minimal_pdf("Built backend services in Python for six years.")
    _upload_pdfs(
        client,
        title="Backend Engineer",
        requirements_json=requirements_json,
        files=[(alice_pdf, "alice.pdf")],
    )

    run_files = list((tmp_path / "runs").glob("*.jsonl"))
    assert len(run_files) == 1
    lines = run_files[0].read_text().splitlines()
    header = json.loads(lines[0])
    assert header["role"]["title"] == "Backend Engineer"
    entry = json.loads(lines[1])
    assert entry["candidate_id"] == "alice"


def test_two_resumes_with_the_same_filename_get_distinct_candidate_ids(client):
    """A regression test: two Candidates uploaded under the same filename
    (e.g. both literally named resume.pdf) must not collide in
    ranking.py's resume-by-id lookup and end up judged against each
    other's Resume text - each needs an id of its own."""
    proposed_html = _propose(client).get_data(as_text=True)
    proposed_ids = _proposed_ids(proposed_html)
    approved = _approve(
        client,
        proposed_ids=proposed_ids,
        proposed_texts=_proposed_texts(proposed_html),
        keep_ids=[proposed_ids[0]],
    )
    requirements_json = _requirements_json_from(approved.get_data(as_text=True))

    alice_pdf = minimal_pdf("Built backend services in Python for six years.")
    bob_pdf = minimal_pdf("Led a design team with a marketing background.")
    response = _upload_pdfs(
        client,
        title="Backend Engineer",
        requirements_json=requirements_json,
        files=[(alice_pdf, "resume.pdf"), (bob_pdf, "resume.pdf")],
    )

    assert response.status_code == 200
    out = response.get_data(as_text=True)
    assert ">resume<" in out
    assert ">resume-2<" in out
    assert "Qualified" in out
    assert "Disqualified" in out


def test_a_kept_requirement_cannot_be_approved_with_empty_text(client):
    """A Requirement whose checkbox is still checked must not be silently
    dropped for having its text cleared - the Recruiter is told instead,
    since a Requirement vanishing unannounced is exactly the kind of
    silent narrowing ADR-0001 rules out."""
    proposed_html = _propose(client).get_data(as_text=True)
    proposed_ids = _proposed_ids(proposed_html)
    proposed_texts = _proposed_texts(proposed_html)

    response = _approve(
        client,
        proposed_ids=proposed_ids,
        proposed_texts=proposed_texts,
        keep_ids=proposed_ids,
        texts={proposed_ids[0]: ""},
    )

    assert response.status_code == 400
    out = response.get_data(as_text=True)
    assert "cannot have empty text" in out
    # The other, still-valid Requirement is preserved for re-editing rather
    # than lost along with the invalid one.
    assert proposed_ids[1] in _proposed_texts(out)
    assert _proposed_texts(out)[proposed_ids[1]] == proposed_texts[proposed_ids[1]]


def test_screening_with_no_uploaded_files_reports_a_clear_error(client):
    proposed_html = _propose(client).get_data(as_text=True)
    proposed_ids = _proposed_ids(proposed_html)
    approved = _approve(
        client,
        proposed_ids=proposed_ids,
        proposed_texts=_proposed_texts(proposed_html),
        keep_ids=proposed_ids,
    )
    requirements_json = _requirements_json_from(approved.get_data(as_text=True))

    response = client.post(
        "/screening",
        data={"title": "Backend Engineer", "requirements_json": json.dumps(requirements_json)},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "No Resume files were uploaded." in response.get_data(as_text=True)


def test_shortlist_has_no_reject_threshold_or_cutoff_control():
    """ADR-0001: no reject verb, no threshold, no way to hide a Candidate.
    Checked against the page's interactive controls (buttons and named
    form fields) rather than the whole page, since the page's own
    disclaimer text legitimately says the word "rejected" while explaining
    that nothing here does that.
    """
    app = create_app(ScriptedModelClient())
    client = app.test_client()
    proposed_html = _propose(client).get_data(as_text=True)
    proposed_ids = _proposed_ids(proposed_html)
    approved = _approve(
        client,
        proposed_ids=proposed_ids,
        proposed_texts=_proposed_texts(proposed_html),
        keep_ids=proposed_ids,
    )
    requirements_json = _requirements_json_from(approved.get_data(as_text=True))

    response = _upload_pdfs(
        client,
        title="Backend Engineer",
        requirements_json=requirements_json,
        files=[(minimal_pdf("Built backend services in Python for six years."), "alice.pdf")],
    )
    out = response.get_data(as_text=True)

    button_labels = re.findall(r"<button[^>]*>(.*?)</button>", out, re.S)
    assert not any(
        re.search(r"reject|threshold|cutoff", label, re.I) for label in button_labels
    )
    field_names = re.findall(r'name="([^"]+)"', out)
    assert not any(re.search(r"reject|threshold|cutoff", name, re.I) for name in field_names)


def test_an_unreadable_pdf_is_reported_rather_than_silently_dropped(client):
    proposed_html = _propose(client).get_data(as_text=True)
    proposed_ids = _proposed_ids(proposed_html)
    approved = _approve(
        client,
        proposed_ids=proposed_ids,
        proposed_texts=_proposed_texts(proposed_html),
        keep_ids=proposed_ids,
    )
    requirements_json = _requirements_json_from(approved.get_data(as_text=True))

    response = _upload_pdfs(
        client,
        title="Backend Engineer",
        requirements_json=requirements_json,
        files=[
            (minimal_pdf("Built backend services in Python for six years."), "alice.pdf"),
            (b"this is not a PDF at all", "broken.pdf"),
        ],
    )

    assert response.status_code == 200
    out = response.get_data(as_text=True)
    assert "alice" in out
    assert "broken.pdf" in out
    assert "Could not be read" in out


def test_screening_fails_clearly_when_no_resume_could_be_read(client):
    proposed_html = _propose(client).get_data(as_text=True)
    proposed_ids = _proposed_ids(proposed_html)
    approved = _approve(
        client,
        proposed_ids=proposed_ids,
        proposed_texts=_proposed_texts(proposed_html),
        keep_ids=proposed_ids,
    )
    requirements_json = _requirements_json_from(approved.get_data(as_text=True))

    response = _upload_pdfs(
        client,
        title="Backend Engineer",
        requirements_json=requirements_json,
        files=[(b"this is not a PDF at all", "broken.pdf")],
    )

    assert response.status_code == 400
    assert "No Resume could be read" in response.get_data(as_text=True)


def test_web_ui_drives_the_core_through_run_screening_and_extract_requirements():
    """Ticket 11's fifth acceptance criterion: the web UI is a consumer of
    the same entry points as the CLI and the evaluation harness, not a
    parallel reimplementation of Screening or Ranking.
    """
    import screening.web as web

    assert web.run_screening.__module__ == "screening.core"
    assert web.extract_requirements.__module__ == "screening.extraction"
