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
from dataclasses import replace

import pytest

from markupsafe import escape

from screening.demo_presets import DemoPreset, load_demo_presets
from screening.domain import Candidate, Resume
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
    """No Demo Presets: these tests walk the flow a Recruiter supplying
    their own Job Description and files walks (ticket 11). The Preset flow
    (ticket 12) has its own fixture below."""
    app = create_app(ScriptedModelClient(), runs_dir=tmp_path / "runs", demo_presets=[])
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
    preset_id: str | None = None,
):
    texts = texts or {}
    form = {"title": title, "proposed_ids": proposed_ids, "additional_requirements": additional_requirements}
    if preset_id is not None:
        form["preset_id"] = preset_id
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
    app = create_app(ScriptedModelClient(), demo_presets=[])
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
    app = create_app(ScriptedModelClient(), demo_presets=[])
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


# --- Demo Presets (ticket 12) -------------------------------------------------

_PRESET = DemoPreset(
    preset_id="backend-engineer",
    label="Backend Engineer",
    family="Software and Data",
    summary="Two Python Resumes against a designer.",
    role_title="Backend Engineer",
    job_description=_JOB_DESCRIPTION,
    source_evaluation_role_id="eval-role-99",
    candidates=(
        Candidate(id="staged-pat", resume=Resume(text="Built backend services in Python for six years.")),
        Candidate(id="staged-sam", resume=Resume(text="Led a design team with a marketing background.")),
    ),
)


@pytest.fixture
def preset_client(tmp_path):
    app = create_app(
        ScriptedModelClient(), runs_dir=tmp_path / "runs", demo_presets=[_PRESET]
    )
    app.testing = True
    return app.test_client()


def _staged_ids(html_text: str) -> list[str]:
    return re.findall(r'name="staged_resumes" value="([^"]+)"', html_text)


def test_the_landing_page_offers_every_demo_preset(preset_client):
    out = preset_client.get("/").get_data(as_text=True)

    assert "Backend Engineer" in out
    assert "Software and Data" in out
    assert "Two Python Resumes against a designer." in out
    assert 'href="/?preset=backend-engineer"' in out


def test_picking_a_preset_prefills_the_role_and_stages_its_resumes(preset_client):
    """A Preset is prefilled input, not a shortcut: the Job Description
    lands in the form the Recruiter still has to submit, and the staged
    Resumes are declared, not screened."""
    out = preset_client.get("/?preset=backend-engineer").get_data(as_text=True)

    assert 'value="Backend Engineer"' in out
    assert "Python programming experience" in out
    assert 'name="preset_id" value="backend-engineer"' in out
    assert "Qualified" not in out


def test_an_unknown_preset_id_degrades_to_the_empty_form(preset_client):
    response = preset_client.get("/?preset=no-such-preset")

    assert response.status_code == 200
    out = response.get_data(as_text=True)
    assert 'name="preset_id"' not in out
    assert 'name="job_description"' in out


def test_a_preset_is_screened_end_to_end_without_uploading_anything(preset_client, tmp_path):
    """The whole point of a Preset: a Shortlist with no file picker touched -
    and still the full path, extraction and approval included."""
    proposed_html = preset_client.post(
        "/requirements",
        data={
            "title": _PRESET.role_title,
            "job_description": _PRESET.job_description,
            "preset_id": _PRESET.preset_id,
        },
    ).get_data(as_text=True)
    proposed_ids = _proposed_ids(proposed_html)
    assert 'name="preset_id" value="backend-engineer"' in proposed_html

    approved_html = _approve(
        preset_client,
        title=_PRESET.role_title,
        proposed_ids=proposed_ids,
        proposed_texts=_proposed_texts(proposed_html),
        keep_ids=[proposed_ids[0]],
        preset_id=_PRESET.preset_id,
    ).get_data(as_text=True)
    assert _staged_ids(approved_html) == ["staged-pat", "staged-sam"]

    response = preset_client.post(
        "/screening",
        data={
            "title": _PRESET.role_title,
            "requirements_json": json.dumps(_requirements_json_from(approved_html)),
            "preset_id": _PRESET.preset_id,
            "staged_resumes": ["staged-pat", "staged-sam"],
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    out = response.get_data(as_text=True)
    assert ">staged-pat<" in out and "Qualified" in out
    assert ">staged-sam<" in out and "Disqualified" in out
    # The run is recorded exactly as any other Screening Run is.
    assert len(list((tmp_path / "runs").glob("*.jsonl"))) == 1


def test_a_staged_resume_can_be_left_out_of_the_run(preset_client):
    """The Recruiter chooses which staged Resumes to submit, the same way
    they choose which files to upload - a decision made before the run, not
    a view that hides a Candidate from a Shortlist (ADR-0001)."""
    response = preset_client.post(
        "/screening",
        data={
            "title": _PRESET.role_title,
            "requirements_json": json.dumps([{"id": "req-1", "text": "Python programming experience"}]),
            "preset_id": _PRESET.preset_id,
            "staged_resumes": ["staged-pat"],
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    out = response.get_data(as_text=True)
    assert ">staged-pat<" in out
    assert ">staged-sam<" not in out


def test_a_staged_candidate_id_the_preset_does_not_carry_is_ignored(preset_client):
    """Staged ids come off a form, so they are filtered against the Preset
    rather than trusted to name a Resume it actually stages."""
    response = preset_client.post(
        "/screening",
        data={
            "title": _PRESET.role_title,
            "requirements_json": json.dumps([{"id": "req-1", "text": "Python programming experience"}]),
            "preset_id": _PRESET.preset_id,
            "staged_resumes": ["staged-pat", "someone-elses-resume"],
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    out = response.get_data(as_text=True)
    assert ">staged-pat<" in out
    assert "someone-elses-resume" not in out


def test_uploads_are_screened_alongside_a_presets_staged_resumes(preset_client):
    response = preset_client.post(
        "/screening",
        data={
            "title": _PRESET.role_title,
            "requirements_json": json.dumps([{"id": "req-1", "text": "Python programming experience"}]),
            "preset_id": _PRESET.preset_id,
            "staged_resumes": ["staged-pat"],
            "resumes": [(io.BytesIO(minimal_pdf("Ten years of Python work.")), "alice.pdf")],
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    out = response.get_data(as_text=True)
    assert ">staged-pat<" in out
    assert ">alice<" in out


def test_an_upload_named_after_a_staged_resume_gets_its_own_candidate_id(preset_client):
    """The same collision guard uploads get among themselves: an upload
    called staged-pat.pdf must not be judged against the staged Resume of
    that name in Ranking's resume-by-id lookup."""
    response = preset_client.post(
        "/screening",
        data={
            "title": _PRESET.role_title,
            "requirements_json": json.dumps([{"id": "req-1", "text": "Python programming experience"}]),
            "preset_id": _PRESET.preset_id,
            "staged_resumes": ["staged-pat"],
            "resumes": [(io.BytesIO(minimal_pdf("Ten years of Python work.")), "staged-pat.pdf")],
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    out = response.get_data(as_text=True)
    assert ">staged-pat<" in out
    assert ">staged-pat-2<" in out


def test_screening_a_preset_with_every_staged_resume_dropped_reports_an_error(preset_client):
    response = preset_client.post(
        "/screening",
        data={
            "title": _PRESET.role_title,
            "requirements_json": json.dumps([{"id": "req-1", "text": "Python programming experience"}]),
            "preset_id": _PRESET.preset_id,
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    out = response.get_data(as_text=True)
    assert "No Resume" in out
    # The staged Resumes are still offered, so the Recruiter can re-check one.
    assert _staged_ids(out) == ["staged-pat", "staged-sam"]


def test_the_checked_in_presets_are_offered_by_default(tmp_path):
    """create_app reads the checked-in Presets when it is not handed any,
    so the wiring between the shipped data and the landing page is covered
    and not only the injected-Preset path."""
    app = create_app(ScriptedModelClient(), runs_dir=tmp_path / "runs")

    out = app.test_client().get("/").get_data(as_text=True)

    assert "Payroll and Staff Accountant" in out
    assert 'href="/?preset=devops-engineer"' in out


def test_every_checked_in_preset_is_reachable_from_the_landing_page(tmp_path):
    """Presets are browsed by family group, so a family the page forgot to
    render would hide every Preset in it."""
    presets = load_demo_presets()
    app = create_app(ScriptedModelClient(), runs_dir=tmp_path / "runs")

    out = app.test_client().get("/").get_data(as_text=True)

    for preset in presets:
        assert f'href="/?preset={preset.preset_id}"' in out, preset.preset_id
        assert escape(preset.label) in out


def test_the_family_holding_the_selected_preset_is_expanded(tmp_path):
    presets = load_demo_presets()
    picked = presets[0]
    app = create_app(ScriptedModelClient(), runs_dir=tmp_path / "runs")

    out = app.test_client().get(f"/?preset={picked.preset_id}").get_data(as_text=True)

    expanded = re.findall(r'<details class="family" open>\s*<summary><span class="family-name">([^<]+)', out)
    assert expanded == [picked.family]


def test_no_presets_are_offered_when_none_are_installed(client):
    """The plain ticket 11 flow, which is what an installation without the
    Preset data serves."""
    out = client.get("/").get_data(as_text=True)

    assert "demo Preset" not in out
    assert 'name="job_description"' in out


# --- Reading a Resume from the Shortlist --------------------------------------


def test_every_shortlist_row_offers_the_resume_behind_it(preset_client):
    """A Recruiter checking the system's reasoning against the Resume itself
    is the point of an advisory Shortlist, so every row - Qualified or not -
    carries its Resume, one click away."""
    response = preset_client.post(
        "/screening",
        data={
            "title": _PRESET.role_title,
            "requirements_json": json.dumps([{"id": "req-1", "text": "Python programming experience"}]),
            "preset_id": _PRESET.preset_id,
            "staged_resumes": ["staged-pat", "staged-sam"],
        },
        content_type="multipart/form-data",
    )

    out = response.get_data(as_text=True)
    assert out.count('<details class="resume-peek">') == 2
    assert "Built backend services in Python for six years." in out
    assert "Led a design team with a marketing background." in out


def test_an_uploaded_resume_is_readable_from_its_shortlist_row(client):
    proposed_html = _propose(client).get_data(as_text=True)
    proposed_ids = _proposed_ids(proposed_html)
    approved = _approve(
        client,
        proposed_ids=proposed_ids,
        proposed_texts=_proposed_texts(proposed_html),
        keep_ids=[proposed_ids[0]],
    )
    requirements_json = _requirements_json_from(approved.get_data(as_text=True))

    response = _upload_pdfs(
        client,
        title="Backend Engineer",
        requirements_json=requirements_json,
        files=[(minimal_pdf("Built backend services in Python for six years."), "alice.pdf")],
    )

    out = response.get_data(as_text=True)
    assert '<details class="resume-peek">' in out
    assert "Built backend services in Python for six years." in out


def test_a_qualified_row_says_ranking_only_saw_the_redacted_resume(preset_client):
    """Reading the Resume here must not leave the impression Ranking read it
    too - blind Ranking is ADR-0005, and the row that shows a Fit judgement
    is exactly where that matters."""
    response = preset_client.post(
        "/screening",
        data={
            "title": _PRESET.role_title,
            "requirements_json": json.dumps([{"id": "req-1", "text": "Python programming experience"}]),
            "preset_id": _PRESET.preset_id,
            "staged_resumes": ["staged-pat"],
        },
        content_type="multipart/form-data",
    )

    out = response.get_data(as_text=True)
    assert "Qualified" in out
    assert "redacted form of this Resume" in out


def test_resume_text_is_escaped_rather_than_rendered(preset_client):
    """A Resume is a document supplied from outside, and it lands on the page
    verbatim - so it must never be able to bring markup with it."""
    preset = replace(
        _PRESET,
        candidates=(
            Candidate(
                id="staged-pat",
                resume=Resume(text="<script>alert('resume')</script> Python programming"),
            ),
        ),
    )
    app = create_app(ScriptedModelClient(), demo_presets=[preset])
    app.testing = True

    response = app.test_client().post(
        "/screening",
        data={
            "title": preset.role_title,
            "requirements_json": json.dumps([{"id": "req-1", "text": "Python programming"}]),
            "preset_id": preset.preset_id,
            "staged_resumes": ["staged-pat"],
        },
        content_type="multipart/form-data",
    )

    out = response.get_data(as_text=True)
    assert "<script>alert" not in out
    assert "&lt;script&gt;alert" in out
