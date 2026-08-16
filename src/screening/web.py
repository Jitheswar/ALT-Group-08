"""The web UI: a thin, server-rendered demo surface for a Recruiter (ticket
11), added last per the spec's Language and shape decision. It consumes the
core through exactly the entry points the CLI and the evaluation harness
use - screening.extraction.extract_requirements and
screening.core.run_screening - and adds nothing beyond them: no reject
action, no threshold or cutoff control, and no view that hides a Candidate
from the Recruiter, per ADR-0001.

The Recruiter moves through four steps - supply a Job Description, review
and approve the proposed Requirements, upload Resumes, read the Shortlist -
each its own page. State travels between them in hidden form fields rather
than server-side session storage, so the app holds no per-Recruiter state
of its own between requests: every request response is a pure function of
what the previous form posted.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import PurePath
from typing import NamedTuple
from uuid import uuid4

from flask import Flask, request
from markupsafe import Markup, escape
from werkzeug.datastructures import FileStorage

from screening.core import run_screening
from screening.domain import (
    Candidate,
    Fit,
    JobDescription,
    Requirement,
    Role,
    Shortlist,
    ScreeningOutcome,
    approve_requirement_set,
    match_outcome,
)
from screening.extraction import extract_requirements
from screening.model_client import ModelClient, ModelClientError
from screening.pdf_adapter import PdfExtractionError, extract_resume


def create_app(model_client: ModelClient) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def role_form() -> str:
        return _role_form_page()

    @app.post("/requirements")
    def propose_requirements() -> tuple[str, int] | str:
        title = request.form["title"].strip()
        job_description_text = request.form["job_description"].strip()
        if not title or not job_description_text:
            return (
                _role_form_page(
                    title=title,
                    job_description_text=job_description_text,
                    error="A Role title and a Job Description are both required.",
                ),
                400,
            )

        try:
            proposed = extract_requirements(
                JobDescription(text=job_description_text), model_client
            )
        except ModelClientError as exc:
            return (
                _role_form_page(
                    title=title,
                    job_description_text=job_description_text,
                    error=f"Could not propose Requirements: {exc}",
                ),
                502,
            )

        rows = [_ReviewRow(id=r.id, text=r.text, keep=True) for r in proposed]
        return _review_requirements_page(title=title, rows=rows, additional_requirements="")

    @app.post("/requirements/approve")
    def approve_requirements() -> tuple[str, int] | str:
        title = request.form["title"].strip()
        rows = _rows_from_review_form(request.form)
        additional_requirements = request.form.get("additional_requirements", "")

        if any(row.keep and not row.text for row in rows):
            return (
                _review_requirements_page(
                    title=title,
                    rows=rows,
                    additional_requirements=additional_requirements,
                    error="A kept Requirement cannot have empty text - edit it or uncheck it to drop it.",
                ),
                400,
            )

        requirements = _approved_requirements(rows, additional_requirements)
        if not requirements:
            return (
                _review_requirements_page(
                    title=title,
                    rows=rows,
                    additional_requirements=additional_requirements,
                    error="Approve at least one Requirement before continuing.",
                ),
                400,
            )

        return _upload_resumes_page(title=title, requirements=requirements)

    @app.post("/screening")
    def screen() -> tuple[str, int] | str:
        title = request.form["title"].strip()
        requirements = _requirements_from_json(request.form["requirements_json"])
        candidates, unreadable = _candidates_from_uploads(request.files.getlist("resumes"))

        if not candidates:
            return (
                _upload_resumes_page(
                    title=title, requirements=requirements, error=_no_candidates_error(unreadable)
                ),
                400,
            )

        role = Role(
            id=str(uuid4()), title=title, requirement_set=approve_requirement_set(requirements)
        )
        shortlist = run_screening(role, candidates, model_client)
        return _shortlist_page(title=title, shortlist=shortlist, unreadable=unreadable)

    return app


# --- Step 1: supply a Job Description -------------------------------------


def _role_form_page(
    *, title: str = "", job_description_text: str = "", error: str | None = None
) -> str:
    body = f"""
    {_error_html(error)}
    <form method="post" action="/requirements">
      <label>Role title
        <input type="text" name="title" value="{escape(title)}" required>
      </label>
      <label>Job Description
        <textarea name="job_description" rows="14" required
          >{escape(job_description_text)}</textarea>
      </label>
      <button type="submit">Propose Requirements</button>
    </form>
    """
    return _layout("Set up a Role", body)


# --- Step 2: review and approve the proposed Requirements ------------------


class _ReviewRow(NamedTuple):
    id: str
    text: str
    keep: bool


def _review_requirements_page(
    *,
    title: str,
    rows: Sequence[_ReviewRow],
    additional_requirements: str,
    error: str | None = None,
) -> str:
    row_items = Markup("").join(_review_row_html(row) for row in rows)
    body = f"""
    <h2>{escape(title)}</h2>
    {_error_html(error)}
    <p>Review each proposed Requirement below. Uncheck one to drop it,
    edit its text, or add Requirements the extraction missed. Nothing
    here is approved until you submit.</p>
    <form method="post" action="/requirements/approve">
      <input type="hidden" name="title" value="{escape(title)}">
      <ul class="requirements">
        {row_items}
      </ul>
      <label>Additional Requirements (one per line)
        <textarea name="additional_requirements" rows="4"
          >{escape(additional_requirements)}</textarea>
      </label>
      <button type="submit">Approve Requirement Set</button>
    </form>
    """
    return _layout("Review Requirements", body)


def _review_row_html(row: _ReviewRow) -> Markup:
    checked = Markup("checked") if row.keep else Markup("")
    return Markup(
        """<li>
          <input type="hidden" name="proposed_ids" value="{}">
          <label>
            <input type="checkbox" name="keep-{}" {}>
            <input type="text" name="text-{}" value="{}">
          </label>
        </li>"""
    ).format(row.id, row.id, checked, row.id, row.text)


def _rows_from_review_form(form) -> list[_ReviewRow]:
    return [
        _ReviewRow(
            id=req_id,
            text=form.get(f"text-{req_id}", "").strip(),
            keep=form.get(f"keep-{req_id}") == "on",
        )
        for req_id in form.getlist("proposed_ids")
    ]


def _approved_requirements(
    rows: Sequence[_ReviewRow], additional_requirements: str
) -> tuple[Requirement, ...]:
    """Assumes no kept row has empty text - the route rejects that combination
    before this is ever called, so a kept Requirement is never silently
    dropped for having nothing left to approve."""
    kept = tuple(Requirement(id=row.id, text=row.text) for row in rows if row.keep)
    added = tuple(
        Requirement(id=f"req-custom-{index}", text=line)
        for index, raw_line in enumerate(additional_requirements.splitlines(), start=1)
        if (line := raw_line.strip())
    )
    return kept + added


# --- Step 3: upload Resumes -------------------------------------------------


def _candidates_from_uploads(
    uploads: Sequence[FileStorage],
) -> tuple[list[Candidate], list[tuple[str, str]]]:
    """Converts uploaded PDFs into Candidates through the adapter. Candidate
    ids are disambiguated on a filename collision - two Resumes uploaded as
    e.g. "resume.pdf" would otherwise collide in ranking.py's
    resume-by-id lookup and one Candidate would silently be judged against
    the other's Resume text.
    """
    candidates: list[Candidate] = []
    unreadable: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    for uploaded in uploads:
        if not uploaded.filename:
            continue
        try:
            resume = extract_resume(uploaded.read())
        except PdfExtractionError as exc:
            unreadable.append((uploaded.filename, str(exc)))
            continue
        candidate_id = _unique_candidate_id(PurePath(uploaded.filename).stem, seen_ids)
        seen_ids.add(candidate_id)
        candidates.append(Candidate(id=candidate_id, resume=resume))
    return candidates, unreadable


def _unique_candidate_id(stem: str, seen: set[str]) -> str:
    if stem not in seen:
        return stem
    suffix = 2
    while f"{stem}-{suffix}" in seen:
        suffix += 1
    return f"{stem}-{suffix}"


def _no_candidates_error(unreadable: Sequence[tuple[str, str]]) -> str:
    if not unreadable:
        return "No Resume files were uploaded."
    listed = "; ".join(f"{name} ({message})" for name, message in unreadable)
    return f"No Resume could be read: {listed}"


def _upload_resumes_page(
    *, title: str, requirements: Sequence[Requirement], error: str | None = None
) -> str:
    requirement_items = Markup("").join(
        Markup("<li>{}</li>").format(r.text) for r in requirements
    )
    requirements_json = json.dumps([{"id": r.id, "text": r.text} for r in requirements])
    body = f"""
    <h2>{escape(title)}</h2>
    {_error_html(error)}
    <p>Approved Requirement Set:</p>
    <ul>{requirement_items}</ul>
    <form method="post" action="/screening" enctype="multipart/form-data">
      <input type="hidden" name="title" value="{escape(title)}">
      <input type="hidden" name="requirements_json" value="{escape(requirements_json)}">
      <label>Resumes (PDF)
        <input type="file" name="resumes" accept="application/pdf" multiple required>
      </label>
      <button type="submit">Run Screening</button>
    </form>
    """
    return _layout("Upload Resumes", body)


def _requirements_from_json(raw: str) -> tuple[Requirement, ...]:
    return tuple(Requirement(id=item["id"], text=item["text"]) for item in json.loads(raw))


# --- Step 4: the Shortlist ---------------------------------------------------


def _shortlist_page(
    *, title: str, shortlist: Shortlist, unreadable: Sequence[tuple[str, str]]
) -> str:
    unreadable_html = _unreadable_html(unreadable)
    rows_html = Markup("").join(_shortlist_row_html(entry) for entry in shortlist.entries)
    body = f"""
    <h2>{escape(title)} - Shortlist</h2>
    <p>Every Candidate submitted appears below, whatever Screening found -
    nothing here is rejected, filtered, or hidden.</p>
    {unreadable_html}
    <table>
      <thead><tr><th>Candidate</th><th>Outcome</th><th>Justification</th></tr></thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
    <p><a href="/">Start a new Role</a></p>
    """
    return _layout("Shortlist", body)


def _unreadable_html(unreadable: Sequence[tuple[str, str]]) -> Markup:
    if not unreadable:
        return Markup("")
    items = Markup("").join(
        Markup("<li>{} - {}</li>").format(name, message) for name, message in unreadable
    )
    return Markup(
        '<section class="unreadable"><h3>Could not be read</h3><ul>{}</ul></section>'
    ).format(items)


def _shortlist_row_html(entry) -> Markup:
    status = _status_label(entry.outcome)
    return Markup('<tr class="{}"><td>{}</td><td>{}</td><td>{}</td></tr>').format(
        status.lower(), entry.candidate_id, status, _outcome_detail(entry.outcome)
    )


def _status_label(outcome: ScreeningOutcome) -> str:
    return match_outcome(
        outcome,
        qualified=lambda o: "Qualified",
        disqualified=lambda o: "Disqualified",
        unresolved=lambda o: "Unresolved",
    )


def _outcome_detail(outcome: ScreeningOutcome) -> Markup:
    return match_outcome(
        outcome,
        qualified=_qualified_detail,
        disqualified=_disqualified_detail,
        unresolved=lambda o: Markup("<span>{}</span>").format(o.reason),
    )


def _qualified_detail(outcome) -> Markup:
    fit: Fit | None = outcome.fit
    if fit is None:
        return Markup("<em>Qualified, but Ranking produced no verdict.</em>")
    items = Markup("").join(
        Markup("<li><strong>{}</strong> ({}): {}</li>").format(
            d.name, d.rating, d.justification
        )
        for d in fit.dimensions
    )
    return Markup("<ul>{}</ul>").format(items)


def _disqualified_detail(outcome) -> Markup:
    justification_by_id = {v.requirement_id: v.justification for v in outcome.verdicts}
    items = Markup("").join(
        Markup("<li>Missed <strong>{}</strong>: {}</li>").format(
            req.text, justification_by_id.get(req.id, "")
        )
        for req in outcome.missed
    )
    return Markup("<ul>{}</ul>").format(items)


# --- Shared layout ------------------------------------------------------------

_STYLE = """
body { font-family: system-ui, sans-serif; max-width: 60rem; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }
label { display: block; margin: 1rem 0; }
input[type=text], textarea { width: 100%; font: inherit; padding: 0.4rem; box-sizing: border-box; }
.requirements { list-style: none; padding: 0; }
.requirements li { margin: 0.5rem 0; }
.requirements label { display: flex; align-items: center; gap: 0.5rem; margin: 0; }
.requirements input[type=text] { flex: 1; }
button { font: inherit; padding: 0.5rem 1rem; margin-top: 1rem; cursor: pointer; }
.error { color: #a00; font-weight: bold; }
table { border-collapse: collapse; width: 100%; margin-top: 1rem; }
th, td { border: 1px solid #ccc; padding: 0.5rem; text-align: left; vertical-align: top; }
tr.qualified td:first-child { border-left: 4px solid #2a7a2a; }
tr.disqualified td:first-child { border-left: 4px solid #a06a00; }
tr.unresolved td:first-child { border-left: 4px solid #888; }
.unreadable { background: #fff4e5; padding: 0.75rem 1rem; margin-top: 1rem; }
"""


def _error_html(error: str | None) -> Markup:
    if not error:
        return Markup("")
    return Markup('<p class="error">{}</p>').format(error)


def _layout(page_title: str, body_html: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{escape(page_title)}</title>
<style>{_STYLE}</style>
</head>
<body>
<header><h1>Resume Screening</h1></header>
<main>
{body_html}
</main>
</body>
</html>"""


# --- Standalone entry point --------------------------------------------------

API_KEY_ENV_VAR = "DEEPSEEK_API_KEY"
LIVE_ENV_VAR = "SCREENING_WEB_LIVE"


def main() -> int:
    """Runs the demo server directly, against the scripted model client by
    default - the same non-network default the CLI uses - or against the
    real deepseek-v4-flash provider when SCREENING_WEB_LIVE=1 is set,
    mirroring the CLI's --live flag.
    """
    import os
    import sys

    from screening.deepseek_client import DeepSeekModelClient, build_live_transport
    from screening.scripted_client import ScriptedModelClient

    model_client: ModelClient
    if os.environ.get(LIVE_ENV_VAR) == "1":
        api_key = os.environ.get(API_KEY_ENV_VAR)
        if not api_key:
            print(f"error: {API_KEY_ENV_VAR} must be set to run with {LIVE_ENV_VAR}=1", file=sys.stderr)
            return 1
        model_client = DeepSeekModelClient(build_live_transport(api_key))
    else:
        model_client = ScriptedModelClient()

    app = create_app(model_client)
    app.run(port=int(os.environ.get("PORT", "5000")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
