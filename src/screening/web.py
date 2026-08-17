"""The web UI: a thin, server-rendered demo surface for a Recruiter (ticket
11), added last per the spec's Language and shape decision. It consumes the
core through exactly the entry points the CLI and the evaluation harness
use - screening.extraction.extract_requirements and
screening.core.run_screening - and adds nothing beyond them: no reject
action, no threshold or cutoff control, and no view that hides a Candidate
from the Recruiter, per ADR-0001.

A completed Screening Run is recorded via FileRunStore exactly as the CLI
records one (spec:56, spec:150), so a Recruiter using this demo surface
gets the same durable record, not a UI-only view of it.

The Recruiter moves through four steps - supply a Job Description, review
and approve the proposed Requirements, upload Resumes, read the Shortlist -
each its own page. State travels between them in hidden form fields rather
than server-side session storage, so the app holds no per-Recruiter state
of its own between requests: every request response is a pure function of
what the previous form posted.

Demo Presets (ticket 12) fill that first form in and stage a batch of
Resumes for the third step, so demonstrating the system does not begin with
pasting text and hunting for files. A Preset is prefilled input and nothing
more: extraction, review, approval and Screening all still run, every part
of it stays editable, and the only state a Preset adds to the wizard is its
id travelling in a hidden field like everything else.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import NamedTuple
from uuid import uuid4

from flask import Flask, request
from markupsafe import Markup, escape
from werkzeug.datastructures import FileStorage, MultiDict

from screening.core import run_screening
from screening.demo_presets import DemoPreset, group_by_family, load_demo_presets
from screening.domain import (
    Candidate,
    Disqualified,
    Fit,
    JobDescription,
    Qualified,
    Requirement,
    Role,
    Shortlist,
    ShortlistEntry,
    ScreeningOutcome,
    approve_requirement_set,
    match_outcome,
)
from screening.extraction import extract_requirements
from screening.model_client import ModelClient, ModelClientError
from screening.pdf_adapter import PdfExtractionError, extract_resume
from screening.store import FileRunStore, ScreeningRun

DEFAULT_RUNS_DIR = Path("runs")


def create_app(
    model_client: ModelClient,
    *,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    demo_presets: Sequence[DemoPreset] | None = None,
) -> Flask:
    """`runs_dir` is where completed Screening Runs are recorded (spec:56,
    spec:150) - the same durable, append-only record the CLI writes via
    FileRunStore, so a Recruiter using this demo surface can revisit how a
    decision was reached weeks later, not just read the Shortlist once.

    `demo_presets` defaults to whatever is checked in, read once here rather
    than per request. Passing an empty sequence serves the UI with no
    Presets offered, which is also what an installation missing the Preset
    data degrades to.
    """
    app = Flask(__name__)
    run_store = FileRunStore(runs_dir)
    presets = tuple(load_demo_presets() if demo_presets is None else demo_presets)
    preset_by_id = {preset.preset_id: preset for preset in presets}

    def selected_preset(preset_id: str | None) -> DemoPreset | None:
        """The Preset a request names, or None - including when it names one
        that does not exist, so a stale link degrades to the plain form
        rather than erroring. Nothing here builds a path from the id."""
        return preset_by_id.get(preset_id) if preset_id else None

    @app.get("/")
    def role_form() -> str:
        preset = selected_preset(request.args.get("preset"))
        if preset is None:
            return _role_form_page(presets=presets)
        return _role_form_page(
            presets=presets,
            preset=preset,
            title=preset.role_title,
            job_description_text=preset.job_description,
        )

    @app.post("/requirements")
    def propose_requirements() -> tuple[str, int] | str:
        title = request.form["title"].strip()
        job_description_text = request.form["job_description"].strip()
        preset = selected_preset(request.form.get("preset_id"))
        if not title or not job_description_text:
            return (
                _role_form_page(
                    presets=presets,
                    preset=preset,
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
                    presets=presets,
                    preset=preset,
                    title=title,
                    job_description_text=job_description_text,
                    error=f"Could not propose Requirements: {exc}",
                ),
                502,
            )

        rows = [_ReviewRow(id=r.id, text=r.text, keep=True) for r in proposed]
        return _review_requirements_page(
            title=title, rows=rows, additional_requirements="", preset=preset
        )

    @app.post("/requirements/approve")
    def approve_requirements() -> tuple[str, int] | str:
        title = request.form["title"].strip()
        rows = _rows_from_review_form(request.form)
        additional_requirements = request.form.get("additional_requirements", "")
        preset = selected_preset(request.form.get("preset_id"))

        if any(row.keep and not row.text for row in rows):
            return (
                _review_requirements_page(
                    title=title,
                    rows=rows,
                    additional_requirements=additional_requirements,
                    preset=preset,
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
                    preset=preset,
                    error="Approve at least one Requirement before continuing.",
                ),
                400,
            )

        return _upload_resumes_page(
            title=title,
            requirements=requirements,
            preset=preset,
            staged_candidate_ids=_all_candidate_ids(preset),
        )

    @app.post("/screening")
    def screen() -> tuple[str, int] | str:
        title = request.form["title"].strip()
        requirements = _requirements_from_json(request.form["requirements_json"])
        preset = selected_preset(request.form.get("preset_id"))
        staged_ids = _staged_candidate_ids(request.form, preset)
        staged = _staged_candidates(preset, staged_ids)
        uploaded, unreadable = _candidates_from_uploads(
            request.files.getlist("resumes"), taken_ids=frozenset(c.id for c in staged)
        )
        candidates = staged + uploaded

        if not candidates:
            return (
                _upload_resumes_page(
                    title=title,
                    requirements=requirements,
                    preset=preset,
                    staged_candidate_ids=staged_ids,
                    error=_no_candidates_error(unreadable),
                ),
                400,
            )

        role = Role(
            id=str(uuid4()), title=title, requirement_set=approve_requirement_set(requirements)
        )
        shortlist = run_screening(role, candidates, model_client)

        run_store.save(
            ScreeningRun(
                run_id=str(uuid4()),
                role=role,
                shortlist=shortlist,
                created_at=datetime.now(timezone.utc),
                metrics=replace(model_client.metrics),
            )
        )

        return _shortlist_page(
            title=title,
            shortlist=shortlist,
            unreadable=unreadable,
            resume_text_by_id={c.id: c.resume.text for c in candidates},
        )

    return app


# --- Step 1: supply a Job Description -------------------------------------


def _role_form_page(
    *,
    presets: Sequence[DemoPreset] = (),
    preset: DemoPreset | None = None,
    title: str = "",
    job_description_text: str = "",
    error: str | None = None,
) -> str:
    body = f"""
    {_presets_html(presets, preset)}
    <section class="card">
      <h2>Set up a Role</h2>
      {_prefilled_note_html(preset)}
      {_error_html(error)}
      <form method="post" action="/requirements">
        {_preset_id_field(preset)}
        <label class="field">
          <span class="field-label">Role title</span>
          <input type="text" name="title" value="{escape(title)}"
            placeholder="Payroll and Staff Accountant" required>
        </label>
        <label class="field">
          <span class="field-label">Job Description</span>
          <span class="field-hint">The Requirements are drawn from this text.
          Nothing is gated on until you have approved them yourself.</span>
          <textarea name="job_description" rows="14" required
            >{escape(job_description_text)}</textarea>
        </label>
        <div class="actions">
          <button type="submit" class="primary">Propose Requirements</button>
        </div>
      </form>
    </section>
    """
    return _layout("Set up a Role", body, step=1)


def _presets_html(presets: Sequence[DemoPreset], selected: DemoPreset | None) -> Markup:
    """The Preset browser: one collapsed group per family, since one Preset
    per field is far too many cards to lay out flat above the form. The
    selected Preset's family is expanded, so picking one still shows where
    it sits and what else is next to it."""
    if not presets:
        return Markup("")
    families = Markup("").join(
        _family_html(family, group, selected) for family, group in group_by_family(presets)
    )
    return Markup(
        """<section class="card presets">
      <h2>Start from a demo Preset</h2>
      <p class="muted">{} Presets, one per field the Resume corpus covers.
      Each fills in a Job Description and stages a batch of Resumes - two from
      the field and two from elsewhere - so a demo does not start with pasting
      text. Every part of it stays editable, and nothing runs until you say
      so.</p>
      {}
    </section>"""
    ).format(len(presets), families)


def _family_html(
    family: str, presets: Sequence[DemoPreset], selected: DemoPreset | None
) -> Markup:
    holds_selection = selected is not None and any(
        preset.preset_id == selected.preset_id for preset in presets
    )
    cards = Markup("").join(_preset_card_html(preset, selected) for preset in presets)
    return Markup(
        """<details class="family"{}>
      <summary><span class="family-name">{}</span>
      <span class="family-count">{}</span></summary>
      <div class="preset-grid">{}</div>
    </details>"""
    ).format(Markup(" open") if holds_selection else Markup(""), family, len(presets), cards)


def _preset_card_html(preset: DemoPreset, selected: DemoPreset | None) -> Markup:
    is_selected = selected is not None and selected.preset_id == preset.preset_id
    return Markup(
        """<a class="preset{}" href="/?preset={}"{}>
      <span class="preset-label">{}</span>
      <span class="preset-summary">{}</span>
      <span class="preset-meta">{} Resumes staged</span>
    </a>"""
    ).format(
        Markup(" selected") if is_selected else Markup(""),
        preset.preset_id,
        Markup(' aria-current="true"') if is_selected else Markup(""),
        preset.label,
        preset.summary,
        len(preset.candidates),
    )


def _prefilled_note_html(preset: DemoPreset | None) -> Markup:
    if preset is None:
        return Markup("")
    return Markup(
        '<p class="note">Prefilled from the <strong>{}</strong> Preset, with '
        "{} Resumes staged for step 3. Edit anything below, or "
        '<a href="/">start from an empty form</a>.</p>'
    ).format(preset.label, len(preset.candidates))


def _preset_id_field(preset: DemoPreset | None) -> Markup:
    """Carries the active Preset through the wizard the same way every other
    piece of state travels - a hidden field, no server-side session."""
    if preset is None:
        return Markup("")
    return Markup('<input type="hidden" name="preset_id" value="{}">').format(preset.preset_id)


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
    preset: DemoPreset | None = None,
    error: str | None = None,
) -> str:
    row_items = Markup("").join(_review_row_html(row) for row in rows)
    body = f"""
    <section class="card">
      <h2>{escape(title)}</h2>
      <p class="muted">Review each proposed Requirement below. Uncheck one to
      drop it, edit its text, or add Requirements the extraction missed.
      Nothing here is approved until you submit.</p>
      {_error_html(error)}
      <form method="post" action="/requirements/approve">
        <input type="hidden" name="title" value="{escape(title)}">
        {_preset_id_field(preset)}
        <ul class="requirements">
          {row_items}
        </ul>
        <label class="field">
          <span class="field-label">Additional Requirements</span>
          <span class="field-hint">One per line.</span>
          <textarea name="additional_requirements" rows="4"
            >{escape(additional_requirements)}</textarea>
        </label>
        <div class="actions">
          <button type="submit" class="primary">Approve Requirement Set</button>
        </div>
      </form>
    </section>
    """
    return _layout("Review Requirements", body, step=2)


def _review_row_html(row: _ReviewRow) -> Markup:
    checked = Markup("checked") if row.keep else Markup("")
    return Markup(
        """<li class="requirement">
          <input type="hidden" name="proposed_ids" value="{}">
          <label class="keep">
            <input type="checkbox" name="keep-{}" {}>
            <span>Keep</span>
          </label>
          <input type="text" name="text-{}" value="{}" aria-label="Requirement text">
        </li>"""
    ).format(row.id, row.id, checked, row.id, row.text)


def _rows_from_review_form(form: MultiDict[str, str]) -> list[_ReviewRow]:
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


class UnreadableUpload(NamedTuple):
    filename: str
    error: str


def _all_candidate_ids(preset: DemoPreset | None) -> list[str]:
    return [] if preset is None else [candidate.id for candidate in preset.candidates]


def _staged_candidate_ids(form: MultiDict[str, str], preset: DemoPreset | None) -> list[str]:
    """Which of the Preset's staged Resumes the Recruiter kept checked, in
    the Preset's own order. Ids are filtered against the Preset rather than
    trusted, so the form cannot introduce a Candidate the Preset does not
    carry.
    """
    kept = set(form.getlist("staged_resumes"))
    return [candidate_id for candidate_id in _all_candidate_ids(preset) if candidate_id in kept]


def _staged_candidates(preset: DemoPreset | None, staged_ids: Sequence[str]) -> list[Candidate]:
    if preset is None:
        return []
    kept = set(staged_ids)
    return [candidate for candidate in preset.candidates if candidate.id in kept]


def _candidates_from_uploads(
    uploads: Sequence[FileStorage], *, taken_ids: frozenset[str] = frozenset()
) -> tuple[list[Candidate], list[UnreadableUpload]]:
    """Converts uploaded PDFs into Candidates through the adapter. Candidate
    ids are disambiguated on a filename collision - two Resumes uploaded as
    e.g. "resume.pdf" would otherwise collide in ranking.py's
    resume-by-id lookup and one Candidate would silently be judged against
    the other's Resume text. `taken_ids` extends that to the ids a Preset's
    staged Resumes already occupy, for the same reason.
    """
    candidates: list[Candidate] = []
    unreadable: list[UnreadableUpload] = []
    seen_ids: set[str] = set(taken_ids)
    for uploaded in uploads:
        if not uploaded.filename:
            continue
        try:
            resume = extract_resume(uploaded.read())
        except PdfExtractionError as exc:
            unreadable.append(UnreadableUpload(uploaded.filename, str(exc)))
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


def _no_candidates_error(unreadable: Sequence[UnreadableUpload]) -> str:
    if not unreadable:
        return "No Resume files were uploaded."
    listed = "; ".join(f"{u.filename} ({u.error})" for u in unreadable)
    return f"No Resume could be read: {listed}"


def _upload_resumes_page(
    *,
    title: str,
    requirements: Sequence[Requirement],
    preset: DemoPreset | None = None,
    staged_candidate_ids: Sequence[str] = (),
    error: str | None = None,
) -> str:
    requirement_items = Markup("").join(
        Markup("<li>{}</li>").format(r.text) for r in requirements
    )
    requirements_json = json.dumps([{"id": r.id, "text": r.text} for r in requirements])
    staged_html = _staged_resumes_html(preset, staged_candidate_ids)
    upload_required = Markup("") if staged_html else Markup("required")
    body = f"""
    <section class="card">
      <h2>{escape(title)}</h2>
      <p class="muted">Approved Requirement Set - what every Candidate will be
      tested against:</p>
      <ul class="requirement-list">{requirement_items}</ul>
    </section>
    <section class="card">
      <h2>Resumes</h2>
      {_error_html(error)}
      <form method="post" action="/screening" enctype="multipart/form-data">
        <input type="hidden" name="title" value="{escape(title)}">
        <input type="hidden" name="requirements_json" value="{escape(requirements_json)}">
        {_preset_id_field(preset)}
        {staged_html}
        <label class="field">
          <span class="field-label">{"Add more Resumes" if staged_html else "Resumes"} (PDF)</span>
          <input type="file" name="resumes" accept="application/pdf" multiple {upload_required}>
        </label>
        <div class="actions">
          <button type="submit" class="primary">Run Screening</button>
        </div>
      </form>
    </section>
    """
    return _layout("Resumes", body, step=3)


def _staged_resumes_html(
    preset: DemoPreset | None, staged_candidate_ids: Sequence[str]
) -> Markup:
    if preset is None:
        return Markup("")
    kept = set(staged_candidate_ids)
    items = Markup("").join(
        Markup(
            '<li><label><input type="checkbox" name="staged_resumes" value="{}" {}>'
            '<span class="candidate-id">{}</span></label></li>'
        ).format(
            candidate.id,
            Markup("checked") if candidate.id in kept else Markup(""),
            candidate.id,
        )
        for candidate in preset.candidates
    )
    return Markup(
        """<fieldset class="staged">
      <legend>{} Resumes staged by the {} Preset</legend>
      <p class="field-hint">These Resumes are checked into the repo, so a demo
      needs no files to hand. Uncheck one to leave it out of this run.</p>
      <ul class="staged-list">{}</ul>
    </fieldset>"""
    ).format(len(preset.candidates), preset.label, items)


def _requirements_from_json(raw: str) -> tuple[Requirement, ...]:
    return tuple(Requirement(id=item["id"], text=item["text"]) for item in json.loads(raw))


# --- Step 4: the Shortlist ---------------------------------------------------


def _shortlist_page(
    *,
    title: str,
    shortlist: Shortlist,
    unreadable: Sequence[UnreadableUpload],
    resume_text_by_id: dict[str, str],
) -> str:
    body = f"""
    <section class="card">
      <h2>{escape(title)}</h2>
      <p class="muted">Every Candidate submitted appears below, whatever
      Screening found - nothing here is rejected, filtered, or hidden. The
      system advises; every decision about a Candidate stays with you.</p>
      {_tally_html(shortlist)}
    </section>
    {_unreadable_html(unreadable)}
    {_entries_html(shortlist, resume_text_by_id)}
    {_comparisons_html(shortlist.comparisons)}
    <p class="actions"><a class="button" href="/">Start a new Role</a></p>
    """
    return _layout("Shortlist", body, step=4)


def _tally_html(shortlist: Shortlist) -> Markup:
    counts = {"Qualified": 0, "Disqualified": 0, "Unresolved": 0}
    for entry in shortlist.entries:
        counts[_status_label(entry.outcome)] += 1
    tiles = Markup("").join(
        Markup('<div class="tile {}"><span class="tile-value">{}</span>'
               '<span class="tile-label">{}</span></div>').format(
            status.lower(), count, status
        )
        for status, count in counts.items()
    )
    return Markup('<div class="tally">{}</div>').format(tiles)


def _entries_html(shortlist: Shortlist, resume_text_by_id: dict[str, str]) -> Markup:
    rank = 0
    rows: list[Markup] = []
    for entry in shortlist.entries:
        resume_text = resume_text_by_id.get(entry.candidate_id, "")
        if isinstance(entry.outcome, Qualified):
            rank += 1
            rows.append(_entry_html(entry, rank=rank, resume_text=resume_text))
        else:
            rows.append(_entry_html(entry, rank=None, resume_text=resume_text))
    return Markup('<ol class="entries">{}</ol>').format(Markup("").join(rows))


def _entry_html(entry: ShortlistEntry, *, rank: int | None, resume_text: str) -> Markup:
    status = _status_label(entry.outcome)
    marker = Markup('<span class="rank">{}</span>').format(rank) if rank else Markup(
        '<span class="rank unranked" aria-hidden="true">-</span>'
    )
    return Markup(
        """<li class="entry {}">
      <div class="entry-head">
        {}
        <span class="candidate-id">{}</span>
        <span class="pill {}">{}</span>
        {}
      </div>
      <div class="entry-detail">{}</div>
    </li>"""
    ).format(
        status.lower(), marker, entry.candidate_id, status.lower(), status,
        _resume_html(entry, resume_text), _outcome_detail(entry.outcome),
    )


def _resume_html(entry: ShortlistEntry, resume_text: str) -> Markup:
    """The Resume behind each row, one click away and collapsed by default.

    A Recruiter checking the system's reasoning against the Resume itself is
    the point of an advisory Shortlist (ADR-0001), so this is the Resume as
    submitted rather than the redacted form - redaction keeps identity
    signals from the model (ADR-0005), not from the Recruiter, who supplied
    the document in the first place. It is rendered inline rather than behind
    its own route because the app holds no state between requests: the text
    is here, in the response to the form that submitted it.
    """
    if not resume_text:
        return Markup("")
    note = (
        "Ranking only ever saw a redacted form of this Resume (ADR-0005)."
        if isinstance(entry.outcome, Qualified)
        else "As submitted."
    )
    return Markup(
        """<details class="resume-peek">
      <summary>Read Resume</summary>
      <p class="field-hint">{}</p>
      <div class="resume-text">{}</div>
    </details>"""
    ).format(note, resume_text)


def _comparisons_html(comparisons: Sequence) -> Markup:
    if not comparisons:
        return Markup("")
    items = Markup("").join(
        Markup(
            '<li><span class="candidate-id">{}</span> over '
            '<span class="candidate-id">{}</span>: {}</li>'
        ).format(c.winner_id, c.loser_id, c.justification)
        for c in comparisons
    )
    return Markup(
        '<section class="card comparisons">'
        "<h2>How the top of the Shortlist was ordered</h2>"
        '<p class="muted">The top band is ordered by judging Candidates '
        "against each other, so each decision carries its own reason.</p>"
        "<ul>{}</ul></section>"
    ).format(items)


def _unreadable_html(unreadable: Sequence[UnreadableUpload]) -> Markup:
    if not unreadable:
        return Markup("")
    items = Markup("").join(
        Markup("<li><strong>{}</strong> - {}</li>").format(u.filename, u.error)
        for u in unreadable
    )
    return Markup(
        '<section class="card unreadable"><h2>Could not be read</h2>'
        '<p class="muted">Reported rather than dropped, so nobody disappears '
        "between what you sent and what you read.</p><ul>{}</ul></section>"
    ).format(items)


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
        unresolved=lambda o: Markup('<p class="reason">{}</p>').format(o.reason),
    )


def _qualified_detail(outcome: Qualified) -> Markup:
    fit: Fit | None = outcome.fit
    if fit is None:
        return Markup(
            '<p class="reason">Qualified, but Ranking produced no verdict - '
            "so this Candidate is unranked rather than placed.</p>"
        )
    items = Markup("").join(
        Markup(
            '<li class="dimension"><span class="dimension-name">{}</span>'
            '<span class="rating rating-{}">{}</span>'
            '<span class="dimension-why">{}</span></li>'
        ).format(_humanised(d.name), d.rating, d.rating, d.justification)
        for d in fit.dimensions
    )
    return Markup('<ul class="dimensions">{}</ul>').format(items)


def _humanised(dimension_name: str) -> str:
    return dimension_name.replace("_", " ").capitalize()


def _disqualified_detail(outcome: Disqualified) -> Markup:
    justification_by_id = {v.requirement_id: v.justification for v in outcome.verdicts}
    items = Markup("").join(
        Markup(
            '<li class="missed"><span class="missed-what">{}</span>'
            '<span class="missed-why">{}</span></li>'
        ).format(req.text, justification_by_id.get(req.id, ""))
        for req in outcome.missed
    )
    return Markup(
        '<p class="reason">Missed {} of the approved Requirements:</p>'
        '<ul class="missed-list">{}</ul>'
    ).format(len(outcome.missed), items)


# --- Shared layout ------------------------------------------------------------

_STEPS = ("Job Description", "Requirements", "Resumes", "Shortlist")

_STYLE = """
:root {
  --bg: #f6f7f9;
  --surface: #ffffff;
  --surface-2: #f0f2f5;
  --border: #dce0e6;
  --text: #16191d;
  --muted: #5b636e;
  --accent: #2f5bd7;
  --accent-text: #ffffff;
  --ok: #1c7a4a;
  --ok-soft: #e4f4ea;
  --warn: #9a5b00;
  --warn-soft: #fdf1de;
  --neutral: #5b636e;
  --neutral-soft: #eceef1;
  --radius: 10px;
  --shadow: 0 1px 2px rgba(16, 20, 26, 0.06), 0 1px 8px rgba(16, 20, 26, 0.04);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #101317;
    --surface: #191d23;
    --surface-2: #21262e;
    --border: #2e343d;
    --text: #e8eaee;
    --muted: #a0a8b4;
    --accent: #6d94ff;
    --accent-text: #0d1015;
    --ok: #6fd39b;
    --ok-soft: #17311f;
    --warn: #e8b163;
    --warn-soft: #33260f;
    --neutral: #a0a8b4;
    --neutral-soft: #262b33;
    --shadow: 0 1px 2px rgba(0, 0, 0, 0.4);
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 16px;
  line-height: 1.55;
}
a { color: var(--accent); }
.topbar {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 0.9rem 1.25rem;
}
.topbar-inner, .wrap { max-width: 58rem; margin: 0 auto; }
.topbar h1 { font-size: 1.05rem; margin: 0; letter-spacing: 0.01em; }
.topbar p { margin: 0.15rem 0 0; color: var(--muted); font-size: 0.85rem; }
.wrap { padding: 1.5rem 1.25rem 4rem; }
.steps {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  list-style: none;
  margin: 0 0 1.5rem;
  padding: 0;
  font-size: 0.85rem;
}
.steps li {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.35rem 0.7rem 0.35rem 0.4rem;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface);
  color: var(--muted);
}
.steps .step-num {
  display: grid;
  place-items: center;
  width: 1.4rem;
  height: 1.4rem;
  border-radius: 50%;
  background: var(--surface-2);
  color: var(--muted);
  font-size: 0.75rem;
  font-weight: 600;
}
.steps li.done { color: var(--text); }
.steps li.done .step-num { background: var(--ok-soft); color: var(--ok); }
.steps li.current { border-color: var(--accent); color: var(--text); font-weight: 600; }
.steps li.current .step-num { background: var(--accent); color: var(--accent-text); }
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 1.25rem 1.4rem 1.4rem;
  margin-bottom: 1.25rem;
}
.card h2 { font-size: 1.15rem; margin: 0 0 0.5rem; }
.muted { color: var(--muted); margin: 0 0 1rem; }
.note {
  background: var(--surface-2);
  border-left: 3px solid var(--accent);
  border-radius: 6px;
  padding: 0.6rem 0.85rem;
  margin: 0 0 1rem;
  font-size: 0.9rem;
}
.error {
  background: var(--warn-soft);
  border-left: 3px solid var(--warn);
  border-radius: 6px;
  color: var(--warn);
  font-weight: 600;
  padding: 0.6rem 0.85rem;
  margin: 0 0 1rem;
}
.field { display: block; margin: 0 0 1.1rem; }
.field-label { display: block; font-weight: 600; font-size: 0.9rem; margin-bottom: 0.25rem; }
.field-hint { display: block; color: var(--muted); font-size: 0.85rem; margin-bottom: 0.4rem; }
input[type=text], textarea, input[type=file] {
  width: 100%;
  font: inherit;
  color: var(--text);
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 7px;
  padding: 0.55rem 0.65rem;
}
textarea { resize: vertical; line-height: 1.5; }
input[type=text]:focus, textarea:focus, input[type=file]:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 1px;
  border-color: var(--accent);
}
input[type=file] { padding: 0.45rem; background: var(--surface-2); }
input[type=file]::file-selector-button {
  font: inherit;
  font-weight: 600;
  margin-right: 0.7rem;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface);
  color: var(--text);
  padding: 0.3rem 0.7rem;
  cursor: pointer;
}
.actions { margin-top: 1.2rem; }
button, .button {
  font: inherit;
  font-weight: 600;
  border: 1px solid var(--border);
  border-radius: 7px;
  background: var(--surface-2);
  color: var(--text);
  padding: 0.55rem 1.1rem;
  cursor: pointer;
  text-decoration: none;
  display: inline-block;
}
button.primary { background: var(--accent); border-color: var(--accent); color: var(--accent-text); }
button:hover, .button:hover { filter: brightness(1.06); }
.family {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface-2);
  margin-bottom: 0.5rem;
}
.family > summary {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
  padding: 0.6rem 0.85rem;
  font-weight: 600;
  border-radius: var(--radius);
  list-style: none;
}
.family > summary::-webkit-details-marker { display: none; }
.family > summary::before {
  content: "";
  width: 0.42rem;
  height: 0.42rem;
  border-right: 2px solid var(--muted);
  border-bottom: 2px solid var(--muted);
  transform: rotate(-45deg);
  transition: transform 120ms ease;
}
.family[open] > summary::before { transform: rotate(45deg); }
.family > summary:hover { color: var(--accent); }
.family[open] > summary { border-bottom: 1px solid var(--border); border-radius: var(--radius) var(--radius) 0 0; }
.family-count {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 999px;
  color: var(--muted);
  font-size: 0.75rem;
  font-weight: 700;
  padding: 0.02rem 0.5rem;
}
.family .preset-grid { padding: 0.85rem; }
.preset-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(20rem, 1fr));
  align-items: stretch;
  gap: 0.75rem;
}
.preset {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
  padding: 0.75rem 0.85rem;
  text-decoration: none;
  color: var(--text);
}
.preset:hover { border-color: var(--accent); }
.preset.selected { border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }
.preset-label { font-weight: 600; }
.preset-summary { color: var(--muted); font-size: 0.82rem; }
.preset-meta { margin-top: auto; color: var(--accent); font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
.requirements { list-style: none; padding: 0; margin: 0 0 1.1rem; }
.requirement {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 7px;
  padding: 0.4rem 0.55rem;
  margin: 0 0 0.4rem;
}
.keep {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  color: var(--muted);
  font-size: 0.78rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  cursor: pointer;
  user-select: none;
}
.requirement input[type=text] { flex: 1; background: var(--surface); }
.requirement-list { margin: 0; padding-left: 1.2rem; }
.requirement-list li { margin: 0.2rem 0; }
.staged { border: 1px solid var(--border); border-radius: var(--radius); padding: 0.85rem 1rem 1rem; margin: 0 0 1.2rem; }
.staged legend { font-weight: 600; font-size: 0.9rem; padding: 0 0.35rem; }
.staged .field-hint { margin: 0.1rem 0 0.7rem; }
.staged-list { list-style: none; margin: 0; padding: 0; display: grid; gap: 0.35rem; }
.staged-list label { display: flex; align-items: center; gap: 0.55rem; font-size: 0.92rem; }
.tally { display: flex; flex-wrap: wrap; gap: 0.6rem; }
.tile {
  flex: 1 1 7rem;
  border: 1px solid var(--border);
  border-left-width: 4px;
  border-radius: 8px;
  background: var(--surface-2);
  padding: 0.6rem 0.8rem;
}
.tile-value { display: block; font-size: 1.45rem; font-weight: 700; line-height: 1.2; }
.tile-label { color: var(--muted); font-size: 0.82rem; }
.tile.qualified { border-left-color: var(--ok); }
.tile.disqualified { border-left-color: var(--warn); }
.tile.unresolved { border-left-color: var(--neutral); }
.entries { list-style: none; margin: 0; padding: 0; }
.entry {
  background: var(--surface);
  border: 1px solid var(--border);
  border-left-width: 4px;
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 0.9rem 1.1rem 1rem;
  margin-bottom: 0.75rem;
}
.entry.qualified { border-left-color: var(--ok); }
.entry.disqualified { border-left-color: var(--warn); }
.entry.unresolved { border-left-color: var(--neutral); }
.entry-head { display: flex; align-items: center; gap: 0.65rem; flex-wrap: wrap; }
.rank {
  display: grid;
  place-items: center;
  min-width: 1.75rem;
  height: 1.75rem;
  border-radius: 50%;
  background: var(--accent);
  color: var(--accent-text);
  font-size: 0.85rem;
  font-weight: 700;
}
.rank.unranked { background: var(--neutral-soft); color: var(--muted); }
.candidate-id { font-weight: 600; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.92rem; }
.pill {
  border-radius: 999px;
  padding: 0.12rem 0.6rem;
  font-size: 0.78rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.pill.qualified { background: var(--ok-soft); color: var(--ok); }
.pill.disqualified { background: var(--warn-soft); color: var(--warn); }
.pill.unresolved { background: var(--neutral-soft); color: var(--neutral); }
.entry-detail { margin-top: 0.7rem; }
.resume-peek { margin-left: auto; }
.resume-peek > summary {
  display: inline-flex;
  cursor: pointer;
  list-style: none;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface-2);
  color: var(--muted);
  font-size: 0.78rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  padding: 0.12rem 0.7rem;
  white-space: nowrap;
}
.resume-peek > summary::-webkit-details-marker { display: none; }
.resume-peek > summary:hover { border-color: var(--accent); color: var(--accent); }
.resume-peek[open] { flex: 1 1 100%; margin-left: 0; order: 9; margin-top: 0.6rem; }
.resume-peek[open] > summary { color: var(--accent); border-color: var(--accent); }
.resume-peek .field-hint { margin: 0.45rem 0 0.3rem; }
.resume-text {
  max-height: 22rem;
  overflow: auto;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 7px;
  padding: 0.65rem 0.8rem;
  font-size: 0.85rem;
  line-height: 1.6;
  white-space: pre-wrap;
  overflow-wrap: break-word;
}
.reason { color: var(--muted); margin: 0 0 0.5rem; font-size: 0.9rem; }
.dimensions, .missed-list { list-style: none; margin: 0; padding: 0; display: grid; gap: 0.4rem; }
.dimension, .missed {
  display: grid;
  grid-template-columns: 9.5rem 6.5rem 1fr;
  align-items: baseline;
  gap: 0.6rem;
  background: var(--surface-2);
  border-radius: 7px;
  padding: 0.45rem 0.6rem;
  font-size: 0.9rem;
}
.missed { grid-template-columns: 16rem 1fr; }
.dimension-name, .missed-what { font-weight: 600; }
.dimension-why, .missed-why { color: var(--muted); }
.rating {
  justify-self: start;
  border-radius: 999px;
  padding: 0.05rem 0.55rem;
  font-size: 0.76rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  background: var(--neutral-soft);
  color: var(--neutral);
}
.rating-strong, .rating-exceptional { background: var(--ok-soft); color: var(--ok); }
.rating-moderate { background: var(--warn-soft); color: var(--warn); }
.comparisons ul { margin: 0; padding-left: 1.2rem; }
.comparisons li { margin: 0.3rem 0; }
.unreadable { border-left: 4px solid var(--warn); }
@media (max-width: 34rem) {
  .dimension, .missed { grid-template-columns: 1fr; gap: 0.2rem; }
}
"""


def _error_html(error: str | None) -> Markup:
    if not error:
        return Markup("")
    return Markup('<p class="error">{}</p>').format(error)


def _steps_html(current: int) -> Markup:
    items = Markup("").join(
        Markup('<li class="{}"><span class="step-num">{}</span>'
               '<span class="step-name">{}</span></li>').format(
            "current" if number == current else ("done" if number < current else ""),
            number,
            name,
        )
        for number, name in enumerate(_STEPS, start=1)
    )
    return Markup('<ol class="steps">{}</ol>').format(items)


def _layout(page_title: str, body_html: str, *, step: int) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(page_title)} - Resume Screening</title>
<style>{_STYLE}</style>
</head>
<body>
<header class="topbar">
  <div class="topbar-inner">
    <h1>Resume Screening</h1>
    <p>Advisory only - the system orders Candidates and shows its reasons;
    every decision stays with you.</p>
  </div>
</header>
<main class="wrap">
{_steps_html(step)}
{body_html}
</main>
</body>
</html>"""


# --- Standalone entry point --------------------------------------------------

LIVE_ENV_VAR = "SCREENING_WEB_LIVE"
RUNS_DIR_ENV_VAR = "SCREENING_RUNS_DIR"


def main() -> int:
    """Runs the demo server directly, against the scripted model client by
    default - the same non-network default the CLI uses - or against the
    real deepseek-v4-flash provider when SCREENING_WEB_LIVE=1 is set,
    mirroring the CLI's --live flag.
    """
    import os
    import sys

    from screening.deepseek_client import MissingApiKey, build_deepseek_client
    from screening.scripted_client import ScriptedModelClient

    model_client: ModelClient
    if os.environ.get(LIVE_ENV_VAR) == "1":
        try:
            model_client = build_deepseek_client(usage=f"to run with {LIVE_ENV_VAR}=1")
        except MissingApiKey as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    else:
        model_client = ScriptedModelClient()

    runs_dir = Path(os.environ[RUNS_DIR_ENV_VAR]) if RUNS_DIR_ENV_VAR in os.environ else DEFAULT_RUNS_DIR
    app = create_app(model_client, runs_dir=runs_dir)
    app.run(port=int(os.environ.get("PORT", "5000")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
