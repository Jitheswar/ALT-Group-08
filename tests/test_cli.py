"""The CLI is the third consumer of the core's public entry points, alongside
the evaluation harness and the web UI. These tests drive it end to end
against a scripted model client, with no network access.

There is no fixture that hands the CLI a pre-approved Requirement Set: the
normal path always goes through `extract` (proposing Requirements from a Job
Description, recorded immediately) and `screen` (gating on a
Recruiter-reviewed, explicitly approved copy of them, recorded separately
and paired to the proposal by id), per ADR-0004.
"""

from __future__ import annotations

import json
from pathlib import Path

from screening.cli import main


def _write_job_description(path: Path) -> None:
    path.write_text(
        "Backend Engineer\n"
        "- Python programming experience\n"
        "- Bachelor's degree in Computer Science\n"
    )


def _extraction_records_dir(tmp_path: Path) -> Path:
    return tmp_path / "extraction-records"


def _run_extract(tmp_path: Path, *, proposed_out: Path | None = None) -> Path:
    job_description_path = tmp_path / "jd.txt"
    _write_job_description(job_description_path)
    out_path = proposed_out or (tmp_path / "proposed.json")

    exit_code = main(
        [
            "extract",
            "--job-description",
            str(job_description_path),
            "--out",
            str(out_path),
            "--extraction-records-dir",
            str(_extraction_records_dir(tmp_path)),
        ]
    )
    assert exit_code == 0
    return out_path


def _write_approved(path: Path, *, requirements: list[dict]) -> None:
    path.write_text(json.dumps({"requirements": requirements}))


def _write_resumes(resumes_dir: Path) -> None:
    resumes_dir.mkdir()
    (resumes_dir / "alice.txt").write_text(
        "Built backend services in Python for six years. B.Sc. Computer Science."
    )
    (resumes_dir / "bob.txt").write_text("Led a design team with a marketing background.")


def _run_screen(tmp_path: Path, *, proposed_path: Path, approved_path: Path, runs_dir: Path) -> int:
    return main(
        [
            "screen",
            "--role-id",
            "role-1",
            "--title",
            "Backend Engineer",
            "--proposed",
            str(proposed_path),
            "--approved",
            str(approved_path),
            "--resumes",
            str(tmp_path / "resumes"),
            "--runs-dir",
            str(runs_dir),
            "--extraction-records-dir",
            str(_extraction_records_dir(tmp_path)),
        ]
    )


def test_cli_extract_writes_discrete_proposed_requirements_for_review(tmp_path: Path):
    proposed_path = _run_extract(tmp_path)

    data = json.loads(proposed_path.read_text())
    assert data["proposal_id"]
    assert [r["text"] for r in data["proposed"]] == [
        "Python programming experience",
        "Bachelor's degree in Computer Science",
    ]
    assert [r["id"] for r in data["proposed"]] == ["req-1", "req-2"]


def test_cli_extract_records_the_proposal_immediately(tmp_path: Path):
    """The proposal is on the record the moment extraction produces it, so a
    proposal a Recruiter goes on to reject outright is still recorded -
    nothing here depends on `screen` ever being run."""
    proposed_path = _run_extract(tmp_path)
    proposal_id = json.loads(proposed_path.read_text())["proposal_id"]

    [proposal_file] = list(_extraction_records_dir(tmp_path).glob("*-proposal.json"))
    assert list(_extraction_records_dir(tmp_path).glob("*-approval.json")) == []

    record = json.loads(proposal_file.read_text())
    assert record["proposal_id"] == proposal_id
    assert [r["text"] for r in record["proposed"]] == [
        "Python programming experience",
        "Bachelor's degree in Computer Science",
    ]


def test_cli_extract_reports_where_the_proposal_was_written(tmp_path: Path, capsys):
    proposed_path = _run_extract(tmp_path)

    out = capsys.readouterr().out
    assert str(proposed_path) in out
    assert "Review, edit, delete, or add" in out


def test_cli_extract_live_flag_without_an_api_key_errors_before_any_extraction(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    # A Job Description path that does not exist: if the CLI validated
    # --live *after* reading the file, this would blow up with a
    # file-not-found traceback instead of the clean MissingApiKey error.
    job_description_path = tmp_path / "does-not-exist.txt"
    out_path = tmp_path / "proposed.json"

    exit_code = main(
        [
            "extract",
            "--job-description",
            str(job_description_path),
            "--out",
            str(out_path),
            "--extraction-records-dir",
            str(_extraction_records_dir(tmp_path)),
            "--live",
        ]
    )

    assert exit_code == 1
    assert not out_path.exists()


def test_cli_screen_drives_a_complete_run_and_reports_where_the_record_was_written(
    tmp_path: Path, capsys
):
    proposed_path = _run_extract(tmp_path)
    approved_path = tmp_path / "approved.json"
    _write_approved(
        approved_path,
        requirements=[
            {"id": "req-1", "text": "Python programming experience"},
            {"id": "req-2", "text": "Bachelor's degree in Computer Science"},
        ],
    )
    _write_resumes(tmp_path / "resumes")
    runs_dir = tmp_path / "runs"

    exit_code = _run_screen(
        tmp_path, proposed_path=proposed_path, approved_path=approved_path, runs_dir=runs_dir
    )

    assert exit_code == 0
    [run_file] = list(runs_dir.glob("*.jsonl"))

    out = capsys.readouterr().out
    assert str(run_file) in out
    assert "alice: Qualified" in out
    assert "bob: Disqualified" in out

    lines = run_file.read_text().splitlines()
    entries = [json.loads(line) for line in lines[1:]]
    assert [e["candidate_id"] for e in entries] == ["alice", "bob"]


def test_cli_screen_records_the_approval_paired_to_its_proposal_by_id(tmp_path: Path):
    proposed_path = _run_extract(tmp_path)
    proposal_id = json.loads(proposed_path.read_text())["proposal_id"]

    approved_path = tmp_path / "approved.json"
    # The Recruiter drops the degree Requirement and adds one extraction missed.
    _write_approved(
        approved_path,
        requirements=[
            {"id": "req-1", "text": "Python programming experience"},
            {"id": "req-custom-1", "text": "Willing to work US hours"},
        ],
    )
    _write_resumes(tmp_path / "resumes")
    runs_dir = tmp_path / "runs"

    exit_code = _run_screen(
        tmp_path, proposed_path=proposed_path, approved_path=approved_path, runs_dir=runs_dir
    )

    assert exit_code == 0
    [approval_file] = list(_extraction_records_dir(tmp_path).glob("*-approval.json"))
    approval = json.loads(approval_file.read_text())

    assert approval["proposal_id"] == proposal_id
    assert approval["role_id"] == "role-1"
    assert [r["text"] for r in approval["approved"]["requirements"]] == [
        "Python programming experience",
        "Willing to work US hours",
    ]
    assert approval["approved"]["approved"] is True

    # The original proposal - including the dropped degree Requirement -
    # is still there, untouched, so extraction quality is measurable
    # against what was actually approved.
    [proposal_file] = list(_extraction_records_dir(tmp_path).glob("*-proposal.json"))
    proposal = json.loads(proposal_file.read_text())
    assert [r["text"] for r in proposal["proposed"]] == [
        "Python programming experience",
        "Bachelor's degree in Computer Science",
    ]


def test_cli_screen_live_flag_without_an_api_key_errors_before_any_run(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    # A --proposed path that does not exist: if the CLI validated --live
    # *after* loading the proposal, this would blow up with a
    # file-not-found traceback instead of the clean MissingApiKey error.
    proposed_path = tmp_path / "does-not-exist.json"
    approved_path = tmp_path / "does-not-exist-either.json"
    runs_dir = tmp_path / "runs"

    exit_code = main(
        [
            "screen",
            "--role-id",
            "role-1",
            "--title",
            "Backend Engineer",
            "--proposed",
            str(proposed_path),
            "--approved",
            str(approved_path),
            "--resumes",
            str(tmp_path / "resumes"),
            "--runs-dir",
            str(runs_dir),
            "--extraction-records-dir",
            str(_extraction_records_dir(tmp_path)),
            "--live",
        ]
    )

    assert exit_code == 1
    assert not list(runs_dir.glob("*.jsonl"))


def test_cli_screen_runs_the_comparative_pass_over_two_or_more_qualified_candidates(
    tmp_path: Path, capsys
):
    """A regression test for the comparative pass added in ticket 05: the
    scripted client backing the CLI's default (non-live) path must itself
    handle a ComparativeResponse call, which only happens once two or more
    Candidates land in the Qualified group - a single Qualified Candidate,
    as in the other CLI tests above, never exercises it.
    """
    proposed_path = _run_extract(tmp_path)
    approved_path = tmp_path / "approved.json"
    _write_approved(
        approved_path,
        requirements=[{"id": "req-1", "text": "Python programming experience"}],
    )
    resumes_dir = tmp_path / "resumes"
    resumes_dir.mkdir()
    (resumes_dir / "alice.txt").write_text("Built backend services in Python for six years.")
    (resumes_dir / "cara.txt").write_text("Wrote Python data pipelines for three years.")
    runs_dir = tmp_path / "runs"

    exit_code = _run_screen(
        tmp_path, proposed_path=proposed_path, approved_path=approved_path, runs_dir=runs_dir
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "alice: Qualified" in out
    assert "cara: Qualified" in out


def test_cli_run_record_includes_zeroed_metrics_for_the_scripted_client(tmp_path: Path):
    proposed_path = _run_extract(tmp_path)
    approved_path = tmp_path / "approved.json"
    _write_approved(
        approved_path,
        requirements=[{"id": "req-1", "text": "Python programming experience"}],
    )
    resumes_dir = tmp_path / "resumes"
    resumes_dir.mkdir()
    (resumes_dir / "alice.txt").write_text("Built backend services in Python for six years.")
    runs_dir = tmp_path / "runs"

    exit_code = _run_screen(
        tmp_path, proposed_path=proposed_path, approved_path=approved_path, runs_dir=runs_dir
    )

    assert exit_code == 0
    [run_file] = list(runs_dir.glob("*.jsonl"))
    header = json.loads(run_file.read_text().splitlines()[0])
    assert header["parse_failure_count"] == 0
    assert header["parse_failure_rate"] == 0.0
    assert header["cache_hit_tokens"] == 0
    assert header["cache_miss_tokens"] == 0
