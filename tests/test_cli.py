"""The CLI is the third consumer of the core's public entry point, alongside
the evaluation harness and the web UI. These tests drive it end to end
against a scripted model client, with no network access.
"""

from __future__ import annotations

import json
from pathlib import Path

from screening.cli import main


def _write_role(path: Path, *, approved: bool) -> None:
    path.write_text(
        json.dumps(
            {
                "id": "role-1",
                "title": "Backend Engineer",
                "approved": approved,
                "requirements": [
                    {"id": "req-python", "text": "Python programming experience"},
                ],
            }
        )
    )


def test_cli_drives_a_complete_run_and_reports_where_the_record_was_written(
    tmp_path: Path, capsys
):
    role_path = tmp_path / "role.json"
    _write_role(role_path, approved=True)

    resumes_dir = tmp_path / "resumes"
    resumes_dir.mkdir()
    (resumes_dir / "alice.txt").write_text(
        "Built backend services in Python for six years."
    )
    (resumes_dir / "bob.txt").write_text("Led a design team with a marketing background.")

    runs_dir = tmp_path / "runs"

    exit_code = main(
        ["--role", str(role_path), "--resumes", str(resumes_dir), "--runs-dir", str(runs_dir)]
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


def test_cli_run_record_includes_zeroed_metrics_for_the_scripted_client(tmp_path: Path):
    role_path = tmp_path / "role.json"
    _write_role(role_path, approved=True)

    resumes_dir = tmp_path / "resumes"
    resumes_dir.mkdir()
    (resumes_dir / "alice.txt").write_text("Built backend services in Python for six years.")

    runs_dir = tmp_path / "runs"

    exit_code = main(
        ["--role", str(role_path), "--resumes", str(resumes_dir), "--runs-dir", str(runs_dir)]
    )

    assert exit_code == 0
    [run_file] = list(runs_dir.glob("*.jsonl"))
    header = json.loads(run_file.read_text().splitlines()[0])
    assert header["parse_failure_count"] == 0
    assert header["parse_failure_rate"] == 0.0
    assert header["cache_hit_tokens"] == 0
    assert header["cache_miss_tokens"] == 0


def test_cli_live_flag_without_an_api_key_errors_before_any_run(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    # A role path that does not exist: if the CLI validated --live *after*
    # loading the role, this would blow up with a file-not-found traceback
    # instead of the clean MissingApiKey error this test expects.
    role_path = tmp_path / "does-not-exist.json"

    resumes_dir = tmp_path / "resumes"
    runs_dir = tmp_path / "runs"

    exit_code = main(
        [
            "--role",
            str(role_path),
            "--resumes",
            str(resumes_dir),
            "--runs-dir",
            str(runs_dir),
            "--live",
        ]
    )

    assert exit_code == 1
    assert not list(runs_dir.glob("*.jsonl"))


def test_cli_refuses_to_run_against_an_unapproved_requirement_set(tmp_path: Path):
    role_path = tmp_path / "role.json"
    _write_role(role_path, approved=False)

    resumes_dir = tmp_path / "resumes"
    resumes_dir.mkdir()
    (resumes_dir / "alice.txt").write_text("Built backend services in Python.")

    runs_dir = tmp_path / "runs"

    exit_code = main(
        ["--role", str(role_path), "--resumes", str(resumes_dir), "--runs-dir", str(runs_dir)]
    )

    assert exit_code == 1
    assert not list(runs_dir.glob("*.jsonl"))
