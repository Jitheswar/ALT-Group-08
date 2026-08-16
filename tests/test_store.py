"""The store is a swap point, not a test double: these tests run against its
real implementation on a temporary directory.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from screening.domain import (
    Disqualified,
    Qualified,
    Requirement,
    RequirementSet,
    RequirementVerdict,
    Role,
    Shortlist,
    ShortlistEntry,
    Unresolved,
)
from screening.store import FileRunStore, RunAlreadyExists, ScreeningRun

REQ_PYTHON = Requirement(id="req-python", text="5+ years of Python")


def _role() -> Role:
    return Role(
        id="role-1",
        title="Backend Engineer",
        requirement_set=RequirementSet(requirements=(REQ_PYTHON,), approved=True),
    )


def _shortlist() -> Shortlist:
    return Shortlist(
        entries=(
            ShortlistEntry(
                candidate_id="alice",
                outcome=Qualified(
                    verdicts=(
                        RequirementVerdict(
                            requirement_id="req-python",
                            met=True,
                            justification='Resume cites "8 years of Python"',
                        ),
                    )
                ),
            ),
            ShortlistEntry(
                candidate_id="bob",
                outcome=Disqualified(
                    verdicts=(
                        RequirementVerdict(
                            requirement_id="req-python",
                            met=False,
                            justification="No Python listed",
                        ),
                    ),
                    missed=(REQ_PYTHON,),
                ),
            ),
            ShortlistEntry(
                candidate_id="cara",
                outcome=Unresolved(reason="Screening produced no valid verdict"),
            ),
        )
    )


def _run(run_id: str = "run-1") -> ScreeningRun:
    return ScreeningRun(
        run_id=run_id,
        role=_role(),
        shortlist=_shortlist(),
        created_at=datetime.now(timezone.utc),
    )


def test_save_writes_one_file_per_run(tmp_path: Path):
    store = FileRunStore(tmp_path)

    path = store.save(_run())

    assert path == tmp_path / "run-1.jsonl"
    assert path.exists()


def test_saved_run_records_the_role_and_every_shortlist_entry(tmp_path: Path):
    store = FileRunStore(tmp_path)

    path = store.save(_run())

    lines = path.read_text().splitlines()
    header = json.loads(lines[0])
    assert header["run_id"] == "run-1"
    assert header["role"]["id"] == "role-1"

    entries = [json.loads(line) for line in lines[1:]]
    assert [e["candidate_id"] for e in entries] == ["alice", "bob", "cara"]
    assert entries[0]["outcome"]["type"] == "qualified"
    assert entries[1]["outcome"]["type"] == "disqualified"
    assert entries[1]["outcome"]["missed"][0]["id"] == "req-python"
    assert entries[2]["outcome"]["type"] == "unresolved"


def test_a_run_is_never_overwritten_once_saved(tmp_path: Path):
    store = FileRunStore(tmp_path)
    store.save(_run())

    with pytest.raises(RunAlreadyExists):
        store.save(_run())


def test_a_failed_write_does_not_permanently_block_the_run_id(tmp_path: Path, monkeypatch):
    import screening.store as store_module

    real_dumps = json.dumps
    state = {"calls": 0}

    def flaky_dumps(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 2:
            raise ValueError("boom")
        return real_dumps(*args, **kwargs)

    monkeypatch.setattr(store_module.json, "dumps", flaky_dumps)
    store = FileRunStore(tmp_path)

    with pytest.raises(ValueError):
        store.save(_run())

    assert not list(tmp_path.glob("*.jsonl"))

    monkeypatch.setattr(store_module.json, "dumps", real_dumps)
    path = store.save(_run())

    assert path.exists()


def test_saved_run_records_parse_failure_and_cache_token_counts(tmp_path: Path):
    store = FileRunStore(tmp_path)
    run = ScreeningRun(
        run_id="run-1",
        role=_role(),
        shortlist=_shortlist(),
        created_at=datetime.now(timezone.utc),
        parse_failure_count=2,
        parse_failure_rate=0.25,
        cache_hit_tokens=1200,
        cache_miss_tokens=300,
    )

    path = store.save(run)

    header = json.loads(path.read_text().splitlines()[0])
    assert header["parse_failure_count"] == 2
    assert header["parse_failure_rate"] == 0.25
    assert header["cache_hit_tokens"] == 1200
    assert header["cache_miss_tokens"] == 300


def test_a_run_with_no_reported_metrics_defaults_them_to_zero(tmp_path: Path):
    store = FileRunStore(tmp_path)

    path = store.save(_run())

    header = json.loads(path.read_text().splitlines()[0])
    assert header["parse_failure_count"] == 0
    assert header["parse_failure_rate"] == 0.0
    assert header["cache_hit_tokens"] == 0
    assert header["cache_miss_tokens"] == 0


def test_different_runs_get_their_own_file(tmp_path: Path):
    store = FileRunStore(tmp_path)

    first = store.save(_run(run_id="run-1"))
    second = store.save(_run(run_id="run-2"))

    assert first != second
    assert first.exists()
    assert second.exists()
