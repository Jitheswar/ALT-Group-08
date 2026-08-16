"""Screening Run and Requirement Extraction Record persistence, behind
interfaces so flat files can be replaced by a database once cross-run
queries actually require one.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from screening.domain import (
    JobDescription,
    Requirement,
    RequirementSet,
    Role,
    ScreeningOutcome,
    Shortlist,
    ShortlistEntry,
    match_outcome,
)


@dataclass(frozen=True)
class ScreeningRun:
    run_id: str
    role: Role
    shortlist: Shortlist
    created_at: datetime
    parse_failure_count: int = 0
    parse_failure_rate: float = 0.0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0


class RunAlreadyExists(Exception):
    pass


class RunStore(Protocol):
    def save(self, run: ScreeningRun) -> Path: ...


class FileRunStore:
    """One append-only JSONL file per Screening Run. A run's file is created
    exclusively and made read-only once written, so a completed run can
    never be overwritten in place.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def save(self, run: ScreeningRun) -> Path:
        path = self._root / f"{run.run_id}.jsonl"
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as exc:
            raise RunAlreadyExists(
                f"Screening Run {run.run_id!r} already recorded at {path}"
            ) from exc

        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(json.dumps(_header(run)) + "\n")
                for entry in run.shortlist.entries:
                    fh.write(json.dumps(_entry_record(entry)) + "\n")
        except BaseException:
            path.unlink(missing_ok=True)
            raise

        path.chmod(0o444)
        return path


def _header(run: ScreeningRun) -> dict:
    return {
        "run_id": run.run_id,
        "created_at": run.created_at.isoformat(),
        "role": asdict(run.role),
        "parse_failure_count": run.parse_failure_count,
        "parse_failure_rate": run.parse_failure_rate,
        "cache_hit_tokens": run.cache_hit_tokens,
        "cache_miss_tokens": run.cache_miss_tokens,
    }


def _entry_record(entry: ShortlistEntry) -> dict:
    return {
        "candidate_id": entry.candidate_id,
        "outcome": _outcome_record(entry.outcome),
    }


def _outcome_record(outcome: ScreeningOutcome) -> dict:
    return match_outcome(
        outcome,
        qualified=lambda o: {
            "type": "qualified",
            "verdicts": [asdict(v) for v in o.verdicts],
        },
        disqualified=lambda o: {
            "type": "disqualified",
            "verdicts": [asdict(v) for v in o.verdicts],
            "missed": [asdict(r) for r in o.missed],
        },
        unresolved=lambda o: {"type": "unresolved", "reason": o.reason},
    )


@dataclass(frozen=True)
class RequirementProposalRecord:
    """What extraction proposed, recorded the moment it is produced - before
    any Recruiter review - so a proposal the Recruiter goes on to reject
    outright is still recorded, and extraction quality can be measured on
    its own rather than only on the proposals that happened to be approved.
    """

    proposal_id: str
    job_description: JobDescription
    proposed: tuple[Requirement, ...]
    created_at: datetime


@dataclass(frozen=True)
class RequirementApprovalRecord:
    """What the Recruiter approved, sharing its originating proposal's id so
    the two can be paired to measure extraction quality against approvals
    (ADR-0004).
    """

    proposal_id: str
    role_id: str
    approved: RequirementSet
    created_at: datetime


class ExtractionRecordAlreadyExists(Exception):
    pass


class ExtractionRecordStore(Protocol):
    def save_proposal(self, record: RequirementProposalRecord) -> Path: ...
    def save_approval(self, record: RequirementApprovalRecord) -> Path: ...


class FileExtractionRecordStore:
    """One append-only JSON file per proposal and per approval. Each file is
    created exclusively and made read-only once written, mirroring
    FileRunStore. Proposal and approval files for the same extraction event
    share their proposal_id, so they can be paired without either one being
    mutated after the fact.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def save_proposal(self, record: RequirementProposalRecord) -> Path:
        return self._write(
            self._root / f"{record.proposal_id}-proposal.json",
            {
                "proposal_id": record.proposal_id,
                "created_at": record.created_at.isoformat(),
                "job_description": asdict(record.job_description),
                "proposed": [asdict(r) for r in record.proposed],
            },
        )

    def save_approval(self, record: RequirementApprovalRecord) -> Path:
        return self._write(
            self._root / f"{record.proposal_id}-approval.json",
            {
                "proposal_id": record.proposal_id,
                "role_id": record.role_id,
                "created_at": record.created_at.isoformat(),
                "approved": asdict(record.approved),
            },
        )

    def _write(self, path: Path, payload: dict) -> Path:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as exc:
            raise ExtractionRecordAlreadyExists(
                f"Requirement Extraction Record already recorded at {path}"
            ) from exc

        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(json.dumps(payload))
        except BaseException:
            path.unlink(missing_ok=True)
            raise

        path.chmod(0o444)
        return path
