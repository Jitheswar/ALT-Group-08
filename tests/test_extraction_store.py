"""The extraction record store is a swap point, not a test double: these
tests run against its real implementation on a temporary directory. A
proposal is recorded the moment extraction produces one - independent of
whether it is ever approved - and an approval is recorded separately,
sharing the proposal's id, so extraction quality can be measured on its own
(ADR-0004).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from screening.domain import JobDescription, Requirement, approve_requirement_set
from screening.store import (
    ExtractionRecordAlreadyExists,
    FileExtractionRecordStore,
    RequirementApprovalRecord,
    RequirementProposalRecord,
)

REQ_PYTHON_PROPOSED = Requirement(id="req-1", text="5+ years of Python")
REQ_DEGREE_PROPOSED = Requirement(id="req-2", text="Willing to relocate")
REQ_PYTHON_APPROVED = Requirement(id="req-1", text="3+ years of Python")
REQ_CUSTOM_APPROVED = Requirement(id="req-custom-1", text="Bachelor's degree in CS")


def _proposal(proposal_id: str = "proposal-1") -> RequirementProposalRecord:
    return RequirementProposalRecord(
        proposal_id=proposal_id,
        job_description=JobDescription(text="Backend Engineer\n- 5+ years of Python\n"),
        proposed=(REQ_PYTHON_PROPOSED, REQ_DEGREE_PROPOSED),
        created_at=datetime.now(timezone.utc),
    )


def _approval(proposal_id: str = "proposal-1") -> RequirementApprovalRecord:
    return RequirementApprovalRecord(
        proposal_id=proposal_id,
        role_id="role-1",
        approved=approve_requirement_set((REQ_PYTHON_APPROVED, REQ_CUSTOM_APPROVED)),
        created_at=datetime.now(timezone.utc),
    )


def test_save_proposal_writes_one_file_per_proposal(tmp_path: Path):
    store = FileExtractionRecordStore(tmp_path)

    path = store.save_proposal(_proposal())

    assert path == tmp_path / "proposal-1-proposal.json"
    data = json.loads(path.read_text())
    assert [r["text"] for r in data["proposed"]] == [
        "5+ years of Python",
        "Willing to relocate",
    ]


def test_a_proposal_is_recorded_even_if_never_approved(tmp_path: Path):
    """A recruiter who rejects the whole proposal still leaves a record
    behind, so extraction quality can be measured on proposals as a whole,
    not only on the ones that happened to be approved."""
    store = FileExtractionRecordStore(tmp_path)

    store.save_proposal(_proposal())

    assert list(tmp_path.glob("*-approval.json")) == []
    [proposal_file] = list(tmp_path.glob("*-proposal.json"))
    assert proposal_file.exists()


def test_save_approval_writes_one_file_per_approval_sharing_the_proposal_id(tmp_path: Path):
    store = FileExtractionRecordStore(tmp_path)
    store.save_proposal(_proposal())

    path = store.save_approval(_approval())

    assert path == tmp_path / "proposal-1-approval.json"
    data = json.loads(path.read_text())
    assert data["proposal_id"] == "proposal-1"
    assert data["role_id"] == "role-1"
    assert [r["text"] for r in data["approved"]["requirements"]] == [
        "3+ years of Python",
        "Bachelor's degree in CS",
    ]
    assert data["approved"]["approved"] is True


def test_a_proposal_and_its_approval_can_be_paired_by_proposal_id(tmp_path: Path):
    store = FileExtractionRecordStore(tmp_path)
    store.save_proposal(_proposal())
    store.save_approval(_approval())

    proposal_data = json.loads((tmp_path / "proposal-1-proposal.json").read_text())
    approval_data = json.loads((tmp_path / "proposal-1-approval.json").read_text())

    assert proposal_data["proposal_id"] == approval_data["proposal_id"]
    assert [r["text"] for r in proposal_data["proposed"]] != [
        r["text"] for r in approval_data["approved"]["requirements"]
    ]


def test_a_record_is_never_overwritten_once_saved(tmp_path: Path):
    store = FileExtractionRecordStore(tmp_path)
    store.save_proposal(_proposal())

    with pytest.raises(ExtractionRecordAlreadyExists):
        store.save_proposal(_proposal())


def test_different_proposals_get_their_own_file(tmp_path: Path):
    store = FileExtractionRecordStore(tmp_path)

    first = store.save_proposal(_proposal(proposal_id="proposal-1"))
    second = store.save_proposal(_proposal(proposal_id="proposal-2"))

    assert first != second
    assert first.exists()
    assert second.exists()
