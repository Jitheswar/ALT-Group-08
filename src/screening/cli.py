from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from screening.core import RequirementSetNotApproved, run_screening
from screening.domain import (
    Candidate,
    Requirement,
    RequirementSet,
    Resume,
    Role,
    ScreeningOutcome,
    match_outcome,
)
from screening.scripted_client import ScriptedModelClient
from screening.store import FileRunStore, ScreeningRun


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="screening",
        description="Run a Screening Run against a Role and a batch of Resumes.",
    )
    parser.add_argument("--role", required=True, type=Path, help="Path to a Role JSON file")
    parser.add_argument(
        "--resumes", required=True, type=Path, help="Directory of .txt Resume files"
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Directory Screening Run records are written to",
    )
    args = parser.parse_args(argv)

    role = _load_role(args.role)
    candidates = _load_candidates(args.resumes)

    try:
        shortlist = run_screening(role, candidates, ScriptedModelClient())
    except RequirementSetNotApproved as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    run = ScreeningRun(
        run_id=str(uuid4()),
        role=role,
        shortlist=shortlist,
        created_at=datetime.now(timezone.utc),
    )
    path = FileRunStore(args.runs_dir).save(run)

    for entry in shortlist.entries:
        print(f"{entry.candidate_id}: {_describe(entry.outcome)}")
    print(f"Screening Run record written to {path}")
    return 0


def _load_role(path: Path) -> Role:
    data = json.loads(path.read_text())
    requirements = tuple(
        Requirement(id=r["id"], text=r["text"]) for r in data["requirements"]
    )
    requirement_set = RequirementSet(
        requirements=requirements, approved=data.get("approved", False)
    )
    return Role(id=data["id"], title=data["title"], requirement_set=requirement_set)


def _load_candidates(resumes_dir: Path) -> list[Candidate]:
    return [
        Candidate(id=path.stem, resume=Resume(text=path.read_text()))
        for path in sorted(resumes_dir.glob("*.txt"))
    ]


def _describe(outcome: ScreeningOutcome) -> str:
    return match_outcome(
        outcome,
        qualified=lambda o: "Qualified",
        disqualified=lambda o: f"Disqualified (missed: {', '.join(r.text for r in o.missed)})",
        unresolved=lambda o: f"Unresolved ({o.reason})",
    )


if __name__ == "__main__":
    sys.exit(main())
