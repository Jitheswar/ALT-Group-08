from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from screening.core import run_screening
from screening.deepseek_client import DeepSeekModelClient, build_live_transport
from screening.domain import (
    Candidate,
    Fit,
    JobDescription,
    Requirement,
    Resume,
    Role,
    ScreeningOutcome,
    approve_requirement_set,
    match_outcome,
)
from screening.extraction import extract_requirements
from screening.model_client import ModelClient, ModelClientError, RunMetrics
from screening.scripted_client import ScriptedModelClient
from screening.store import (
    FileExtractionRecordStore,
    FileRunStore,
    RequirementApprovalRecord,
    RequirementProposalRecord,
    ScreeningRun,
)

API_KEY_ENV_VAR = "DEEPSEEK_API_KEY"


class MissingApiKey(Exception):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="screening",
        description="Propose Requirements from a Job Description, or run a Screening Run.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser(
        "extract", help="Propose Requirements from a Job Description for Recruiter review"
    )
    extract_parser.add_argument(
        "--job-description", required=True, type=Path, help="Path to a Job Description text file"
    )
    extract_parser.add_argument(
        "--out", required=True, type=Path, help="Path the proposed Requirements are written to"
    )
    extract_parser.add_argument(
        "--extraction-records-dir",
        type=Path,
        default=Path("extraction-records"),
        help="Directory Requirement Extraction Records are written to",
    )
    extract_parser.add_argument(
        "--live",
        action="store_true",
        help=f"Extract against the real deepseek-v4-flash provider (requires {API_KEY_ENV_VAR})",
    )

    screen_parser = subparsers.add_parser(
        "screen", help="Run a Screening Run against a Recruiter-approved Requirement Set"
    )
    screen_parser.add_argument("--role-id", required=True)
    screen_parser.add_argument("--title", required=True)
    screen_parser.add_argument(
        "--proposed",
        required=True,
        type=Path,
        help="Path to the proposed Requirements written by `extract`",
    )
    screen_parser.add_argument(
        "--approved",
        required=True,
        type=Path,
        help="Path to the Recruiter-reviewed Requirements to approve and gate Screening on",
    )
    screen_parser.add_argument(
        "--resumes", required=True, type=Path, help="Directory of .txt Resume files"
    )
    screen_parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Directory Screening Run records are written to",
    )
    screen_parser.add_argument(
        "--extraction-records-dir",
        type=Path,
        default=Path("extraction-records"),
        help="Directory Requirement Extraction Records are written to (paired with the matching proposal by id)",
    )
    screen_parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Run against the real deepseek-v4-flash provider instead of the "
            f"scripted client (requires {API_KEY_ENV_VAR})"
        ),
    )

    args = parser.parse_args(argv)
    if args.command == "extract":
        return _run_extract(args)
    return _run_screen(args)


def _run_extract(args: argparse.Namespace) -> int:
    try:
        model_client = _build_model_client(live=args.live)
    except MissingApiKey as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    job_description = JobDescription(text=args.job_description.read_text())
    try:
        proposed = extract_requirements(job_description, model_client)
    except ModelClientError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Recorded immediately, before any Recruiter review, so a proposal that
    # is later rejected outright is still on the record (ADR-0004).
    proposal_id = str(uuid4())
    FileExtractionRecordStore(args.extraction_records_dir).save_proposal(
        RequirementProposalRecord(
            proposal_id=proposal_id,
            job_description=job_description,
            proposed=proposed,
            created_at=datetime.now(timezone.utc),
        )
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "proposal_id": proposal_id,
                "proposed": [{"id": r.id, "text": r.text} for r in proposed],
            },
            indent=2,
        )
    )
    print(f"Proposed {len(proposed)} Requirement(s), written to {args.out}")
    print("Review, edit, delete, or add Requirements before approving.")
    return 0


def _run_screen(args: argparse.Namespace) -> int:
    try:
        model_client = _build_model_client(live=args.live)
    except MissingApiKey as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    proposal_id = json.loads(args.proposed.read_text())["proposal_id"]

    approved_data = json.loads(args.approved.read_text())
    approved_requirements = _parse_requirements(approved_data["requirements"])
    requirement_set = approve_requirement_set(approved_requirements)
    role = Role(id=args.role_id, title=args.title, requirement_set=requirement_set)

    candidates = _load_candidates(args.resumes)
    shortlist = run_screening(role, candidates, model_client)

    metrics = getattr(model_client, "metrics", RunMetrics())
    run = ScreeningRun(
        run_id=str(uuid4()),
        role=role,
        shortlist=shortlist,
        created_at=datetime.now(timezone.utc),
        parse_failure_count=metrics.parse_failures,
        parse_failure_rate=metrics.parse_failure_rate,
        cache_hit_tokens=metrics.cache_hit_tokens,
        cache_miss_tokens=metrics.cache_miss_tokens,
    )
    run_path = FileRunStore(args.runs_dir).save(run)

    FileExtractionRecordStore(args.extraction_records_dir).save_approval(
        RequirementApprovalRecord(
            proposal_id=proposal_id,
            role_id=role.id,
            approved=requirement_set,
            created_at=run.created_at,
        )
    )

    for entry in shortlist.entries:
        print(f"{entry.candidate_id}: {_describe(entry.outcome)}")
    print(f"Screening Run record written to {run_path}")
    return 0


def _build_model_client(*, live: bool) -> ModelClient:
    if not live:
        return ScriptedModelClient()

    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key:
        raise MissingApiKey(f"{API_KEY_ENV_VAR} must be set to run with --live")
    return DeepSeekModelClient(build_live_transport(api_key))


def _parse_requirements(items: list[dict]) -> tuple[Requirement, ...]:
    return tuple(Requirement(id=item["id"], text=item["text"]) for item in items)


def _load_candidates(resumes_dir: Path) -> list[Candidate]:
    return [
        Candidate(id=path.stem, resume=Resume(text=path.read_text()))
        for path in sorted(resumes_dir.glob("*.txt"))
    ]


def _describe(outcome: ScreeningOutcome) -> str:
    return match_outcome(
        outcome,
        qualified=lambda o: f"Qualified (fit: {_describe_fit(o.fit)})",
        disqualified=lambda o: f"Disqualified (missed: {', '.join(r.text for r in o.missed)})",
        unresolved=lambda o: f"Unresolved ({o.reason})",
    )


def _describe_fit(fit: Fit | None) -> str:
    if fit is None:
        return "unranked"
    return ", ".join(f"{d.name}={d.rating}" for d in fit.dimensions)


if __name__ == "__main__":
    sys.exit(main())
