from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from screening.core import RequirementSetNotApproved, run_screening
from screening.deepseek_client import DeepSeekModelClient, build_live_transport
from screening.domain import (
    Candidate,
    Requirement,
    RequirementSet,
    Resume,
    Role,
    ScreeningOutcome,
    match_outcome,
)
from screening.model_client import ModelClient, RunMetrics
from screening.scripted_client import ScriptedModelClient
from screening.store import FileRunStore, ScreeningRun

API_KEY_ENV_VAR = "DEEPSEEK_API_KEY"


class MissingApiKey(Exception):
    pass


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
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Run against the real deepseek-v4-flash provider instead of the "
            f"scripted client (requires {API_KEY_ENV_VAR})"
        ),
    )
    args = parser.parse_args(argv)

    try:
        model_client = _build_model_client(live=args.live)
    except MissingApiKey as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    role = _load_role(args.role)
    candidates = _load_candidates(args.resumes)

    try:
        shortlist = run_screening(role, candidates, model_client)
    except RequirementSetNotApproved as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

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
    path = FileRunStore(args.runs_dir).save(run)

    for entry in shortlist.entries:
        print(f"{entry.candidate_id}: {_describe(entry.outcome)}")
    print(f"Screening Run record written to {path}")
    return 0


def _build_model_client(*, live: bool) -> ModelClient:
    if not live:
        return ScriptedModelClient()

    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key:
        raise MissingApiKey(f"{API_KEY_ENV_VAR} must be set to run with --live")
    return DeepSeekModelClient(build_live_transport(api_key))


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
