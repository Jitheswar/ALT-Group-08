"""Generates the checked-in Evaluation Roles (ticket 06, ADR-0008).

Two phases, because the sample-review step genuinely needs a human and
can't be scripted away:

    propose   Matches every resume-atlas category to a real posting from
              the postings corpus (screening.eval_roles.match_all_categories),
              runs live extraction against each matched Job Description, and
              writes the 43 proposals plus the reviewable mapping to --out.
              Nothing here is checked in yet - a proposal is what extraction
              produced, unreviewed.

    finalize  Reads the proposals written by `propose` and a review file
              (JSON: {category: {"reviewed": bool, "requirements": [{"id",
              "text"}]}}) produced by a human going through a sample of at
              least 10, and writes the 43 checked-in Evaluation Role files.
              A category absent from the review file is auto-approved as
              extraction produced it - the blanket auto-approval ADR-0008
              warns would make Requirement Set review decorative. That is a
              deliberate, bounded exception for the categories outside the
              required review sample, not a rejection of the ADR's concern:
              reviewed_sample_size is gated at check-in and reported
              alongside every result, so a reader can see how much of the
              43 a human actually reviewed rather than assume all of it.

Usage:
    uv run --group data python scripts/generate_evaluation_roles.py propose \
        --resume-atlas-parquet PATH --postings-parquet PATH \
        --out data/evaluation_roles/.proposals.json \
        --mapping-out data/evaluation_roles/mapping.md

    uv run python scripts/generate_evaluation_roles.py finalize \
        --proposals data/evaluation_roles/.proposals.json \
        --review REVIEW.json \
        --reviewed-sample-size 10 \
        --out-dir data/evaluation_roles/roles \
        --metadata-out data/evaluation_roles/metadata.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from screening.deepseek_client import MissingApiKey, build_deepseek_client
from screening.domain import JobDescription, Requirement, Role, approve_requirement_set
from screening.eval_roles import (
    CATEGORY_SEARCH_TERMS,
    CategoryMatch,
    EvaluationRole,
    PostingCandidate,
    PostingProvenance,
    match_all_categories,
    write_evaluation_role,
)
from screening.extraction import extract_requirements
from screening.model_client import ModelClientError
from screening.resume_atlas import RESUME_ATLAS_DATASET

POSTINGS_DATASET = "NextGig-Rocks/global-job-postings-multi-ats"
MIN_REVIEWED_SAMPLE_SIZE = 10  # ADR-0008: "reviewed on a sample of at least 10 of the 43"


def load_resume_atlas_categories(parquet_path: Path) -> list[str]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["Category"])
    categories = sorted(set(table.column("Category").to_pylist()))
    return categories


def _as_text(value: object) -> str:
    """Postings corpus qualification columns are either a plain string, a
    list of bullet strings, or null. Normalized to one string per row so
    PostingCandidate never has to know the corpus's column types.
    """
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(f"- {item}" for item in value)
    return str(value)


def load_postings(parquet_path: Path) -> list[PostingCandidate]:
    import pyarrow.parquet as pq

    columns = [
        "title",
        "normalized_title",
        "company_name",
        "job_description",
        "minimum_qualifications",
        "preferred_qualifications",
        "skills_required",
        "education_level",
        "certifications",
        "experience_level",
    ]
    table = pq.read_table(parquet_path, columns=columns)
    cols = {name: table.column(name).to_pylist() for name in columns}
    return [
        PostingCandidate(
            posting_id=str(index),
            title=cols["title"][index] or "",
            normalized_title=cols["normalized_title"][index] or "",
            company_name=cols["company_name"][index] or "",
            job_description=cols["job_description"][index] or "",
            minimum_qualifications=_as_text(cols["minimum_qualifications"][index]),
            preferred_qualifications=_as_text(cols["preferred_qualifications"][index]),
            skills_required=_as_text(cols["skills_required"][index]),
            education_level=_as_text(cols["education_level"][index]),
            certifications=_as_text(cols["certifications"][index]),
            experience_level=_as_text(cols["experience_level"][index]),
        )
        for index in range(table.num_rows)
    ]


def build_job_description_text(posting: PostingCandidate) -> str:
    """The narrative text alone frequently states no explicit hard
    Requirement at all (many postings in this corpus are pure role
    summaries), so every non-empty qualification field is appended as its
    own labeled section - that is where the corpus actually puts the hard
    Requirements extraction is looking for.
    """
    sections = [f"{posting.title}\n\n{posting.job_description}"]
    labeled_fields = (
        ("Minimum Qualifications", posting.minimum_qualifications),
        ("Preferred Qualifications", posting.preferred_qualifications),
        ("Required Skills", posting.skills_required),
        ("Education", posting.education_level),
        ("Certifications", posting.certifications),
        ("Experience Level", posting.experience_level),
    )
    for label, value in labeled_fields:
        if value.strip():
            sections.append(f"{label}:\n{value}")
    return "\n\n".join(sections)


def write_mapping_report(matches: dict[str, CategoryMatch | None], path: Path) -> None:
    lines = [
        "# Evaluation Role category-to-posting mapping",
        "",
        f"Source postings corpus: `{POSTINGS_DATASET}`.",
        "Matched by script (screening.eval_roles.match_all_categories) against the keyword table in CATEGORY_SEARCH_TERMS - see that module for the matching rule.",
        "Review this table for a category matched to the wrong kind of role: a bad match here silently corrupts Proxy Relevance for that Role (ADR-0008).",
        "",
        "| Category | Matched term | Posting title | Company |",
        "|---|---|---|---|",
    ]
    for category, match in matches.items():
        if match is None:
            lines.append(f"| {category} | (none) | - | - |")
            continue
        title = match.posting.title.replace("|", "\\|")
        company = match.posting.company_name.replace("|", "\\|")
        lines.append(f"| {category} | {match.matched_term} | {title} | {company} |")
    path.write_text("\n".join(lines) + "\n")


def cmd_propose(args: argparse.Namespace) -> int:
    try:
        model_client = build_deepseek_client(usage="to run live extraction")
    except MissingApiKey as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    categories = load_resume_atlas_categories(args.resume_atlas_parquet)
    missing = set(categories) - set(CATEGORY_SEARCH_TERMS)
    if missing:
        print(f"error: no search terms for categories: {sorted(missing)}", file=sys.stderr)
        return 1

    postings = load_postings(args.postings_parquet)
    matches = match_all_categories(sorted(CATEGORY_SEARCH_TERMS), postings)
    unmatched = [c for c, m in matches.items() if m is None]
    if unmatched:
        print(f"error: no posting matched for categories: {unmatched}", file=sys.stderr)
        return 1

    args.mapping_out.parent.mkdir(parents=True, exist_ok=True)
    write_mapping_report(matches, args.mapping_out)
    print(f"Wrote mapping report to {args.mapping_out}")

    proposals = {}
    for index, (category, match) in enumerate(sorted(matches.items()), start=1):
        assert match is not None
        job_description = JobDescription(text=build_job_description_text(match.posting))
        try:
            proposed = extract_requirements(job_description, model_client)
        except ModelClientError as exc:
            print(f"error: extraction failed for {category!r}: {exc}", file=sys.stderr)
            return 1

        proposals[category] = {
            "job_description": job_description.text,
            "requirements": [{"id": r.id, "text": r.text} for r in proposed],
            "posting": {
                "source_dataset": POSTINGS_DATASET,
                "posting_id": match.posting.posting_id,
                "title": match.posting.title,
                "company_name": match.posting.company_name,
                "matched_term": match.matched_term,
            },
        }
        print(f"[{index}/{len(matches)}] {category}: {len(proposed)} Requirement(s) proposed")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(proposals, indent=2))
    print(f"Wrote {len(proposals)} proposals to {args.out}")
    print(
        f"Parse-failure rate this run: {model_client.metrics.parse_failure_rate:.1%} "
        f"({model_client.metrics.parse_failures}/{model_client.metrics.total_attempts} attempts)"
    )
    return 0


def cmd_finalize(args: argparse.Namespace) -> int:
    if args.reviewed_sample_size < MIN_REVIEWED_SAMPLE_SIZE:
        print(
            f"error: --reviewed-sample-size must be at least {MIN_REVIEWED_SAMPLE_SIZE} "
            f"(ADR-0008), got {args.reviewed_sample_size}",
            file=sys.stderr,
        )
        return 1
    if not args.review.exists():
        print(f"error: --review file not found at {args.review}", file=sys.stderr)
        return 1

    proposals = json.loads(args.proposals.read_text())
    review = json.loads(args.review.read_text())

    generated_at = datetime.now(timezone.utc)
    evaluation_roles = []
    for index, category in enumerate(sorted(proposals), start=1):
        proposal = proposals[category]
        review_entry = review.get(category)
        if review_entry is not None:
            requirements = tuple(
                Requirement(id=r["id"], text=r["text"]) for r in review_entry["requirements"]
            )
            reviewed = bool(review_entry.get("reviewed", True))
        else:
            requirements = tuple(
                Requirement(id=r["id"], text=r["text"]) for r in proposal["requirements"]
            )
            reviewed = False

        requirement_set = approve_requirement_set(requirements)
        evaluation_role_id = f"eval-role-{index:02d}"
        posting = PostingProvenance(**proposal["posting"])
        # The real posting's own title, not the resume-atlas category label:
        # Proxy Relevance is defined by category match (screening.proxy_
        # relevance), so a Role titled after the category would hand the
        # Ranking prompt the exact string its own evaluation ground truth is
        # keyed on - the same circularity ADR-0008 forbids for Job
        # Description generation, one step downstream in the Role itself.
        role = Role(id=evaluation_role_id, title=posting.title, requirement_set=requirement_set)
        evaluation_roles.append(
            EvaluationRole(
                evaluation_role_id=evaluation_role_id,
                category=category,
                role=role,
                posting=posting,
                reviewed_by_human=reviewed,
                generated_at=generated_at,
            )
        )

    # Checked before any file is written: a gate that only fires after the
    # 43 read-only files already exist on disk would force a manual cleanup
    # of every one of them before a corrected re-run could even attempt it
    # again, on top of masking the real failure behind FileExistsError.
    reviewed_count = sum(1 for role in evaluation_roles if role.reviewed_by_human)
    if reviewed_count < args.reviewed_sample_size:
        print(
            f"error: only {reviewed_count} Evaluation Role(s) marked reviewed, "
            f"need at least {args.reviewed_sample_size}",
            file=sys.stderr,
        )
        return 1

    for evaluation_role in evaluation_roles:
        write_evaluation_role(evaluation_role, args.out_dir)

    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_out.write_text(
        json.dumps(
            {
                "total_evaluation_roles": len(proposals),
                "reviewed_sample_size": reviewed_count,
                "source_resume_corpus": RESUME_ATLAS_DATASET,
                "source_postings_corpus": POSTINGS_DATASET,
                "generated_at": generated_at.isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {len(proposals)} Evaluation Roles to {args.out_dir} ({reviewed_count} reviewed)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    propose = subparsers.add_parser("propose")
    propose.add_argument("--resume-atlas-parquet", required=True, type=Path)
    propose.add_argument("--postings-parquet", required=True, type=Path)
    propose.add_argument("--out", required=True, type=Path)
    propose.add_argument("--mapping-out", required=True, type=Path)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--proposals", required=True, type=Path)
    finalize.add_argument("--review", required=True, type=Path)
    finalize.add_argument("--reviewed-sample-size", required=True, type=int)
    finalize.add_argument("--out-dir", required=True, type=Path)
    finalize.add_argument("--metadata-out", required=True, type=Path)

    args = parser.parse_args(argv)
    if args.command == "propose":
        return cmd_propose(args)
    return cmd_finalize(args)


if __name__ == "__main__":
    sys.exit(main())
