"""Category-to-posting matching, and the checked-in file format, for the
Evaluation Roles ticket 06 builds (ADR-0008).

Matching is a fixed keyword table rather than a similarity score or a model
call: a model call here would recreate, one step upstream, the exact
circularity ADR-0008 forbids in Job Description generation, and a fixed
table is what makes the mapping reviewable rather than an opaque number.
Callers should treat CATEGORY_SEARCH_TERMS itself as the artifact under
review - it is what a match is judged against.

Every category in the resume-atlas corpus (ahmedheakl/resume-atlas) has an
entry here, checked by test_eval_roles.py against a snapshot of that
corpus's category set, so a corpus update that adds or renames a category
fails loudly here rather than silently producing 42 Evaluation Roles.

An EvaluationRole pairs that matched posting with the Role extraction and
human review produced. It is the unit checked into version control - see
CONTEXT.md's Evaluation Role entry - and read back by the evaluation
harness (ticket 07) rather than by the core, which only ever sees a Role.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from screening.domain import Requirement, RequirementSet, Role


@dataclass(frozen=True)
class PostingCandidate:
    """One row of a public job-postings corpus, reduced to the fields
    matching and extraction need. `posting_id` is whatever the source
    corpus uses to identify the row, so a match stays traceable back to it.

    `job_description` is the narrative text matching runs against. The
    qualification fields are separate because a posting's narrative
    frequently carries no explicit hard Requirements at all - the
    qualifications are what extraction actually needs to find any - so
    callers building an extraction input should append whichever of these
    are non-empty rather than passing `job_description` alone.
    """

    posting_id: str
    title: str
    normalized_title: str
    company_name: str
    job_description: str
    minimum_qualifications: str = ""
    preferred_qualifications: str = ""
    skills_required: str = ""
    education_level: str = ""
    certifications: str = ""
    experience_level: str = ""


@dataclass(frozen=True)
class CategoryMatch:
    category: str
    posting: PostingCandidate
    matched_term: str


# Ordered most to least specific per category: the first term found in a
# posting's title wins, so a specific title ("Staff Accountant") is
# preferred over a merely-plausible one ("Accounting"). Terms are matched
# case-insensitively as substrings.
CATEGORY_SEARCH_TERMS: dict[str, tuple[str, ...]] = {
    "Accountant": ("Staff Accountant", "Senior Accountant", "Accountant", "Accounting Associate"),
    # Plain "Advocate" is excluded: in this postings corpus every hit is a
    # "Customer Advocate" (a customer-service role), never a legal one, so
    # it would silently mismatch the resume-atlas legal-resume category.
    "Advocate": ("Attorney", "Lawyer", "Legal Counsel"),
    "Agriculture": ("Agronomist", "Agricultural Engineer", "Farm Manager", "Agriculture", "Agricultural"),
    # "Apparel" alone is excluded for the same reason: it only ever hits
    # retail sales-clerk postings here, not the apparel-industry design and
    # merchandising resumes the category actually holds.
    "Apparel": ("Apparel Designer", "Fashion Designer", "Garment Technologist", "Apparel Merchandiser"),
    "Architecture": ("Architectural Designer", "Architect", "Architecture"),
    "Arts": ("Graphic Artist", "Fine Artist", "Illustrator", "Sculptor"),
    "Automobile": ("Automotive Technician", "Auto Mechanic", "Automotive Engineer", "Automotive", "Automobile"),
    "Aviation": ("Flight Attendant", "Airline Pilot", "Aircraft Mechanic", "Aviation", "Airline"),
    # Literal "BPO" is excluded: in this corpus it almost always expands to
    # "Business Process Owner" (an internal ops title) or appears only in
    # Spanish-language postings, neither of which is the call-center/BPO
    # industry work the resume-atlas category holds.
    "BPO": (
        "Call Center Customer Service Representative",
        "Call Center Representative",
        "Call Centre Agent",
        "Call Center Agent",
        "Customer Service Representative",
    ),
    "Banking": ("Bank Teller", "Loan Officer", "Banking Associate", "Bank Manager", "Banking"),
    "Blockchain": ("Blockchain Developer", "Blockchain Engineer", "Smart Contract Developer", "Blockchain", "Web3"),
    "Building and Construction": (
        "Construction Superintendent",
        "Construction Manager",
        "Construction Foreman",
        "Building Inspector",
        "Construction",
    ),
    "Business Analyst": ("Business Systems Analyst", "Business Analyst"),
    "Civil Engineer": ("Civil Engineer", "Structural Engineer"),
    "Consultant": ("Management Consultant", "Strategy Consultant", "Business Consultant", "Consultant"),
    "Data Science": ("Data Scientist", "Machine Learning Engineer", "ML Engineer", "Data Science"),
    "Database": ("Database Administrator", "Database Engineer", "Database Developer", "DBA"),
    "Designing": ("Graphic Designer", "Product Designer", "Industrial Designer", "Visual Designer", "Interior Designer"),
    "DevOps": ("DevOps Engineer", "Site Reliability Engineer", "SRE Engineer", "DevOps"),
    "Digital Media": ("Digital Media Specialist", "Social Media Manager", "Content Creator", "Digital Media"),
    "DotNet Developer": (".NET Developer", "Dot Net Developer", "ASP.NET Developer", "C# Developer"),
    "ETL Developer": ("ETL Developer", "ETL Engineer", "Data Engineer"),
    "Education": ("Teacher", "Instructor", "Professor", "Educator", "Tutor"),
    "Electrical Engineering": ("Electrical Engineer", "Electrical Engineering"),
    "Finance": ("Financial Analyst", "Finance Manager", "Financial Advisor", "Finance"),
    "Food and Beverages": ("Food and Beverage Manager", "Restaurant Manager", "Culinary", "Chef", "Barista"),
    "Health and Fitness": ("Personal Trainer", "Fitness Instructor", "Fitness Coach", "Wellness Coach"),
    "Human Resources": (
        "Human Resources Manager",
        "HR Business Partner",
        "HR Generalist",
        "Talent Acquisition",
        "Human Resources",
        "Recruiter",
    ),
    "Information Technology": (
        "IT Support Specialist",
        "IT Administrator",
        "Systems Administrator",
        "Information Technology",
        "IT Technician",
    ),
    "Java Developer": ("Java Developer", "Java Engineer"),
    "Management": ("General Manager", "Program Manager", "Management Trainee", "Managing Director"),
    "Mechanical Engineer": ("Mechanical Engineer", "Mechanical Engineering"),
    "Network Security Engineer": (
        "Network Security Engineer",
        "Cybersecurity Engineer",
        "Information Security Engineer",
        "Network Engineer",
    ),
    "Operations Manager": ("Operations Manager", "Operations Director"),
    "PMO": ("PMO Analyst", "PMO Manager", "PMO Lead", "Project Management Office", "PMO"),
    "Public Relations": ("Public Relations Manager", "Public Relations Specialist", "PR Specialist", "Public Relations"),
    "Python Developer": ("Python Developer", "Python Engineer"),
    "React Developer": ("React Developer", "React Native Developer", "React.js Developer"),
    "SAP Developer": ("SAP Developer", "SAP Consultant", "SAP Analyst"),
    "SQL Developer": ("SQL Developer", "T-SQL Developer"),
    "Sales": ("Sales Representative", "Account Executive", "Sales Manager", "Sales Associate", "Sales Executive"),
    "Testing": ("QA Engineer", "Test Engineer", "Quality Assurance Engineer", "Software Tester", "QA Analyst"),
    "Web Designing": ("Web Designer", "Website Designer", "UI/UX Designer", "Web Design"),
}


@lru_cache(maxsize=None)
def _haystack(posting: PostingCandidate) -> str:
    """Cached per posting: match_all_categories scores every posting once
    per category, and a real postings corpus is large enough (six figures
    of rows) that re-lower-casing the same title on every one of those
    passes is wasted work worth avoiding.
    """
    return f"{posting.title} {posting.normalized_title}".lower()


def score_posting(category: str, posting: PostingCandidate) -> tuple[int, str] | None:
    """The rank of the most specific search term found in the posting's
    title, and that term - or None if no term for this category appears.
    Higher rank is more specific, so callers can pick the best match with a
    plain max().
    """
    terms = CATEGORY_SEARCH_TERMS[category]
    haystack = _haystack(posting)
    for index, term in enumerate(terms):
        if term.lower() in haystack:
            return len(terms) - index, term
    return None


def _has_qualification_data(posting: PostingCandidate) -> bool:
    """Whether this posting states any qualification a Candidate could
    actually be checked against. Many postings in a real corpus carry only
    a narrative summary - "on-site in Lake Buena Vista, FL" is not a hard
    Requirement - so a posting with real qualification fields is preferred
    over one without, even at equal title-match specificity: extraction
    can only find Requirements that are actually stated.
    """
    return bool(
        posting.minimum_qualifications.strip()
        or posting.preferred_qualifications.strip()
        or posting.skills_required.strip()
    )


def match_category_to_posting(
    category: str,
    postings: Sequence[PostingCandidate],
    *,
    excluded_posting_ids: frozenset[str] = frozenset(),
) -> CategoryMatch | None:
    """The single best-scoring posting for `category`, skipping postings
    already claimed by another category and postings with no Job
    Description text to extract from. Ranked first by title-match
    specificity, then by whether the posting actually has qualification
    data to extract Requirements from. Ties are broken by corpus order, so
    the result is deterministic given the same postings and exclusions.
    """
    best: CategoryMatch | None = None
    best_key: tuple[int, bool] = (-1, False)
    for posting in postings:
        if posting.posting_id in excluded_posting_ids:
            continue
        if not posting.job_description.strip():
            continue
        scored = score_posting(category, posting)
        if scored is None:
            continue
        score, term = scored
        key = (score, _has_qualification_data(posting))
        if key > best_key:
            best_key = key
            best = CategoryMatch(category=category, posting=posting, matched_term=term)
    return best


def match_all_categories(
    categories: Sequence[str], postings: Sequence[PostingCandidate]
) -> dict[str, CategoryMatch | None]:
    """Matches every category in order, each matched posting excluded from
    consideration for every category after it, so no two Evaluation Roles
    ever cite the same real-world listing.
    """
    results: dict[str, CategoryMatch | None] = {}
    used: set[str] = set()
    for category in categories:
        match = match_category_to_posting(category, postings, excluded_posting_ids=frozenset(used))
        results[category] = match
        if match is not None:
            used.add(match.posting.posting_id)
    return results


@dataclass(frozen=True)
class PostingProvenance:
    """Where an Evaluation Role's Job Description came from, kept on the
    checked-in record so the category-to-posting mapping stays traceable
    back to the corpus row it was matched from - the reviewable-artifact
    requirement, attached to the thing it is about rather than living only
    in a separate report that can drift out of sync.
    """

    source_dataset: str
    posting_id: str
    title: str
    company_name: str
    matched_term: str


@dataclass(frozen=True)
class EvaluationRole:
    """A Role used by the evaluation harness rather than supplied by a
    Recruiter (CONTEXT.md). `reviewed_by_human` marks membership in the
    ADR-0008 sample: at least 10 of the 43 must carry True before check-in,
    and the total is recorded alongside the checked-in set for reporting.
    """

    evaluation_role_id: str
    category: str
    role: Role
    posting: PostingProvenance
    reviewed_by_human: bool
    generated_at: datetime


def write_evaluation_role(evaluation_role: EvaluationRole, root: Path) -> Path:
    """Writes one Evaluation Role as its own checked-in file, created
    exclusively and made read-only, mirroring FileRunStore and
    FileExtractionRecordStore: regenerating the set is a deliberate
    versioned event (ADR-0008), never a silent overwrite of one row.
    """
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{evaluation_role.evaluation_role_id}.json"
    payload = {
        "evaluation_role_id": evaluation_role.evaluation_role_id,
        "category": evaluation_role.category,
        "role": asdict(evaluation_role.role),
        "posting": asdict(evaluation_role.posting),
        "reviewed_by_human": evaluation_role.reviewed_by_human,
        "generated_at": evaluation_role.generated_at.isoformat(),
    }

    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(payload, indent=2) + "\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    path.chmod(0o444)
    return path


def read_evaluation_role(path: Path) -> EvaluationRole:
    data = json.loads(path.read_text())
    role_data = data["role"]
    requirement_set_data = role_data["requirement_set"]
    role = Role(
        id=role_data["id"],
        title=role_data["title"],
        requirement_set=RequirementSet(
            requirements=tuple(
                Requirement(id=r["id"], text=r["text"])
                for r in requirement_set_data["requirements"]
            ),
            approved=requirement_set_data["approved"],
        ),
    )
    posting = PostingProvenance(**data["posting"])
    return EvaluationRole(
        evaluation_role_id=data["evaluation_role_id"],
        category=data["category"],
        role=role,
        posting=posting,
        reviewed_by_human=data["reviewed_by_human"],
        generated_at=datetime.fromisoformat(data["generated_at"]),
    )


def read_evaluation_roles(root: Path) -> list[EvaluationRole]:
    """Every checked-in Evaluation Role under `root`, in filename order -
    the read side the evaluation harness (ticket 07) consumes."""
    return [read_evaluation_role(path) for path in sorted(root.glob("*.json"))]
