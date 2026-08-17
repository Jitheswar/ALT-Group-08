"""Builds the checked-in Demo Presets under `data/demo_presets/` - one per
Evaluation Role, so the web demo offers the whole spread of fields the
corpus covers rather than a handful.

    uv run python scripts/build_demo_presets.py

Three things go into each Preset, and only the first is written here by
hand:

1. The **opener and label** for its field, from ROLE_PRESENTATION below.
   That table is the artifact under review in this script: the Job
   Description prose a Recruiter reads is hand-authored demo material, and
   naming the real posting it stands in for keeps it honest about what it
   is - a demo stand-in, never an Evaluation Role's real posting (ADR-0008).
2. The **Requirements**, taken verbatim from the Evaluation Role's reviewed
   Requirement Set - not invented here, so a Preset tests Candidates against
   the same conditions the evaluation harness does.
3. The **Resumes**, chosen from the Gold Set: two sharing the Role's corpus
   category and two from other fields, referenced by candidate id so the
   Resume text itself lives in one place.

Which of the Role's Requirements become Job Description bullets is then
*selected* rather than authored: this script screens the chosen Resumes
against the full Requirement Set and keeps the Requirements both in-field
Candidates meet, so every Preset lands a Shortlist with two Qualified
Candidates for the comparative pass to order and a Disqualified one with a
cited missed Requirement. A Preset whose whole batch fails Screening
demonstrates half the system. Requirements left out are still shown to the
Recruiter, as the Job Description's closing "helpful but not essential"
line - so nothing from the reviewed Set is discarded, and step 2 has
something real to add.

Selection runs against ScriptedModelClient, the demo's own default model
client, so what is verified here is what a demo actually shows. Presets are
regenerable and are written plainly rather than through write_once: unlike
the Gold Set's hand labels, nothing in the output is unrecoverable - the
hand-written half lives in this file, under version control.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from screening.core import run_screening
from screening.domain import (
    Candidate,
    Disqualified,
    Qualified,
    Requirement,
    Resume,
    Role,
    approve_requirement_set,
)
from screening.eval_roles import EvaluationRole, read_evaluation_roles
from screening.gold_set import GoldSetLabel, read_gold_set
from screening.scripted_client import ScriptedModelClient

DEFAULT_EVALUATION_ROLES_DIR = Path("data/evaluation_roles/roles")
DEFAULT_GOLD_SET_PATH = Path("data/gold_set/gold_set.json")
DEFAULT_OUT_DIR = Path("data/demo_presets")

# How many Resumes of each kind a Preset stages. Two in-field Resumes is the
# smallest batch the comparative pass has anything to say about; two
# out-of-field ones make the Disqualified half of the Shortlist more than a
# single example.
IN_FIELD_RESUMES = 2
OUT_OF_FIELD_RESUMES = 2

# Below this, the review page in step 2 has too little on it to be worth
# reading, and the Requirement Set gates on almost nothing.
MIN_REQUIREMENTS = 3

# Requirements added by ROLE_PRESENTATION rather than drawn from a reviewed
# Requirement Set, kept distinguishable so the Job Description never claims
# the real posting asked for one.
DEMO_REQUIREMENT_ID_PREFIX = "req-demo-"


@dataclass(frozen=True)
class RolePresentation:
    """The hand-authored half of a Preset: the short name its card carries
    and the Job Description's opening paragraph. `label` is also the Role
    title a Screening Run is conducted against, so it reads as a job title
    rather than a corpus category.

    `extra_requirements` is an escape hatch for a field whose reviewed
    Requirement Set is too specific to the one posting it came from for any
    two Resumes in that field to meet three of them - "Native or fluent
    Russian" gates on something no Resume in this corpus states. These are
    hand-authored, ordinary conditions of the field, added to the pool the
    selection below chooses from; they are not written to make a particular
    Candidate qualify, and a field that still cannot demonstrate both stages
    is skipped rather than propped up.
    """

    family: str
    label: str
    opener: str
    extra_requirements: tuple[str, ...] = ()


# Families group the 43 fields into something browsable on one page. A field
# sits in exactly one.
SOFTWARE = "Software and Data"
ENGINEERING = "Engineering and Industry"
BUSINESS = "Business and Finance"
PEOPLE = "People, Legal and Education"
CREATIVE = "Creative and Media"
WELLBEING = "Health, Food and Agriculture"

# Keyed by the Evaluation Role's corpus category. Every category the
# evaluation set covers needs an entry; build fails loudly on a missing one
# rather than quietly shipping fewer Presets.
ROLE_PRESENTATION: dict[str, RolePresentation] = {
    "Accountant": RolePresentation(
        BUSINESS,
        "Payroll and Staff Accountant",
        "Landis Communities is hiring a Payroll and Staff Accountant to own biweekly "
        "payroll and the month-end close for a 900-person organisation. You will report "
        "to the Controller and work alongside two staff accountants.",
    ),
    "Advocate": RolePresentation(
        PEOPLE,
        "Labor and Employment Attorney",
        "O'Hagan Meyer is looking for an associate attorney to join its labor and "
        "employment practice, handling discrimination and wage-and-hour matters from "
        "intake through trial.",
    ),
    "Agriculture": RolePresentation(
        WELLBEING,
        "Agronomist Coordinator",
        "Syngenta needs an Agronomist Coordinator for its seed production programme, "
        "planning trials across grower sites and reporting on what each field yields.",
    ),
    "Apparel": RolePresentation(
        CREATIVE,
        "Apparel Designer",
        "We are hiring an Apparel Designer for the boyswear licensed-brands line, "
        "taking each season from concept sketches through fit sessions to handover.",
        extra_requirements=(
            "Apparel product development",
            "Working with factories and suppliers on samples",
            "Quality and specification review",
        ),
    ),
    "Architecture": RolePresentation(
        ENGINEERING,
        "Enterprise Architect",
        "AND Digital is looking for an Enterprise Architect to set the target "
        "architecture for a multi-year platform programme and keep delivery teams "
        "aligned to it.",
    ),
    "Arts": RolePresentation(
        CREATIVE,
        "Production Graphic Artist",
        "Forney Industries needs a Production Graphic Artist to prepare packaging and "
        "point-of-sale artwork for print, working from brand guidelines and press specs.",
    ),
    "Automobile": RolePresentation(
        ENGINEERING,
        "Automotive Technician",
        "Our service centre is hiring an Automotive Technician to diagnose and repair "
        "customer vehicles across engine, brake and electrical work.",
    ),
    "Aviation": RolePresentation(
        ENGINEERING,
        "Flight Attendant",
        "We are hiring Flight Attendants for domestic and international routes, "
        "responsible for cabin safety first and for the passenger experience second.",
    ),
    "BPO": RolePresentation(
        BUSINESS,
        "Call Center Customer Service Representative",
        "Veolia is hiring Call Center Customer Service Representatives to handle "
        "inbound billing and service enquiries for utility customers.",
    ),
    "Banking": RolePresentation(
        BUSINESS,
        "Bank Teller",
        "Our branch network is hiring Bank Tellers to handle customer transactions "
        "accurately and to spot where a customer would be better served by another "
        "product.",
    ),
    "Blockchain": RolePresentation(
        SOFTWARE,
        "Blockchain Engineer",
        "Travoom is hiring a senior Blockchain Engineer for its payments and exchange "
        "stack, working on settlement flows and the smart contracts behind them.",
    ),
    "Building and Construction": RolePresentation(
        ENGINEERING,
        "Construction Superintendent",
        "Wood needs a Construction Superintendent to run day-to-day site work on a "
        "structural, mechanical and piping package, owning schedule and site safety.",
    ),
    "Business Analyst": RolePresentation(
        BUSINESS,
        "Business Systems Analyst",
        "MedImpact is hiring a Business Systems Analyst for its Medicaid "
        "fee-for-service line, turning claims-processing requirements into something "
        "engineering can build against.",
    ),
    "Civil Engineer": RolePresentation(
        ENGINEERING,
        "Civil Engineer",
        "GEK TERNA is hiring a Civil Engineer for its procurement function, reviewing "
        "structural designs and quantities on major infrastructure works.",
    ),
    "Consultant": RolePresentation(
        BUSINESS,
        "Change Management Consultant",
        "Michael Baker International is hiring a Change Management Consultant to lead "
        "the people side of a large systems rollout: stakeholder analysis, training and "
        "adoption.",
        extra_requirements=(
            "Business process improvement",
            "Stakeholder requirements gathering",
            "Project management",
        ),
    ),
    "Data Science": RolePresentation(
        SOFTWARE,
        "Lead Marketing Data Scientist",
        "One Park Financial is hiring a Lead Marketing Data Scientist to model "
        "acquisition performance end to end, from channel attribution through to "
        "lifetime value.",
    ),
    "Database": RolePresentation(
        SOFTWARE,
        "Database Administrator",
        "Exadel is hiring an Associate Database Administrator to keep production "
        "databases healthy: backups, tuning, and the migrations that touch them.",
    ),
    "Designing": RolePresentation(
        CREATIVE,
        "Production Graphic Designer",
        "We are hiring a Production Graphic Designer to turn campaign concepts into "
        "finished assets across web, social and print, at the volume a busy calendar "
        "needs.",
    ),
    "DevOps": RolePresentation(
        SOFTWARE,
        "DevOps Engineer",
        "We are hiring a DevOps Engineer to take ownership of the delivery pipeline and "
        "the container platform every one of our product teams deploys onto.",
    ),
    "Digital Media": RolePresentation(
        CREATIVE,
        "Social Media Manager",
        "Jamf is hiring a Social Media Manager to own the content calendar across "
        "channels and to report on what the audience actually engages with.",
    ),
    "DotNet Developer": RolePresentation(
        SOFTWARE,
        ".NET Developer",
        "Barclays is hiring a senior C#/.NET developer for its custody liquid finance "
        "platform, working on the services that move and reconcile client positions.",
    ),
    "ETL Developer": RolePresentation(
        SOFTWARE,
        "ETL Developer",
        "IQVIA is hiring a Big Data ETL Developer to build the pipelines that land "
        "clinical and claims data into the warehouse, and to keep them trustworthy.",
    ),
    "Education": RolePresentation(
        PEOPLE,
        "Special Education Teacher",
        "Empowerment Academy is hiring a High School Special Education Teacher to run "
        "a resource classroom, write and track IEPs, and co-teach with subject staff.",
    ),
    "Electrical Engineering": RolePresentation(
        ENGINEERING,
        "Electrical Engineer",
        "Leidos is hiring an Electrical Engineer for site development and installation "
        "work: power distribution design, field surveys and commissioning support.",
    ),
    "Finance": RolePresentation(
        BUSINESS,
        "Financial Analyst",
        "Millennium Health is hiring a Financial Analyst to own the monthly forecast, "
        "explain variance to budget, and build the models behind planning decisions.",
    ),
    "Food and Beverages": RolePresentation(
        WELLBEING,
        "Food and Beverage Manager",
        "Sonesta Hotels is hiring a Food and Beverage Manager to run restaurant and "
        "banquet service for a full-service property, owning cost, staffing and guest "
        "experience.",
    ),
    "Health and Fitness": RolePresentation(
        WELLBEING,
        "Personal Trainer",
        "Bethany Athletic Club is hiring Personal Trainers to build and coach "
        "individual programmes, and to keep members progressing safely.",
    ),
    "Human Resources": RolePresentation(
        PEOPLE,
        "Human Resources Manager",
        "Atrium Hospitality is hiring a Human Resources Manager for a 300-room "
        "property, leading recruiting, onboarding and employee relations for a team of "
        "180.",
    ),
    "Information Technology": RolePresentation(
        SOFTWARE,
        "IT Support Specialist",
        "Lozier is hiring a Tier 1 IT Support Specialist to be the first line for "
        "hardware, account and desktop issues across the plant and the office.",
    ),
    "Java Developer": RolePresentation(
        SOFTWARE,
        "Java Developer",
        "We are hiring a Java Developer to build and maintain the backend services "
        "behind our order platform, working closely with a bilingual product team.",
    ),
    "Management": RolePresentation(
        BUSINESS,
        "General Manager",
        "We are hiring a General Manager to run a single high-volume store end to end: "
        "the team, the shift, the numbers and the standards.",
    ),
    "Mechanical Engineer": RolePresentation(
        ENGINEERING,
        "Mechanical Engineer",
        "GEK TERNA is hiring a Mechanical Engineer for plant maintenance, owning "
        "preventive schedules and the diagnosis of rotating-equipment failures.",
        extra_requirements=(
            "Mechanical engineering fundamentals",
            "Producing engineering drawings",
            "Mechanical design and modelling",
        ),
    ),
    "Network Security Engineer": RolePresentation(
        SOFTWARE,
        "Network Security Engineer",
        "We are hiring a senior Network Security Engineer to own firewall and segmentation "
        "design, and to lead the response when something reaches the perimeter.",
    ),
    "Operations Manager": RolePresentation(
        BUSINESS,
        "Operations Manager",
        "USA Vein Clinics is hiring an Operations Manager to run a group of clinics: "
        "scheduling, staffing, throughput and the process fixes behind them.",
    ),
    "PMO": RolePresentation(
        BUSINESS,
        "PMO Analyst",
        "Accenture Federal Services is hiring a PMO Analyst to keep a large programme "
        "honest: plans, dependencies, risk logs and the reporting leadership reads.",
    ),
    "Public Relations": RolePresentation(
        CREATIVE,
        "Public Relations Manager",
        "The City of San Antonio is hiring a Public Relations Manager to handle media "
        "relations and public communications, including when the story is a difficult "
        "one.",
    ),
    "Python Developer": RolePresentation(
        SOFTWARE,
        "Python Developer",
        "Betmaster is looking for a Python Developer to join the risk and compliance "
        "team, working on the services that decide what gets flagged for review.",
        extra_requirements=(
            "Python 3",
            "Backend web applications",
            "Relational database design",
        ),
    ),
    "React Developer": RolePresentation(
        SOFTWARE,
        "React Developer",
        "Flux IT is hiring a junior React Developer to build product screens against a "
        "design system, alongside a senior who will review your work.",
    ),
    "SAP Developer": RolePresentation(
        SOFTWARE,
        "SAP Developer",
        "L.L.Bean is hiring a senior SAP Developer to build and support the ABAP "
        "extensions behind its retail and supply-chain processes.",
    ),
    "SQL Developer": RolePresentation(
        SOFTWARE,
        "PL/SQL Developer",
        "Version 1 is hiring an Oracle PL/SQL Developer to write and tune the stored "
        "procedures a high-volume transactional system runs on.",
    ),
    "Sales": RolePresentation(
        BUSINESS,
        "Sales Representative",
        "We are hiring a Sales Representative for the state and local education "
        "territory, owning the full cycle from first contact to signed order.",
    ),
    "Testing": RolePresentation(
        SOFTWARE,
        "Senior QA Engineer",
        "Rapid Finance is hiring a Senior QA Engineer to own the test strategy for a "
        "lending platform, automating the coverage that keeps releases boring.",
    ),
    "Web Designing": RolePresentation(
        CREATIVE,
        "Web Designer",
        "KPMG Netherlands is hiring a Web Designer to design and build responsive "
        "pages and campaign sites within an established brand system.",
    ),
}

FAMILY_ORDER = (SOFTWARE, ENGINEERING, BUSINESS, PEOPLE, CREATIVE, WELLBEING)


class CannotBuildPreset(Exception):
    """This Evaluation Role cannot make a Preset that demonstrates both
    stages - reported per Role and skipped, rather than shipped as a Preset
    whose Shortlist is all Disqualified."""


@dataclass(frozen=True)
class _StagedResume:
    candidate_id: str
    category: str
    in_field: bool


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _met_requirement_ids(
    role_title: str,
    requirements: Sequence[Requirement],
    candidate_ids: Sequence[str],
    resume_text_by_id: dict[str, str],
) -> dict[str, frozenset[str]]:
    """The ids of the Requirements each Candidate met, read off a real
    Screening Run against the demo's own model client rather than
    reimplementing its judgement here. One Run over the whole pool, so
    selection sees exactly what a demo would.
    """
    candidates = [
        Candidate(id=candidate_id, resume=Resume(text=resume_text_by_id[candidate_id]))
        for candidate_id in candidate_ids
    ]
    role = Role(
        id=str(uuid4()),
        title=role_title,
        requirement_set=approve_requirement_set(tuple(requirements)),
    )
    shortlist = run_screening(role, candidates, ScriptedModelClient())

    met: dict[str, frozenset[str]] = {}
    for entry in shortlist.entries:
        outcome = entry.outcome
        if not isinstance(outcome, (Qualified, Disqualified)):
            raise CannotBuildPreset(f"{entry.candidate_id} produced no verdict")
        met[entry.candidate_id] = frozenset(v.requirement_id for v in outcome.verdicts if v.met)
    return met


def _choose_in_field_pair(
    same_field_ids: Sequence[str], met: dict[str, frozenset[str]]
) -> tuple[list[str], frozenset[str]]:
    """The pair of in-field Resumes that share the most Requirements, and
    that shared set.

    Which Requirements a Preset gates on falls out of this choice: the
    shared set is what both Candidates meet, so it is what makes them both
    Qualified and gives the comparative pass a pair to order. Searching the
    pair rather than taking the first two is what makes most fields usable -
    one weak Resume in a field otherwise drags the shared set to nothing.
    """
    best: tuple[list[str], frozenset[str]] | None = None
    for first_index, first in enumerate(same_field_ids):
        for second in same_field_ids[first_index + 1 :]:
            shared = met[first] & met[second]
            if best is None or len(shared) > len(best[1]):
                best = ([first, second], shared)
    if best is None:
        raise CannotBuildPreset(
            f"fewer than {IN_FIELD_RESUMES} in-field Gold Set Resumes for this field"
        )
    if len(best[1]) < MIN_REQUIREMENTS:
        raise CannotBuildPreset(
            f"no two in-field Resumes share {MIN_REQUIREMENTS} met Requirements "
            f"(best pair shares {len(best[1])})"
        )
    return best


def _choose_out_of_field(
    other_field_ids: Sequence[str],
    met: dict[str, frozenset[str]],
    kept_ids: frozenset[str],
    category_by_id: dict[str, str],
) -> list[str]:
    """Resumes from other fields that each miss at least one of the kept
    Requirements, so the Shortlist shows Screening excluding somebody and
    citing what they missed. A Resume that happens to meet all of them is
    passed over rather than staged as a Disqualified example it would not
    be."""
    usable = [
        candidate_id
        for candidate_id in other_field_ids
        if not kept_ids <= met[candidate_id]
    ]
    if len(usable) < OUT_OF_FIELD_RESUMES:
        raise CannotBuildPreset(
            f"only {len(usable)} out-of-field Resumes miss any kept Requirement, so "
            "the Shortlist would show nothing being screened out"
        )

    # Two Resumes from two different fields, where the pool allows it: a batch
    # whose out-of-field half is two Resumes from the same field shows the
    # Recruiter one near-miss twice over.
    chosen: list[str] = []
    seen_categories: set[str] = set()
    for candidate_id in usable:
        if category_by_id[candidate_id] in seen_categories:
            continue
        chosen.append(candidate_id)
        seen_categories.add(category_by_id[candidate_id])
        if len(chosen) == OUT_OF_FIELD_RESUMES:
            return chosen
    return (chosen + [c for c in usable if c not in chosen])[:OUT_OF_FIELD_RESUMES]


def _job_description(
    presentation: RolePresentation, kept: Sequence[Requirement], dropped: Sequence[Requirement]
) -> str:
    bullets = "\n".join(f"- {requirement.text}" for requirement in kept)
    text = f"{presentation.opener}\n\nWhat we need:\n{bullets}\n"
    if dropped:
        also = "; ".join(requirement.text for requirement in dropped)
        text += (
            "\nThe original posting also asked for the following, which this demo "
            f"does not gate on: {also}.\n"
        )
    return text


def _summary(category: str, staged: Sequence[_StagedResume]) -> str:
    """What the card says the batch is. Named by corpus category rather than
    by the Role's title, since that is what the Resumes were labelled as -
    the batch is two Resumes from the field and two from elsewhere."""
    others = " and ".join(s.category for s in staged if not s.in_field)
    return f"Two {category} Resumes against {others}."


def build_preset(
    evaluation_role: EvaluationRole,
    gold_set: Sequence[GoldSetLabel],
    resume_text_by_id: dict[str, str],
) -> dict:
    presentation = ROLE_PRESENTATION[evaluation_role.category]
    reviewed = evaluation_role.role.requirement_set.requirements
    requirements = reviewed + tuple(
        Requirement(id=f"{DEMO_REQUIREMENT_ID_PREFIX}{index}", text=text)
        for index, text in enumerate(presentation.extra_requirements, start=1)
    )

    category_by_id: dict[str, str] = {}
    for label in gold_set:
        category_by_id.setdefault(label.resume.candidate_id, label.resume.category)
    judged_here = {
        label.resume.candidate_id
        for label in gold_set
        if label.evaluation_role_id == evaluation_role.evaluation_role_id
    }

    def pool(in_field: bool) -> list[str]:
        """Candidates for one half of the batch, in preference order: a
        Resume the Gold Set judged against *this* Role first, since a human
        has already read it against this field, then the rest by candidate
        id - so the choice is deterministic and reviewable."""
        return sorted(
            (
                candidate_id
                for candidate_id, category in category_by_id.items()
                if (category == evaluation_role.category) == in_field
            ),
            key=lambda candidate_id: (candidate_id not in judged_here, candidate_id),
        )

    same_field, other_field = pool(True), pool(False)
    met = _met_requirement_ids(
        presentation.label, requirements, same_field + other_field, resume_text_by_id
    )
    in_field_ids, kept_ids = _choose_in_field_pair(same_field, met)
    out_of_field_ids = _choose_out_of_field(other_field, met, kept_ids, category_by_id)

    kept = [r for r in requirements if r.id in kept_ids]
    dropped_from_posting = [r for r in reviewed if r.id not in kept_ids]
    staged = [
        _StagedResume(candidate_id, category_by_id[candidate_id], in_field=is_in_field)
        for is_in_field, group in ((True, in_field_ids), (False, out_of_field_ids))
        for candidate_id in group
    ]
    return {
        "preset_id": _slug(presentation.label),
        "label": presentation.label,
        "family": presentation.family,
        "summary": _summary(evaluation_role.category, staged),
        "role_title": presentation.label,
        "job_description": _job_description(presentation, kept, dropped_from_posting),
        "source_evaluation_role_id": evaluation_role.evaluation_role_id,
        "resumes": [{"candidate_id": s.candidate_id} for s in staged],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-roles-dir", type=Path, default=DEFAULT_EVALUATION_ROLES_DIR)
    parser.add_argument("--gold-set", type=Path, default=DEFAULT_GOLD_SET_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    evaluation_roles = read_evaluation_roles(args.evaluation_roles_dir)
    gold_set = read_gold_set(args.gold_set)
    resume_text_by_id: dict[str, str] = {}
    for label in gold_set:
        resume_text_by_id.setdefault(label.resume.candidate_id, label.resume.text)

    missing = sorted({r.category for r in evaluation_roles} - set(ROLE_PRESENTATION))
    if missing:
        raise SystemExit(
            f"error: no ROLE_PRESENTATION entry for {', '.join(missing)} - add one in "
            f"{__file__} rather than shipping fewer Presets than there are fields."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    for stale in sorted(args.out.glob("*.json")):
        stale.unlink()

    written = 0
    for index, evaluation_role in enumerate(evaluation_roles, start=1):
        try:
            preset = build_preset(evaluation_role, gold_set, resume_text_by_id)
        except CannotBuildPreset as exc:
            print(f"skipped {evaluation_role.category}: {exc}")
            continue
        path = args.out / f"{index:02d}-{preset['preset_id']}.json"
        path.write_text(json.dumps(preset, indent=2, ensure_ascii=False) + "\n")
        written += 1

    print(f"wrote {written} of {len(evaluation_roles)} Demo Presets to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
