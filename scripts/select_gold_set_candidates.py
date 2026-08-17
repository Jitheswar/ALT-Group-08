"""Selects the roughly 150 Role-and-Resume pairs a human hand-labels into
the Gold Set (ticket 08, ADR-0003). This is the scriptable half of the
process, mirroring `scripts/generate_evaluation_roles.py`'s propose/finalize
split: sampling which pairs to look at can be deterministic, but the
relevance judgement on each one genuinely needs a human and cannot be
scripted away, so this script only writes an unlabelled candidates file for
a human to work through (`scripts/build_gold_set.py` finalize consumes the
result).

Three pairs per Evaluation Role, plus a fourth for a subset, to land at
"roughly 150" from 43 Evaluation Roles (43*3 + 21 = 150):

    same_category      A Resume from the Role's own corpus category -
                        Proxy Relevance says relevant.
    adjacent_category   A Resume from a different but related category
                        (same job family per _FAMILY below) - the case
                        most likely to reveal a Proxy Relevance false
                        negative.
    distant_category    A Resume from an unrelated family - the easy-negative
                        case Proxy Relevance is least likely to get wrong,
                        included so the Gold Set is not composed entirely of
                        hard cases.

Selection within a category is deterministic (a SHA-256 offset keyed by the
Evaluation Role id and slot, not Python's hash()) rather than random, so
re-running this script reproduces the same candidates file, and picks are
tracked in a global used-set so no Resume is ever sampled into two pairs.

Usage:
    uv run --group data python scripts/select_gold_set_candidates.py \
        --resume-atlas-parquet PATH \
        --evaluation-roles-dir data/evaluation_roles/roles \
        --out data/gold_set/.candidates.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from screening.eval_roles import EvaluationRole, read_evaluation_roles
from screening.proxy_relevance import CorpusResume
from screening.resume_atlas import load_resume_atlas_corpus

# Five broad job families covering every resume-atlas category (43 total),
# used only to pick a same-discipline "adjacent" category and a
# different-discipline "distant" category for each Evaluation Role - see
# the module docstring. This grouping is itself a reviewable judgement
# call, not a scored or learned mapping.
_FAMILIES: dict[str, tuple[str, ...]] = {
    "Technology": (
        "Blockchain",
        "Business Analyst",
        "Data Science",
        "Database",
        "DevOps",
        "DotNet Developer",
        "ETL Developer",
        "Information Technology",
        "Java Developer",
        "Network Security Engineer",
        "PMO",
        "Python Developer",
        "React Developer",
        "SAP Developer",
        "SQL Developer",
        "Testing",
    ),
    "Engineering": (
        "Agriculture",
        "Architecture",
        "Automobile",
        "Aviation",
        "Building and Construction",
        "Civil Engineer",
        "Electrical Engineering",
        "Mechanical Engineer",
    ),
    "Business and Finance": (
        "Accountant",
        "Banking",
        "Consultant",
        "Finance",
        "Human Resources",
        "Management",
        "Operations Manager",
        "Public Relations",
        "Sales",
    ),
    "Creative and Media": (
        "Apparel",
        "Arts",
        "Designing",
        "Digital Media",
        "Web Designing",
    ),
    "Service and People": (
        "Advocate",
        "BPO",
        "Education",
        "Food and Beverages",
        "Health and Fitness",
    ),
}

_FAMILY_ORDER = tuple(_FAMILIES)
_FAMILY_OF: dict[str, str] = {
    category: family for family, categories in _FAMILIES.items() for category in categories
}


def adjacent_category(category: str) -> str:
    """The next category (wrapping) in the same family, alphabetical order -
    a different discipline close enough that Proxy Relevance mislabelling it
    as irrelevant would be a real finding.
    """
    family = _FAMILY_OF[category]
    members = sorted(_FAMILIES[family])
    index = members.index(category)
    return members[(index + 1) % len(members)]


def distant_category(category: str) -> str:
    """A category from the family two positions away in _FAMILY_ORDER, at
    the same relative position within it - deliberately unrelated, the
    easy-negative case.
    """
    family = _FAMILY_OF[category]
    members = sorted(_FAMILIES[family])
    index = members.index(category)
    distant_family = _FAMILY_ORDER[(_FAMILY_ORDER.index(family) + 2) % len(_FAMILY_ORDER)]
    distant_members = sorted(_FAMILIES[distant_family])
    return distant_members[index % len(distant_members)]


def _stable_offset(key: str, modulus: int) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest, 16) % modulus


def _picker(corpus_by_category: dict[str, list[CorpusResume]], used: set[str]):
    def pick(category: str, key: str) -> CorpusResume:
        rows = corpus_by_category[category]
        if not rows:
            raise RuntimeError(f"no resume-atlas rows for category {category!r}")
        start = _stable_offset(key, len(rows))
        for offset in range(len(rows)):
            row = rows[(start + offset) % len(rows)]
            if row.candidate_id not in used:
                used.add(row.candidate_id)
                return row
        raise RuntimeError(f"category {category!r} exhausted before a unique Resume was found")

    return pick


def select_candidates(
    evaluation_roles: list[EvaluationRole], corpus: list[CorpusResume]
) -> list[dict]:
    corpus_by_category: dict[str, list[CorpusResume]] = defaultdict(list)
    for row in corpus:
        corpus_by_category[row.category].append(row)

    used: set[str] = set()
    pick = _picker(corpus_by_category, used)

    ordered_roles = sorted(evaluation_roles, key=lambda r: r.evaluation_role_id)
    # The first 21 Evaluation Roles (of 43) get a second same-category pair
    # so the total lands at 43*3 + 21 = 150 - see the module docstring.
    extra_same_category_ids = {r.evaluation_role_id for r in ordered_roles[:21]}

    candidates: list[dict] = []
    for role in ordered_roles:
        requirement_texts = [r.text for r in role.role.requirement_set.requirements]
        slots = [
            ("same_category", role.category, "same:1"),
            ("adjacent_category", adjacent_category(role.category), "adjacent"),
            ("distant_category", distant_category(role.category), "distant"),
        ]
        if role.evaluation_role_id in extra_same_category_ids:
            slots.append(("same_category", role.category, "same:2"))

        for pair_kind, resume_category, slot in slots:
            row = pick(resume_category, f"{role.evaluation_role_id}:{slot}")
            candidates.append(
                {
                    "evaluation_role_id": role.evaluation_role_id,
                    "role_title": role.role.title,
                    "role_category": role.category,
                    "requirements": requirement_texts,
                    "pair_kind": pair_kind,
                    "candidate_id": row.candidate_id,
                    "resume_category": row.category,
                    "resume_text": row.text,
                }
            )
    return candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resume-atlas-parquet", required=True, type=Path)
    parser.add_argument("--evaluation-roles-dir", type=Path, default=Path("data/evaluation_roles/roles"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    evaluation_roles = read_evaluation_roles(args.evaluation_roles_dir)
    if not evaluation_roles:
        print(f"error: no Evaluation Roles found under {args.evaluation_roles_dir}", file=sys.stderr)
        return 1

    missing = set(r.category for r in evaluation_roles) - set(_FAMILY_OF)
    if missing:
        print(f"error: no job family assigned for categories: {sorted(missing)}", file=sys.stderr)
        return 1

    corpus = load_resume_atlas_corpus(args.resume_atlas_parquet)
    candidates = select_candidates(evaluation_roles, corpus)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(candidates, indent=2) + "\n")
    print(f"Wrote {len(candidates)} unlabelled candidate pair(s) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
