"""Demo Presets: checked-in, ready-made inputs for the web demo surface
(ticket 12), so demonstrating the system does not begin with pasting a Job
Description and hunting for Resume files.

A Preset is a filled-in form, not a shortcut past the Recruiter. It
prefills the Job Description and stages a batch of Resumes; extraction,
review, approval and Screening then run exactly as they do for a Recruiter
who typed everything in, and every part of it stays editable. Nothing here
skips the approval step (ADR-0004) or hides a Candidate (ADR-0001).

Resume text is referenced, never duplicated: a Preset names Gold Set
candidate ids and the text is resolved from `data/gold_set/gold_set.json`
at load time, so there is one copy of a Resume in the repo rather than two
that can drift apart. The Gold Set is read strictly as demo material here -
no judgement it carries is consulted, and nothing on this path produces a
reported metric - so using it as a demo source cannot feed back into what
it measures.

The Job Description text on a Preset is hand-authored demo material, which
is why Presets live in `data/demo_presets/` rather than alongside the
Evaluation Roles: an Evaluation Role's Job Description must be a real
posting (ADR-0008), and nothing here may ever be mistaken for one.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from screening.domain import Candidate, Resume
from screening.gold_set import DEFAULT_GOLD_SET_PATH, read_gold_set

DEFAULT_DEMO_PRESETS_DIR = Path("data/demo_presets")


class MalformedDemoPreset(Exception):
    """A Preset file the repo ships is wrong - a missing field, or a
    candidate id no Gold Set Resume carries. Raised rather than skipped:
    an absent Preset directory is a deployment choice, but a broken Preset
    inside one is a bug, and a demo silently missing a Candidate is exactly
    the failure mode ADR-0001 rules out.
    """


@dataclass(frozen=True)
class DemoPreset:
    """One ready-made demo input: the Role title and Job Description that
    prefill step 1, and the Candidates staged for step 3, already resolved
    to their Resume text.

    `family` is the coarse grouping the landing page browses by - one field
    per Preset is too many to lay out flat. `source_evaluation_role_id` is
    the Evaluation Role whose reviewed Requirement Set the Job Description's
    conditions were drawn from, so a Preset stays traceable back to it.
    """

    preset_id: str
    label: str
    family: str
    summary: str
    role_title: str
    job_description: str
    source_evaluation_role_id: str
    candidates: tuple[Candidate, ...]


def load_demo_presets(
    presets_dir: Path = DEFAULT_DEMO_PRESETS_DIR,
    *,
    gold_set_path: Path = DEFAULT_GOLD_SET_PATH,
) -> tuple[DemoPreset, ...]:
    """Every Preset under `presets_dir`, in filename order - the numbered
    filenames are what fixes the order they are offered in.

    Returns no Presets at all when either the Preset directory or the Gold
    Set they resolve their Resumes from is absent, so a deployment carrying
    neither still serves the plain Recruiter flow rather than failing to
    start.
    """
    if not presets_dir.is_dir() or not gold_set_path.is_file():
        return ()

    resume_text_by_candidate_id = {
        label.resume.candidate_id: label.resume.text for label in read_gold_set(gold_set_path)
    }
    return tuple(
        _read_demo_preset(path, resume_text_by_candidate_id)
        for path in sorted(presets_dir.glob("*.json"))
    )


def _read_demo_preset(path: Path, resume_text_by_candidate_id: dict[str, str]) -> DemoPreset:
    data = json.loads(path.read_text())
    try:
        return DemoPreset(
            preset_id=data["preset_id"],
            label=data["label"],
            family=data["family"],
            summary=data["summary"],
            role_title=data["role_title"],
            job_description=data["job_description"],
            source_evaluation_role_id=data["source_evaluation_role_id"],
            candidates=tuple(
                _resolve_candidate(resume["candidate_id"], resume_text_by_candidate_id, path)
                for resume in data["resumes"]
            ),
        )
    except KeyError as exc:
        raise MalformedDemoPreset(f"{path}: missing field {exc}") from exc


def _resolve_candidate(
    candidate_id: str, resume_text_by_candidate_id: dict[str, str], path: Path
) -> Candidate:
    text = resume_text_by_candidate_id.get(candidate_id)
    if text is None:
        raise MalformedDemoPreset(f"{path}: no Gold Set Resume for candidate id {candidate_id!r}")
    return Candidate(id=candidate_id, resume=Resume(text=text))


def group_by_family(presets: Sequence[DemoPreset]) -> list[tuple[str, list[DemoPreset]]]:
    """Presets grouped for browsing, richest family first and each family's
    Presets in the order they were loaded.

    Ordering by how many Presets a family holds puts the widest choice at the
    top of the page and needs nothing recorded alongside the Presets to
    maintain; ties fall back to the family name so the result is stable.
    """
    by_family: dict[str, list[DemoPreset]] = {}
    for preset in presets:
        by_family.setdefault(preset.family, []).append(preset)
    counts = Counter({family: len(group) for family, group in by_family.items()})
    return [
        (family, by_family[family])
        for family, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    ]
