"""Demo Presets (ticket 12) are checked-in demo material, so these tests
hold the shipped files themselves to the promises the web UI makes about
them - that every Preset resolves, and that each one actually demonstrates
both stages rather than only Screening.
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from screening.core import run_screening
from screening.demo_presets import (
    DEFAULT_DEMO_PRESETS_DIR,
    MalformedDemoPreset,
    group_by_family,
    load_demo_presets,
)
from screening.domain import (
    JobDescription,
    Qualified,
    Role,
    approve_requirement_set,
)
from screening.extraction import extract_requirements
from screening.gold_set import DEFAULT_GOLD_SET_PATH
from screening.scripted_client import ScriptedModelClient


@pytest.fixture(scope="module")
def shipped_presets():
    presets = load_demo_presets()
    assert presets, "the repo ships Demo Presets; none were loaded"
    return presets


def test_every_shipped_preset_resolves_its_resumes(shipped_presets):
    for preset in shipped_presets:
        assert preset.candidates, f"{preset.preset_id} stages no Resumes"
        assert all(candidate.resume.text.strip() for candidate in preset.candidates)


def test_preset_ids_are_unique(shipped_presets):
    ids = [preset.preset_id for preset in shipped_presets]
    assert len(ids) == len(set(ids))


def test_each_preset_shows_both_screening_and_ranking(shipped_presets):
    """A Preset whose whole batch fails Screening demonstrates half the
    system: no Fit judgement, no comparative pass, nothing to rank. Each
    shipped Preset is chosen so its Shortlist carries at least two Qualified
    Candidates - enough for the comparative pass to have something to say -
    and at least one Disqualified one, so the Recruiter also sees a missed
    Requirement cited.
    """
    for preset in shipped_presets:
        _assert_shows_both_stages(preset)


def _assert_shows_both_stages(preset) -> None:
    model_client = ScriptedModelClient()
    requirements = extract_requirements(
        JobDescription(text=preset.job_description), model_client
    )
    role = Role(
        id=str(uuid4()),
        title=preset.role_title,
        requirement_set=approve_requirement_set(requirements),
    )

    shortlist = run_screening(role, list(preset.candidates), model_client)

    qualified = [e for e in shortlist.entries if isinstance(e.outcome, Qualified)]
    assert len(qualified) >= 2, f"{preset.preset_id}: {len(qualified)} Qualified"
    assert len(qualified) < len(shortlist.entries), f"{preset.preset_id}: none Disqualified"
    assert shortlist.comparisons, f"{preset.preset_id}: comparative pass said nothing"


def test_a_preset_job_description_states_requirements_extraction_can_find(shipped_presets):
    """Extraction reads a Job Description's list lines, so a Preset whose
    text states no Requirements as a list would land the Recruiter on an
    empty review page."""
    for preset in shipped_presets:
        proposed = extract_requirements(
            JobDescription(text=preset.job_description), ScriptedModelClient()
        )
        assert len(proposed) >= 3, f"{preset.preset_id} proposed {len(proposed)} Requirements"


def test_no_presets_are_offered_when_the_directory_is_absent(tmp_path):
    assert load_demo_presets(tmp_path / "nothing-here") == ()


def test_no_presets_are_offered_when_the_gold_set_is_absent(tmp_path):
    assert (
        load_demo_presets(DEFAULT_DEMO_PRESETS_DIR, gold_set_path=tmp_path / "no-gold-set.json")
        == ()
    )


def test_a_preset_naming_an_unknown_resume_fails_loudly(tmp_path):
    """A Preset silently losing a Candidate is the failure mode ADR-0001
    rules out, so a candidate id the Gold Set does not carry is an error
    rather than a skipped row."""
    presets_dir = tmp_path / "demo_presets"
    presets_dir.mkdir()
    (presets_dir / "01-broken.json").write_text(
        json.dumps(
            {
                "preset_id": "broken",
                "label": "Broken",
                "family": "Software and Data",
                "summary": "Names a Resume that does not exist.",
                "role_title": "Broken Role",
                "job_description": "- Something\n",
                "source_evaluation_role_id": "eval-role-99",
                "resumes": [{"candidate_id": "resume-atlas-does-not-exist"}],
            }
        )
    )

    with pytest.raises(MalformedDemoPreset, match="resume-atlas-does-not-exist"):
        load_demo_presets(presets_dir, gold_set_path=DEFAULT_GOLD_SET_PATH)


def test_a_preset_missing_a_field_fails_loudly(tmp_path):
    presets_dir = tmp_path / "demo_presets"
    presets_dir.mkdir()
    (presets_dir / "01-incomplete.json").write_text(json.dumps({"preset_id": "incomplete"}))

    with pytest.raises(MalformedDemoPreset, match="missing field"):
        load_demo_presets(presets_dir, gold_set_path=DEFAULT_GOLD_SET_PATH)


def test_the_shipped_presets_cover_the_breadth_of_the_corpus(shipped_presets):
    """The point of generating a Preset per Evaluation Role rather than
    hand-picking a few: a demo can open on whatever field the audience cares
    about. A regeneration that silently drops most fields - a selection rule
    that got stricter, say - should fail here rather than ship a demo with
    four Presets in it."""
    assert len(shipped_presets) >= 35, f"only {len(shipped_presets)} Presets"
    families = group_by_family(shipped_presets)
    assert len(families) >= 5
    assert all(group for _, group in families)


def test_families_are_ordered_richest_first(shipped_presets):
    sizes = [len(group) for _, group in group_by_family(shipped_presets)]
    assert sizes == sorted(sizes, reverse=True)


def test_every_preset_names_the_evaluation_role_its_requirements_came_from(shipped_presets):
    """A Preset's conditions are drawn from a reviewed Requirement Set, not
    invented for the demo, so each one stays traceable back to the
    Evaluation Role that holds them."""
    for preset in shipped_presets:
        assert preset.source_evaluation_role_id.startswith("eval-role-")
