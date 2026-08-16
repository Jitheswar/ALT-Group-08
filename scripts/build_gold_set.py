"""Finalizes the Gold Set (ticket 08, ADR-0003) from a human-produced labels
file and the candidates file `scripts/select_gold_set_candidates.py` wrote,
writing the checked-in `data/gold_set/gold_set.json`.

Mirrors `scripts/generate_evaluation_roles.py`'s propose/finalize split: the
candidates file is what was sampled for a human to look at, unlabelled; the
labels file is what a human produced by applying `data/gold_set/rubric.md`
to each one. This step only pairs the two back together, validates every
sampled pair actually got a label (and nothing extra snuck in), and writes
the result - it does no judging of its own.

The labels file is a JSON array of
{"evaluation_role_id", "candidate_id", "relevant", "justification"} objects,
one per candidate pair, in any order.

Usage:
    uv run python scripts/build_gold_set.py \
        --candidates data/gold_set/.candidates.json \
        --labels data/gold_set/.labels.json \
        --out data/gold_set/gold_set.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from screening.gold_set import GoldSetLabel, write_gold_set
from screening.proxy_relevance import CorpusResume


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    candidates = json.loads(args.candidates.read_text())
    labels = json.loads(args.labels.read_text())

    candidate_by_key = {(c["evaluation_role_id"], c["candidate_id"]): c for c in candidates}
    label_keys = [(l["evaluation_role_id"], l["candidate_id"]) for l in labels]

    duplicate_labels = {key for key in label_keys if label_keys.count(key) > 1}
    if duplicate_labels:
        print(f"error: {len(duplicate_labels)} pair(s) labelled more than once: {sorted(duplicate_labels)}", file=sys.stderr)
        return 1

    missing = set(candidate_by_key) - set(label_keys)
    extra = set(label_keys) - set(candidate_by_key)
    if missing:
        print(f"error: {len(missing)} candidate pair(s) have no label: {sorted(missing)}", file=sys.stderr)
        return 1
    if extra:
        print(f"error: {len(extra)} label(s) reference a pair not in the candidates file: {sorted(extra)}", file=sys.stderr)
        return 1

    gold_set = []
    for label in labels:
        key = (label["evaluation_role_id"], label["candidate_id"])
        candidate = candidate_by_key[key]
        gold_set.append(
            GoldSetLabel(
                evaluation_role_id=label["evaluation_role_id"],
                resume=CorpusResume(
                    candidate_id=candidate["candidate_id"],
                    category=candidate["resume_category"],
                    text=candidate["resume_text"],
                ),
                relevant=bool(label["relevant"]),
                justification=label["justification"],
            )
        )
    # Deterministic order regardless of the labels file's own order, so the
    # checked-in file's diff is stable across a re-finalize.
    gold_set.sort(key=lambda label: (label.evaluation_role_id, label.resume.candidate_id))

    write_gold_set(gold_set, args.out)
    relevant_count = sum(1 for g in gold_set if g.relevant)
    print(
        f"Wrote {len(gold_set)} hand-labelled Gold Set pair(s) to {args.out} "
        f"({relevant_count} relevant, {len(gold_set) - relevant_count} not relevant)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
