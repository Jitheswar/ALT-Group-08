"""Compares the checked-in Gold Set's hand labels against Proxy Relevance
for the same pairs (ticket 08, ADR-0003) and writes the divergence report.

Needs no model client and no corpus download: the Gold Set already carries
its own Resume text, and the checked-in Evaluation Roles already carry their
own category, so this runs entirely from checked-in data.

Usage:
    uv run python scripts/run_gold_set_eval.py \
        --out data/gold_set/gold-set-report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from screening.eval_roles import read_evaluation_roles
from screening.gold_set import compare_gold_set_to_proxy_relevance, read_gold_set


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold-set", type=Path, default=Path("data/gold_set/gold_set.json"))
    parser.add_argument("--evaluation-roles-dir", type=Path, default=Path("data/evaluation_roles/roles"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    if not args.gold_set.exists():
        print(f"error: no Gold Set found at {args.gold_set}", file=sys.stderr)
        return 1

    gold_set = read_gold_set(args.gold_set)
    evaluation_roles = read_evaluation_roles(args.evaluation_roles_dir)
    if not evaluation_roles:
        print(f"error: no Evaluation Roles found under {args.evaluation_roles_dir}", file=sys.stderr)
        return 1

    report = compare_gold_set_to_proxy_relevance(gold_set, evaluation_roles)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(report), indent=2) + "\n")

    print(f"Compared {len(gold_set)} hand-labelled Gold Set pair(s) against Proxy Relevance:")
    print(
        f"  agreement-rate={report.agreement_rate:.1%} "
        f"proxy-false-positive-rate={report.proxy_false_positive_rate:.1%} "
        f"proxy-false-negative-rate={report.proxy_false_negative_rate:.1%}"
    )
    print(f"Wrote Gold Set comparison report to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
